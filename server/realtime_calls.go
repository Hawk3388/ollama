package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"net"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/pion/webrtc/v4"
	"golang.org/x/net/websocket"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

type realtimeRunner interface {
	WithRealtime(context.Context, func(*websocket.Conn) error) error
}

type callOptions struct {
	Type         string `json:"type"`
	Model        string `json:"model"`
	Instructions string `json:"instructions"`
}

// Accept a raw SDP offer or the SDP/session multipart form used by WebRTC clients.
func readCallOffer(w http.ResponseWriter, r *http.Request) (webrtc.SessionDescription, callOptions, error) {
	r.Body = http.MaxBytesReader(w, r.Body, 256<<10)
	options := callOptions{Model: r.URL.Query().Get("model")}
	offer := webrtc.SessionDescription{Type: webrtc.SDPTypeOffer}
	kind, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type"))
	switch kind {
	case "application/sdp":
		data, err := io.ReadAll(r.Body)
		if err != nil {
			return offer, options, err
		}
		offer.SDP = string(data)
	case "multipart/form-data":
		if err := r.ParseMultipartForm(256 << 10); err != nil {
			return offer, options, err
		}
		defer r.MultipartForm.RemoveAll()
		offer.SDP = r.PostForm.Get("sdp")
		if session := r.PostForm.Get("session"); session != "" {
			dec := json.NewDecoder(strings.NewReader(session))
			dec.DisallowUnknownFields()
			if err := dec.Decode(&options); err != nil {
				return offer, options, err
			}
			if dec.Decode(new(any)) != io.EOF {
				return offer, options, errors.New("invalid session JSON")
			}
		}
	default:
		return offer, options, errors.New("expected application/sdp or multipart/form-data")
	}
	if options.Type != "" && options.Type != "realtime" {
		return offer, options, errors.New("expected a realtime session")
	}
	if options.Model == "" {
		return offer, options, errors.New("specify a model in the query or session")
	}
	desc, err := offer.Unmarshal()
	if err != nil {
		return offer, options, err
	}
	audio := 0
	for _, m := range desc.MediaDescriptions {
		if m.MediaName.Media == "application" {
			continue
		}
		if m.MediaName.Media != "audio" || !slices.Contains(m.MediaName.Formats, "0") || m.MediaName.Port.Value == 0 {
			return offer, options, errors.New("offer must support bidirectional PCMU audio (payload type 0)")
		}
		for _, dir := range []string{"inactive", "sendonly", "recvonly"} {
			if _, ok := m.Attribute(dir); ok {
				return offer, options, errors.New("sendrecv audio is required")
			}
		}
		audio++
	}
	if audio != 1 {
		return offer, options, errors.New("one microphone track is required")
	}
	return offer, options, nil
}

func (s *Server) realtimeCallHandler(c *gin.Context) {
	host, _, _ := net.SplitHostPort(c.Request.RemoteAddr)
	if !net.ParseIP(host).IsLoopback() {
		c.JSON(403, gin.H{"error": "realtime calls currently require a local client"})
		return
	}
	offer, options, err := readCallOffer(c.Writer, c.Request)
	if err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	ref, err := parseAndValidateModelRef(options.Model)
	if err != nil {
		writeModelRefParseError(c, err, 400, "model is required")
		return
	}
	if ref.Source == modelSourceCloud {
		c.JSON(400, gin.H{"error": "realtime calls require a local model"})
		return
	}
	m, err := s.getExistingModelForRunner(ref.Name.String(), manifest.RunnerLlamaCPP)
	if err != nil {
		handleScheduleError(c, options.Model, err)
		return
	}
	if !m.supportsNativeAudio() || m.Config.RemoteHost != "" {
		c.JSON(400, gin.H{"error": "realtime calls require LFM3-Omni GGUF with its audio projector"})
		return
	}
	if options.Instructions == "" {
		options.Instructions = m.System
	}

	// The call outlives the signaling POST. Until the SDP answer is ready,
	// a disconnected HTTP client must still cancel loading and negotiation.
	ctx, cancel := context.WithTimeout(context.WithoutCancel(c.Request.Context()), 30*time.Minute)
	stop := context.AfterFunc(c.Request.Context(), cancel)
	defer stop()
	handedOff := false
	defer func() {
		if !handedOff {
			cancel()
		}
	}()
	r, _, opts, err := s.scheduleRunnerForModel(ctx, m, []model.Capability{model.CapabilityCompletion, model.CapabilityAudioOutput}, nil, nil, nil)
	if err != nil {
		handleScheduleError(c, options.Model, err)
		return
	}
	runner, ok := r.(realtimeRunner)
	if !ok {
		c.JSON(400, gin.H{"error": "runner does not support realtime calls"})
		return
	}

	call, err := newVoiceCall(ctx, cancel, offer)
	if err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	answer := call.peer.LocalDescription().SDP
	startup := make(chan error, 1)
	go func() {
		defer cancel()
		defer call.peer.Close()
		configured := false
		err := runner.WithRealtime(ctx, func(ws *websocket.Conn) error {
			if err := configureVoice(ws, options.Instructions, opts.NumPredict); err != nil {
				return err
			}
			configured = true
			startup <- nil
			return call.run(ctx, cancel, ws)
		})
		if !configured {
			startup <- err
		}
		if err != nil && ctx.Err() == nil {
			slog.Debug("realtime call ended", "error", err)
		}
	}()
	select {
	case err := <-startup:
		if err != nil {
			c.JSON(http.StatusBadGateway, gin.H{"error": fmt.Sprintf("starting realtime call: %v", err)})
			return
		}
	case <-c.Request.Context().Done():
		return
	}
	if !stop() || ctx.Err() != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "call was canceled during setup"})
		return
	}
	c.Header("Content-Type", "application/sdp")
	c.Status(http.StatusCreated)
	_, err = io.WriteString(c.Writer, answer)
	handedOff = err == nil
}
