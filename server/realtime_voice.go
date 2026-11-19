package server

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"sync/atomic"
	"time"

	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
	"github.com/zaf/g711"
	"golang.org/x/net/websocket"
)

type voiceCall struct {
	peer     *webrtc.PeerConnection
	output   *webrtc.TrackLocalStaticSample
	input    chan *webrtc.TrackRemote
	speaking atomic.Bool
}

func newVoiceCall(ctx context.Context, cancel context.CancelFunc, offer webrtc.SessionDescription) (*voiceCall, error) {
	codec := webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypePCMU, ClockRate: 8000}
	var codecs webrtc.MediaEngine
	if err := codecs.RegisterCodec(webrtc.RTPCodecParameters{RTPCodecCapability: codec, PayloadType: 0}, webrtc.RTPCodecTypeAudio); err != nil {
		return nil, err
	}
	var settings webrtc.SettingEngine
	settings.SetIncludeLoopbackCandidate(true)
	peer, err := webrtc.NewAPI(webrtc.WithMediaEngine(&codecs), webrtc.WithSettingEngine(settings)).NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		return nil, err
	}
	ok := false
	defer func() {
		if !ok {
			peer.Close()
		}
	}()
	output, err := webrtc.NewTrackLocalStaticSample(codec, "voice", "ollama")
	if err != nil {
		return nil, err
	}
	call := &voiceCall{peer: peer, output: output, input: make(chan *webrtc.TrackRemote, 1)}
	peer.OnConnectionStateChange(func(state webrtc.PeerConnectionState) {
		switch state {
		case webrtc.PeerConnectionStateClosed, webrtc.PeerConnectionStateFailed, webrtc.PeerConnectionStateDisconnected:
			cancel()
		}
	})
	peer.OnTrack(func(t *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		select {
		case call.input <- t:
		case <-ctx.Done():
		}
	})
	if err = peer.SetRemoteDescription(offer); err != nil {
		return nil, err
	}
	sender, err := peer.AddTrack(output)
	if err != nil {
		return nil, err
	}
	go func() {
		buf := make([]byte, 1500)
		for {
			if _, _, err := sender.Read(buf); err != nil {
				return
			}
		}
	}()
	answer, err := peer.CreateAnswer(nil)
	if err != nil {
		return nil, err
	}
	gathered := webrtc.GatheringCompletePromise(peer)
	if err = peer.SetLocalDescription(answer); err != nil {
		return nil, err
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(10 * time.Second):
		return nil, errors.New("ICE gathering timed out")
	case <-gathered:
	}
	ok = true
	return call, nil
}

func configureVoice(ws *websocket.Conn, instructions string, maxTokens int) error {
	if maxTokens <= 0 {
		maxTokens = 2048
	}
	for _, want := range []string{"session.created", "session.updated"} {
		var event struct {
			Type    string
			Session struct {
				Audio struct {
					Input, Output struct {
						Format struct {
							Type string
							Rate int
						}
					}
				}
			}
		}
		ws.SetDeadline(time.Now().Add(30 * time.Second))
		if err := websocket.JSON.Receive(ws, &event); err != nil {
			return err
		}
		a := event.Session.Audio
		if event.Type != want || a.Input.Format.Type != "audio/pcm" || a.Input.Format.Rate != 16000 || a.Output.Format.Type != "audio/pcm" || a.Output.Format.Rate != 24000 {
			return fmt.Errorf("expected %s with 16 kHz input and 24 kHz output, received %s", want, event.Type)
		}
		if want == "session.created" {
			if err := websocket.JSON.Send(ws, map[string]any{"type": "session.update", "session": map[string]any{
				"type": "realtime", "instructions": instructions, "output_modalities": []string{"text", "audio"}, "max_output_tokens": maxTokens,
			}}); err != nil {
				return err
			}
		}
	}
	return ws.SetDeadline(time.Time{})
}

// Both directions share the runner session so follow-up questions retain context.
func (c *voiceCall) run(ctx context.Context, cancel context.CancelFunc, ws *websocket.Conn) error {
	var track *webrtc.TrackRemote
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(30 * time.Second):
		return errors.New("microphone connection timed out")
	case track = <-c.input:
	}
	go func() {
		defer cancel()
		if err := c.listen(ctx, track, func(event any) error { return websocket.JSON.Send(ws, event) }); err != nil && ctx.Err() == nil {
			slog.Debug("realtime microphone ended", "error", err)
		}
	}()
	var produced bool
	var pending []byte
	var packet []byte
	next := time.Now()
	sendPacket := func() error {
		if wait := time.Until(next); wait > 0 {
			timer := time.NewTimer(wait)
			defer timer.Stop()
			select {
			case <-timer.C:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		duration := time.Duration(len(packet)) * time.Second / 8000
		if err := c.output.WriteSample(media.Sample{Data: packet, Duration: duration}); err != nil {
			return err
		}
		if now := time.Now(); next.Before(now.Add(-20 * time.Millisecond)) {
			next = now
		}
		next = next.Add(duration)
		packet = packet[:0]
		return nil
	}
	for {
		var event struct {
			Type     string
			Delta    json.RawMessage
			Error    struct{ Message string }
			Response struct{ Status string }
		}
		if err := websocket.JSON.Receive(ws, &event); err != nil {
			return err
		}
		switch event.Type {
		case "error":
			return errors.New(event.Error.Message)
		case "response.output_audio.delta":
			var pcm []byte
			if err := json.Unmarshal(event.Delta, &pcm); err != nil {
				return err
			}
			if len(pcm)%2 != 0 {
				return errors.New("invalid PCM16 audio")
			}
			produced = produced || len(pcm) > 0
			pending = append(pending, pcm...)
			for len(pending) >= 6 {
				// Low-pass each group of three 24 kHz samples before decimation.
				sum := int(int16(binary.LittleEndian.Uint16(pending))) + int(int16(binary.LittleEndian.Uint16(pending[2:]))) + int(int16(binary.LittleEndian.Uint16(pending[4:])))
				packet = append(packet, g711.EncodeUlawFrame(int16(sum/3)))
				pending = pending[6:]
				if len(packet) == 160 {
					if err := sendPacket(); err != nil {
						return err
					}
				}
			}
		case "response.done":
			if event.Response.Status != "completed" {
				return fmt.Errorf("realtime response ended with status %q", event.Response.Status)
			}
			if len(packet) > 0 {
				if err := sendPacket(); err != nil {
					return err
				}
			}
			if !produced {
				return errors.New("realtime model produced no audio")
			}
			produced = false
			pending = nil
			c.speaking.Store(false)
		}
	}
}

// PCMU supplies one sample per byte at 8 kHz. A short history retains the
// initial consonant; a pause or a 30-second turn limit starts the response.
func (c *voiceCall) listen(ctx context.Context, track *webrtc.TrackRemote, send func(any) error) error {
	var history, chunk []byte
	var active, quiet int
	var previous int16
	var last uint16
	haveSeq := false
	flush := func() error {
		if len(chunk) == 0 {
			return nil
		}
		err := send(map[string]any{"type": "input_audio_buffer.append", "audio": chunk})
		chunk = chunk[:0]
		return err
	}
	for ctx.Err() == nil {
		p, _, err := track.ReadRTP()
		if err != nil {
			return err
		}
		if haveSeq && int16(p.SequenceNumber-last) <= 0 {
			continue
		}
		last, haveSeq = p.SequenceNumber, true
		if c.speaking.Load() || len(p.Payload) == 0 {
			continue
		}
		power := int64(0)
		for _, b := range p.Payload {
			v := int64(g711.DecodeUlawFrame(b))
			power += v * v
		}
		voiced := power > int64(len(p.Payload))*350*350
		if active == 0 && !voiced {
			history = append(history, p.Payload...)
			if len(history) > 800 {
				history = history[len(history)-800:]
			}
			continue
		}
		data := p.Payload
		if active == 0 {
			data = append(history, p.Payload...)
			history = nil
		}
		for _, b := range data {
			v := g711.DecodeUlawFrame(b)
			chunk = binary.LittleEndian.AppendUint16(chunk, uint16((int(previous)+int(v))/2))
			chunk = binary.LittleEndian.AppendUint16(chunk, uint16(v))
			previous = v
		}
		active += len(data)
		if voiced {
			quiet = 0
		} else {
			quiet += len(data)
		}
		end := quiet >= 4800 || active >= 30*8000
		if end || len(chunk) >= 3200 {
			if err := flush(); err != nil {
				return err
			}
		}
		if end {
			c.speaking.Store(true)
			if err := send(map[string]string{"type": "input_audio_buffer.commit"}); err != nil {
				return err
			}
			if err := send(map[string]string{"type": "response.create"}); err != nil {
				return err
			}
			active, quiet, previous = 0, 0, 0
		}
	}
	return ctx.Err()
}
