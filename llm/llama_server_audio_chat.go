package llm

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"net"
	"net/http"
	"time"

	"golang.org/x/net/websocket"

	"github.com/ollama/ollama/api"
)

var _ AudioChatter = (*llamaServerRunner)(nil)

// AudioChat uses b11963's native realtime conversation. There is exactly one
// response.create; neither Completion nor Speech (/tts) is involved.
func (s *llamaServerRunner) AudioChat(ctx context.Context, req AudioChatRequest, emit func(api.AudioChunk) error) error {
	if !s.launch.config.NativeAudio || s.launch.config.DisableJinja || len(s.launch.projectors) == 0 {
		return api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: "native audio requires LFM3-Omni GGUF with its audio codec projector and native chat templates"}
	}
	if err := s.sem.Acquire(ctx, 1); err != nil {
		return err
	}
	defer s.sem.Release(1)
	status, err := s.getServerStatusRetry(ctx)
	if err != nil {
		return err
	}
	if status != ServerStatusReady {
		return fmt.Errorf("unexpected server status: %s", status)
	}
	config, err := websocket.NewConfig(fmt.Sprintf("ws://127.0.0.1:%d/v1/realtime", s.port), "http://localhost")
	if err != nil {
		return err
	}
	config.Dialer = &net.Dialer{Timeout: 10 * time.Second}
	handshakeCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	conn, err := config.DialContext(handshakeCtx)
	cancel()
	if err != nil {
		return fmt.Errorf("llama-server native audio requires /v1/realtime (rebuild the pinned llama.cpp payload): %w", err)
	}
	// Bound both memory and inactivity. Closing the upstream connection cancels
	// its realtime task and releases its slot, including on a blocked read.
	conn.MaxPayloadBytes = 1 << 20
	closeConn := func() {
		conn.SetDeadline(time.Now())
		conn.Close()
	}
	defer closeConn()
	stop := context.AfterFunc(ctx, closeConn)
	defer stop()
	return streamRealtimeAudio(ctx, conn, req, emit)
}

type realtimeAudioEvent struct {
	Type    string `json:"type"`
	Delta   string `json:"delta"`
	Session struct {
		Audio struct {
			Output struct {
				Format struct {
					Type string `json:"type"`
					Rate int    `json:"rate"`
				} `json:"format"`
			} `json:"output"`
		} `json:"audio"`
	} `json:"session"`
	Error struct {
		Message string `json:"message"`
	} `json:"error"`
	Response struct {
		Status string `json:"status"`
	} `json:"response"`
}

func streamRealtimeAudio(ctx context.Context, conn *websocket.Conn, req AudioChatRequest, emit func(api.AudioChunk) error) error {
	// Select Omni's native interleaved text/audio response mode. Both modalities
	// belong to one decoding loop; text deltas are discarded, never sent to a
	// separate speech request.
	modalities := []string{"text", "audio"}
	send := func(event any) error {
		if err := conn.SetWriteDeadline(time.Now().Add(30 * time.Second)); err != nil {
			return err
		}
		return websocket.JSON.Send(conn, event)
	}
	read := func() (realtimeAudioEvent, error) {
		var event realtimeAudioEvent
		// Receive also writes automatic pong frames. Refresh both deadlines so
		// a ping during a long response cannot inherit an expired send deadline.
		if err := conn.SetDeadline(time.Now().Add(5 * time.Minute)); err != nil {
			return event, err
		}
		if err := websocket.JSON.Receive(conn, &event); err != nil {
			if ctx.Err() != nil {
				return event, ctx.Err()
			}
			return event, fmt.Errorf("llama-server realtime stream ended before response.done: %w", err)
		}
		if event.Type == "error" {
			return event, fmt.Errorf("llama-server realtime: %s", event.Error.Message)
		}
		return event, nil
	}
	// Negotiate before submitting the user turn, so unsupported backends fail
	// without accidentally producing a text-only response.
	for _, want := range []string{"session.created", "session.updated"} {
		event, err := read()
		if err != nil {
			return err
		}
		if event.Type != want {
			return fmt.Errorf("llama-server realtime: expected %s, got %s", want, event.Type)
		}
		format := event.Session.Audio.Output.Format
		if format.Type != "audio/pcm" || format.Rate != SpeechSampleRate {
			return fmt.Errorf("llama-server native audio format %q at %d Hz is unsupported; need PCM16 mono at %d Hz", format.Type, format.Rate, SpeechSampleRate)
		}
		if want == "session.created" {
			if err := send(map[string]any{"type": "session.update", "session": map[string]any{
				"type": "realtime", "instructions": req.System, "output_modalities": modalities, "max_output_tokens": req.MaxTokens,
			}}); err != nil {
				return err
			}
		}
	}
	if err := send(map[string]any{"type": "conversation.item.create", "item": map[string]any{
		"type": "message", "role": "user", "content": []map[string]string{{"type": "input_text", "text": req.Prompt}},
	}}); err != nil {
		return err
	}
	if err := send(map[string]any{"type": "response.create", "response": map[string]any{"output_modalities": modalities}}); err != nil {
		return err
	}
	produced := false
	for {
		event, err := read()
		if err != nil {
			return err
		}
		switch event.Type {
		case "response.output_audio.delta":
			pcm, err := base64.StdEncoding.DecodeString(event.Delta)
			if err != nil || len(pcm)%2 != 0 {
				return errors.New("llama-server realtime returned invalid PCM16 audio")
			}
			if len(pcm) == 0 {
				continue
			}
			produced = true
			if err := emit(api.AudioChunk{Data: pcm, Format: "pcm16", SampleRate: SpeechSampleRate, Channels: 1}); err != nil {
				return err
			}
		case "response.done":
			if event.Response.Status != "completed" {
				return fmt.Errorf("llama-server audio response ended with status %q", event.Response.Status)
			}
			if !produced {
				return errors.New("native audio model produced no audio")
			}
			return nil
		}
	}
}
