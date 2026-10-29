package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"golang.org/x/net/websocket"
	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
)

func newAudioChatRunner(t *testing.T, handle func(*websocket.Conn)) *llamaServerRunner {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			fmt.Fprint(w, `{"status":"ok"}`)
		case "/v1/realtime":
			websocket.Handler(handle).ServeHTTP(w, r)
		default:
			t.Errorf("unexpected endpoint %s: native audio must never call completion or TTS", r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	return &llamaServerRunner{
		port:   srv.Listener.Addr().(*net.TCPAddr).Port,
		client: srv.Client(),
		cmd:    fakeRunningCmd(),
		sem:    semaphore.NewWeighted(1),
		launch: llamaServerLaunchConfig{
			projectors: []string{"mmproj"},
			config:     LlamaServerConfig{NativeAudio: true},
		},
	}
}

func realtimeSession(t *testing.T, c *websocket.Conn, kind string, rate int) {
	t.Helper()
	c.SetDeadline(time.Now().Add(5 * time.Second))
	if err := websocket.Message.Send(c, fmt.Sprintf(`{"type":%q,"session":{"audio":{"output":{"format":{"type":"audio/pcm","rate":%d}}}}}`, kind, rate)); err != nil {
		t.Error(err)
	}
}

func receiveRealtime(t *testing.T, c *websocket.Conn, want string) {
	t.Helper()
	var got, expected any
	if err := websocket.JSON.Receive(c, &got); err != nil {
		t.Error(err)
		return
	}
	if err := json.Unmarshal([]byte(want), &expected); err != nil {
		t.Error(err)
		return
	}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("event %#v, want %#v", got, expected)
	}
}

func negotiateRealtime(t *testing.T, c *websocket.Conn) {
	t.Helper()
	realtimeSession(t, c, "session.created", 24000)
	receiveRealtime(t, c, `{"type":"session.update","session":{"type":"realtime","instructions":"Antworte kurz auf Deutsch","output_modalities":["text","audio"],"max_output_tokens":512}}`)
	realtimeSession(t, c, "session.updated", 24000)
	receiveRealtime(t, c, `{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"Warum ist der Himmel blau?"}]}}`)
	receiveRealtime(t, c, `{"type":"response.create","response":{"output_modalities":["text","audio"]}}`)
}

var audioChatTurn = AudioChatRequest{Prompt: "Warum ist der Himmel blau?", System: "Antworte kurz auf Deutsch", MaxTokens: 512}

func TestNativeAudioChatPingAfterSendDeadline(t *testing.T) {
	r := newAudioChatRunner(t, func(c *websocket.Conn) {
		negotiateRealtime(t, c)
		websocket.Message.Send(c, `{"type":"response.output_audio.delta","delta":"AQA="}`)
		c.PayloadType = websocket.PingFrame
		if _, err := c.Write([]byte("ping")); err != nil {
			t.Error(err)
		}
		websocket.Message.Send(c, `{"type":"response.done","response":{"status":"completed"}}`)
	})
	c, err := websocket.Dial(fmt.Sprintf("ws://127.0.0.1:%d/v1/realtime", r.port), "", "http://localhost")
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	err = streamRealtimeAudio(t.Context(), c, audioChatTurn, func(api.AudioChunk) error {
		// Simulate playback taking longer than the last explicit send's timeout.
		return c.SetWriteDeadline(time.Now().Add(-time.Second))
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestNativeAudioChatStreamsOneGeneration(t *testing.T) {
	first := make(chan struct{})
	r := newAudioChatRunner(t, func(c *websocket.Conn) {
		negotiateRealtime(t, c)
		websocket.Message.Send(c, `{"type":"response.output_text.delta","delta":"The atmosphere scatters sunlight."}`)
		websocket.Message.Send(c, `{"type":"response.output_audio.delta","delta":"AQACAA=="}`)
		select {
		case <-first:
		case <-time.After(3 * time.Second):
			t.Error("audio buffered until generation ended")
			return
		}
		websocket.Message.Send(c, `{"type":"response.output_audio.delta","delta":"AwA="}`)
		websocket.Message.Send(c, `{"type":"response.done","response":{"status":"completed"}}`)
	})
	var pcm []byte
	err := r.AudioChat(t.Context(), audioChatTurn, func(chunk api.AudioChunk) error {
		if chunk.Format != "pcm16" || chunk.SampleRate != 24000 || chunk.Channels != 1 {
			t.Errorf("invalid metadata %+v", chunk)
		}
		if len(pcm) == 0 {
			close(first)
		}
		pcm = append(pcm, chunk.Data...)
		return nil
	})
	if err != nil || !bytes.Equal(pcm, []byte{1, 0, 2, 0, 3, 0}) {
		t.Fatalf("audio %v, error %v", pcm, err)
	}
	if !r.sem.TryAcquire(1) {
		t.Fatal("runner semaphore leaked")
	}
	r.sem.Release(1)
}

func TestNativeAudioChatFailures(t *testing.T) {
	for _, tt := range []struct{ name, event, want string }{
		{"backend", `{"type":"error","error":{"message":"codec failed"}}`, "codec failed"},
		{"base64", `{"type":"response.output_audio.delta","delta":"!"}`, "invalid PCM16"},
		{"sample", `{"type":"response.output_audio.delta","delta":"AA=="}`, "invalid PCM16"},
		{"empty", `{"type":"response.done","response":{"status":"completed"}}`, "no audio"},
		{"cancelled", `{"type":"response.done","response":{"status":"cancelled"}}`, "cancelled"},
		{"eof", "", "before response.done"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := newAudioChatRunner(t, func(c *websocket.Conn) {
				negotiateRealtime(t, c)
				if tt.event != "" {
					websocket.Message.Send(c, tt.event)
				}
			})
			err := r.AudioChat(t.Context(), audioChatTurn, func(api.AudioChunk) error { t.Error("unexpected audio"); return nil })
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("got %v, want %s", err, tt.want)
			}
		})
	}
	t.Run("format", func(t *testing.T) {
		r := newAudioChatRunner(t, func(c *websocket.Conn) { realtimeSession(t, c, "session.created", 48000) })
		if err := r.AudioChat(t.Context(), audioChatTurn, nil); err == nil || !strings.Contains(err.Error(), "unsupported") {
			t.Fatalf("got %v", err)
		}
	})
	t.Run("unsupported runner configuration", func(t *testing.T) {
		r := &llamaServerRunner{}
		if err := r.AudioChat(t.Context(), audioChatTurn, nil); err == nil {
			t.Fatal("accepted unsupported model")
		}
	})
}

func TestNativeAudioChatCancellation(t *testing.T) {
	for _, callbackError := range []bool{false, true} {
		t.Run(fmt.Sprint(callbackError), func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			closed := make(chan struct{})
			sentinel := errors.New("client write failed")
			r := newAudioChatRunner(t, func(c *websocket.Conn) {
				negotiateRealtime(t, c)
				if callbackError {
					websocket.Message.Send(c, `{"type":"response.output_audio.delta","delta":"AAA="}`)
				} else {
					cancel()
				}
				var extra string
				if err := websocket.Message.Receive(c, &extra); err == nil {
					t.Errorf("unexpected second generation: %s", extra)
				}
				close(closed)
			})
			err := r.AudioChat(ctx, audioChatTurn, func(api.AudioChunk) error { return sentinel })
			want := error(context.Canceled)
			if callbackError {
				want = sentinel
			}
			if !errors.Is(err, want) {
				t.Fatalf("got %v, want %v", err, want)
			}
			select {
			case <-closed:
			case <-time.After(3 * time.Second):
				t.Fatal("upstream connection leaked")
			}
			if !r.sem.TryAcquire(1) {
				t.Fatal("runner slot leaked")
			}
			r.sem.Release(1)
		})
	}
}

func TestNativeAudioChatCancelWhileQueued(t *testing.T) {
	r := &llamaServerRunner{sem: semaphore.NewWeighted(1), launch: llamaServerLaunchConfig{projectors: []string{"mmproj"}, config: LlamaServerConfig{NativeAudio: true}}}
	r.sem.Acquire(t.Context(), 1)
	defer r.sem.Release(1)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err := r.AudioChat(ctx, audioChatTurn, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v", err)
	}
}
