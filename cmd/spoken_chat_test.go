package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlxrunner/model/audio"
	"github.com/ollama/ollama/types/model"
)

func TestRunSpokenChat(t *testing.T) {
	for _, tc := range []struct {
		name, voice string
		flags       []string
	}{
		{"speak", "male-german", []string{"--speak", "--voice", "male-german"}},
		{"default voice", "", []string{"--speak"}},
		{"legacy alias", "male-german", []string{"--audio", "--voice", "male-german"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const prompt = "Warum ist der Himmel blau?"
			const answer = "Der Himmel ist blau. Licht wird gestreut."
			spoken := make(chan string, 4)
			firstSpeech := make(chan struct{})
			var speechCalls atomic.Int32
			var chats atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/show":
					json.NewEncoder(w).Encode(api.ShowResponse{Capabilities: []model.Capability{model.CapabilityCompletion, model.CapabilityAudioOutput, model.CapabilitySpeech}})
				case "/api/chat":
					chats.Add(1)
					var req api.ChatRequest
					if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
						t.Error(err)
					}
					if req.Model != "lfm3-omni" || req.Runner != "llamacpp" || len(req.Messages) != 1 || req.Messages[0].Content != prompt || req.Audio != nil || len(req.OutputModalities) != 1 || req.OutputModalities[0] != "text" || req.Think == nil || req.Think.Bool() || req.KeepAlive == nil || req.KeepAlive.Duration != time.Minute {
						t.Errorf("unexpected chat request: %#v", req)
					}
					enc := json.NewEncoder(w)
					enc.Encode(api.ChatResponse{Message: api.Message{Thinking: "Do not speak this."}})
					enc.Encode(api.ChatResponse{Message: api.Message{Content: "Der Himmel ist blau. "}})
					w.(http.Flusher).Flush()
					select {
					case <-firstSpeech:
					case <-time.After(5 * time.Second):
						t.Error("TTS did not start while text generation was still in progress")
					}
					enc.Encode(api.ChatResponse{Message: api.Message{Content: "Licht wird gestreut."}})
					enc.Encode(api.ChatResponse{Done: true})
				case "/v1/audio/speech":
					if speechCalls.Add(1) == 1 {
						close(firstSpeech)
					}
					var req map[string]string
					if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
						t.Error(err)
					}
					if req["model"] != "lfm3-omni" || req["voice"] != tc.voice || req["response_format"] != "wav" {
						t.Errorf("unexpected TTS request: %v", req)
					}
					spoken <- req["input"]
					w.Write(append(audio.WAVHeader(24000, 2), 1, 0))
				default:
					t.Errorf("unexpected endpoint %s", r.URL.Path)
					http.NotFound(w, r)
				}
			}))
			defer srv.Close()
			t.Setenv("OLLAMA_HOST", srv.URL)
			args := append([]string{"lfm3-omni", "--keepalive", "1m", prompt}, tc.flags...)
			out, err := runAudioCLI(t, "", args...)
			if err != nil {
				t.Fatal(err)
			}
			var parts []string
			for len(spoken) > 0 {
				parts = append(parts, <-spoken)
			}
			if chats.Load() != 1 || strings.Join(parts, " ") != answer {
				t.Fatalf("chats=%d, spoken=%q; want the answer, not the prompt or thinking", chats.Load(), parts)
			}
			want := append(audio.WAVHeader(24000, uint32(2*len(parts))), bytes.Repeat([]byte{1, 0}, len(parts))...)
			if !bytes.Equal([]byte(out), want) {
				t.Fatalf("stdout is not a clean WAV: %x", out)
			}
		})
	}
}

func TestReadInputAloud(t *testing.T) {
	for _, flags := range [][]string{{"--tts"}, {"--tts", "--voice", "male-german"}, {"--voice", "male-german"}} {
		t.Run(strings.Join(flags, " "), func(t *testing.T) {
			spoken := make(chan string, 1)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/show":
					json.NewEncoder(w).Encode(api.ShowResponse{Capabilities: []model.Capability{model.CapabilityCompletion, model.CapabilityAudioOutput, model.CapabilitySpeech}})
				case "/v1/audio/speech":
					var req map[string]string
					json.NewDecoder(r.Body).Decode(&req)
					spoken <- req["input"]
					w.Write(append(audio.WAVHeader(24000, 2), 1, 0))
				default:
					t.Errorf("reading input must not generate an answer: %s", r.URL.Path)
					http.NotFound(w, r)
				}
			}))
			defer srv.Close()
			t.Setenv("OLLAMA_HOST", srv.URL)
			out, err := runAudioCLI(t, "", append([]string{"lfm3-omni", "Warum ist der Himmel blau?"}, flags...)...)
			if err != nil || !strings.HasPrefix(out, "RIFF") {
				t.Fatalf("output=%q, err=%v", out, err)
			}
			select {
			case text := <-spoken:
				if text != "Warum ist der Himmel blau?" {
					t.Fatalf("read %q instead of input", text)
				}
			default:
				t.Fatal("input was not sent to speech")
			}
		})
	}
}

func TestAudioModeConflicts(t *testing.T) {
	for _, flags := range [][]string{{"--speak", "--tts"}, {"--speak", "--audio"}, {"--tts", "--audio"}} {
		out, err := runAudioCLI(t, "", append([]string{"lfm3-omni", "Question"}, flags...)...)
		if err == nil || !strings.Contains(err.Error(), "choose one audio mode") || out != "" {
			t.Fatalf("flags=%v, output=%q, error=%v", flags, out, err)
		}
	}
}

func TestSpokenChatErrors(t *testing.T) {
	for _, tc := range []struct{ name, response, want string }{
		{"empty", `{"done":true}`, "no text"},
		{"truncated", `{"message":{"content":"partial"}}`, "unexpected EOF"},
		{"server", `{"error":"model failed"}`, "model failed"},
		{"tools", `{"message":{"tool_calls":[{"function":{"name":"lookup"}}]},"done":true}`, "expected a text answer"},
		{"limit", `{"message":{"content":"` + strings.Repeat("a", (1<<20)+1) + `"}}`, "1 MiB"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := audioChatClient(t, func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, tc.response+"\n") })
			err := speakChatAnswer(t.Context(), client, runOptions{Prompt: "Question"}, func(_ context.Context, r io.Reader) error {
				_, err := io.Copy(io.Discard, r)
				return err
			})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want %q", err, tc.want)
			}
		})
	}
}

func TestSpokenChatCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	closed := make(chan struct{})
	client := audioChatClient(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(api.ChatResponse{Message: api.Message{Content: "Partial answer."}})
		w.(http.Flusher).Flush()
		cancel()
		select {
		case <-r.Context().Done():
			close(closed)
		case <-time.After(5 * time.Second):
			t.Error("chat request did not disconnect")
		}
	})
	err := speakChatAnswer(ctx, client, runOptions{Prompt: "Question"}, func(_ context.Context, r io.Reader) error {
		_, err := io.Copy(io.Discard, r)
		return err
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want cancellation", err)
	}
	select {
	case <-closed:
	case <-time.After(5 * time.Second):
		t.Fatal("connection was not released")
	}
}

func TestSpokenChatSpeechFailure(t *testing.T) {
	closed := make(chan struct{})
	client := audioChatClient(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(api.ChatResponse{Message: api.Message{Content: "An answer."}})
		w.(http.Flusher).Flush()
		select {
		case <-r.Context().Done():
			close(closed)
		case <-time.After(5 * time.Second):
			t.Error("speech failure did not cancel text generation")
		}
	})
	want := errors.New("playback failed")
	err := speakChatAnswer(t.Context(), client, runOptions{Prompt: "Question"}, func(_ context.Context, r io.Reader) error {
		var first [1]byte
		if _, err := r.Read(first[:]); err != nil {
			return err
		}
		return want
	})
	if !errors.Is(err, want) {
		t.Fatalf("got %v, want %v", err, want)
	}
	select {
	case <-closed:
	case <-time.After(5 * time.Second):
		t.Fatal("text stream still open")
	}
}

func TestSpokenChatCancelsBothRequests(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	closed := make(chan string, 2)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		switch r.URL.Path {
		case "/api/chat":
			json.NewEncoder(w).Encode(api.ChatResponse{Message: api.Message{Content: "A complete sentence. "}})
			w.(http.Flusher).Flush()
		case "/v1/audio/speech":
			cancel() // Cancel only after text and speech requests are both active.
		default:
			t.Errorf("unexpected endpoint %s", r.URL.Path)
			return
		}
		select {
		case <-r.Context().Done():
			closed <- r.URL.Path
		case <-time.After(5 * time.Second):
			t.Errorf("request did not disconnect: %s", r.URL.Path)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)
	client, err := api.ClientFromEnvironment()
	if err != nil {
		t.Fatal(err)
	}
	err = speakChatAnswer(ctx, client, runOptions{Model: "lfm3-omni", Prompt: "Question"}, func(ctx context.Context, r io.Reader) error {
		return speak(ctx, "lfm3-omni", "male-german", r)
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want cancellation after both requests started", err)
	}
	for range 2 {
		select {
		case <-closed:
		case <-time.After(5 * time.Second):
			t.Fatal("text or speech request still open")
		}
	}
}

func TestSpokenChatSingleSlot(t *testing.T) {
	finished := make(chan struct{})
	const sentence = "A complete sentence. "
	client := audioChatClient(t, func(w http.ResponseWriter, r *http.Request) {
		enc := json.NewEncoder(w)
		for range 4096 {
			enc.Encode(api.ChatResponse{Message: api.Message{Content: sentence}})
		}
		enc.Encode(api.ChatResponse{Done: true})
		close(finished)
	})
	err := speakChatAnswer(t.Context(), client, runOptions{Prompt: "Question"}, func(_ context.Context, r io.Reader) error {
		// Model speech is queued until the text request releases its only slot.
		select {
		case <-finished:
		case <-time.After(5 * time.Second):
			return errors.New("text blocked behind queued speech")
		}
		data, err := io.ReadAll(r)
		if err == nil && string(data) != strings.Repeat(sentence, 4096) {
			t.Error("queued text lost or reordered")
		}
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestSpokenChatInvalidVoice(t *testing.T) {
	// Validation must happen before any HTTP request, so a client is not needed.
	err := runSpokenChat(t.Context(), nil, runOptions{Prompt: "Question"}, "male-klingon", false)
	if err == nil || !strings.Contains(err.Error(), "not supported") {
		t.Fatalf("invalid voice was not rejected: %v", err)
	}
}
