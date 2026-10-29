package cmd

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlxrunner/model/audio"
	"github.com/ollama/ollama/types/model"
)

func audioChatClient(t *testing.T, handler http.HandlerFunc) *api.Client {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	return api.NewClient(u, srv.Client())
}

func audioResponse(pcm []byte) api.ChatResponse {
	return api.ChatResponse{Audio: &api.AudioChunk{Data: pcm, Format: "pcm16", SampleRate: 24000, Channels: 1}}
}

func TestNativeAudioShowVoices(t *testing.T) {
	var out bytes.Buffer
	err := showInfo(&api.ShowResponse{
		Capabilities: []model.Capability{model.CapabilitySpeech, model.CapabilityAudioOutput},
		Voices:       []string{"female-german", "male-german"},
	}, false, &out)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "Voices (--tts / --speak)") {
		t.Fatalf("voice scope not shown: %s", &out)
	}
}

func TestRunNativeAudio(t *testing.T) {
	for _, tc := range []struct {
		name, stdin, prompt string
	}{
		{"argument", "", "Why is the sky blue?"},
		{"stdin", "Why is the sky blue?", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			requests := make(chan api.ChatRequest, 1)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/show":
					var req api.ShowRequest
					if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Runner != manifest.RunnerLlamaCPP {
						t.Errorf("GGUF selection: %+v, %v", req, err)
					}
					json.NewEncoder(w).Encode(api.ShowResponse{Capabilities: []model.Capability{model.CapabilityCompletion, model.CapabilityThinking, model.CapabilityAudioOutput}})
				case "/api/chat":
					var req api.ChatRequest
					if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
						t.Error(err)
					}
					requests <- req
					enc := json.NewEncoder(w)
					enc.Encode(audioResponse([]byte{1, 0}))
					enc.Encode(audioResponse([]byte{2, 0}))
					enc.Encode(api.ChatResponse{Done: true})
				default:
					t.Errorf("unexpected request (no separate completion or TTS allowed): %s", r.URL.Path)
					http.NotFound(w, r)
				}
			}))
			defer srv.Close()
			t.Setenv("OLLAMA_HOST", srv.URL)
			args := []string{"lfm3-omni", "--audio", "--keepalive", "2m"}
			if tc.prompt != "" {
				args = append(args, tc.prompt)
			}
			out, err := runAudioCLI(t, tc.stdin, args...)
			if err != nil {
				t.Fatal(err)
			}
			want := append(audio.WAVHeader(24000, ^uint32(0)), 1, 0, 2, 0)
			if !bytes.Equal([]byte(out), want) {
				t.Fatalf("output is not the expected streamed WAV: %x", out)
			}
			req := <-requests
			if req.Runner != manifest.RunnerLlamaCPP || req.Model != "lfm3-omni" || !reflect.DeepEqual(req.Messages, []api.Message{{Role: "user", Content: "Why is the sky blue?"}}) ||
				!reflect.DeepEqual(req.OutputModalities, []string{"audio"}) || req.Audio == nil || req.Audio.Format != "pcm16" || req.Audio.Voice != "" || req.Stream == nil || !*req.Stream || req.Think != nil || req.KeepAlive == nil || req.KeepAlive.Duration != 2*time.Minute {
				t.Fatalf("unexpected native request: %+v", req)
			}
		})
	}
}

// Use the real run flags and a pipe for stdin, without contacting a real model.
func runAudioCLI(t *testing.T, stdin string, args ...string) (string, error) {
	t.Helper()
	cmd, _, err := NewCLI().Find([]string{"run"})
	if err != nil {
		t.Fatal(err)
	}
	cmd.SetContext(t.Context())
	if err := cmd.ParseFlags(args); err != nil {
		return "", err
	}
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	io.WriteString(w, stdin)
	w.Close()
	previous := os.Stdin
	os.Stdin = r
	defer func() { os.Stdin = previous }()
	out := captureStdout(t, func() { err = RunHandler(cmd, cmd.Flags().Args()) })
	return out, err
}

func TestRunNativeAudioValidation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/show" {
			t.Errorf("unsupported request reached %s", r.URL.Path)
			http.NotFound(w, r)
			return
		}
		json.NewEncoder(w).Encode(api.ShowResponse{Capabilities: []model.Capability{model.CapabilitySpeech}})
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)
	for _, tc := range []struct {
		args []string
		want string
	}{
		{nil, "does not support native audio"},
		{[]string{"--format", "json"}, "--format"},
		{[]string{"--think"}, "thinking"},
		{[]string{"--runner", "mlx"}, "GGUF runner"},
		{[]string{"--dimensions", "256"}, "--dimensions"},
		{[]string{"--ask", "Yes?"}, "--ask"},
	} {
		t.Run(tc.want, func(t *testing.T) {
			out, err := runAudioCLI(t, "", append([]string{"phonon", "--audio", "Hello"}, tc.args...)...)
			if err == nil || !strings.Contains(err.Error(), tc.want) || out != "" {
				t.Fatalf("output %q, error %v; want %q", out, err, tc.want)
			}
		})
	}
}

func TestWriteAudioChatFile(t *testing.T) {
	client := audioChatClient(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(audioResponse([]byte{1, 0, 2, 0}))
		json.NewEncoder(w).Encode(api.ChatResponse{Done: true})
	})
	f, err := os.Create(filepath.Join(t.TempDir(), "answer.wav"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := writeAudioChat(t.Context(), client, runOptions{Prompt: "Question"}, f); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(f.Name())
	if err != nil {
		t.Fatal(err)
	}
	if want := append(audio.WAVHeader(24000, 4), 1, 0, 2, 0); !bytes.Equal(got, want) {
		t.Fatalf("invalid saved WAV: %x", got)
	}
}

func TestAudioChatStreamsBeforeCompletion(t *testing.T) {
	for _, playback := range []bool{false, true} {
		t.Run(map[bool]string{false: "stdout", true: "playback"}[playback], func(t *testing.T) {
			emitted := make(chan struct{}, 1)
			client := audioChatClient(t, func(w http.ResponseWriter, r *http.Request) {
				json.NewEncoder(w).Encode(audioResponse(make([]byte, 48000)))
				w.(http.Flusher).Flush()
				select {
				case <-emitted:
				case <-time.After(5 * time.Second):
					t.Error("audio was buffered until completion")
				}
				json.NewEncoder(w).Encode(api.ChatResponse{Done: true})
			})
			opts := runOptions{Prompt: "Question"}
			var err error
			if playback {
				err = playAudioChat(t.Context(), client, opts, func(ctx context.Context, wav []byte) error {
					if len(wav) != 48044 || binary.LittleEndian.Uint32(wav[40:]) != 48000 {
						t.Errorf("invalid playback WAV: %d bytes", len(wav))
					}
					emitted <- struct{}{}
					return nil
				})
			} else {
				err = writeAudioChat(t.Context(), client, opts, audioWriter(func(p []byte) (int, error) {
					if len(p) == 48000 {
						emitted <- struct{}{}
					}
					return len(p), nil
				}))
			}
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}

type audioWriter func([]byte) (int, error)

func (f audioWriter) Write(p []byte) (int, error) { return f(p) }

func TestAudioChatErrors(t *testing.T) {
	for _, tc := range []struct {
		name, response, want string
	}{
		{"empty", `{"done":true}`, "no audio"},
		{"truncated", `{"audio":{"format":"pcm16","sample_rate":24000,"channels":1,"data":"AQA="}}`, "before completion"},
		{"format", `{"audio":{"format":"mp3","sample_rate":24000,"channels":1,"data":"AQA="}}`, "PCM16"},
		{"rate", `{"audio":{"format":"pcm16","sample_rate":48000,"channels":1,"data":"AQA="}}`, "24000 Hz"},
		{"channels", `{"audio":{"format":"pcm16","sample_rate":24000,"channels":2,"data":"AQA="}}`, "mono"},
		{"partial sample", `{"audio":{"format":"pcm16","sample_rate":24000,"channels":1,"data":"AQ=="}}`, "complete samples"},
		{"server", `{"error":"backend failed"}`, "backend failed"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := audioChatClient(t, func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, tc.response+"\n") })
			err := writeAudioChat(t.Context(), client, runOptions{Prompt: "Question"}, io.Discard)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want %q", err, tc.want)
			}
		})
	}
}

func TestAudioChatCancellation(t *testing.T) {
	for _, mode := range []string{"cancel", "playback failure", "writer failure"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			closed := make(chan struct{})
			client := audioChatClient(t, func(w http.ResponseWriter, r *http.Request) {
				// More than the bounded playback queue can hold.
				json.NewEncoder(w).Encode(audioResponse(make([]byte, 48000*8)))
				w.(http.Flusher).Flush()
				select {
				case <-r.Context().Done():
					close(closed)
				case <-time.After(5 * time.Second):
					t.Error("HTTP generation was not canceled")
				}
			})
			failure := errors.New("output failed")
			var err error
			if mode == "writer failure" {
				err = writeAudioChat(ctx, client, runOptions{Prompt: "Question"}, audioWriter(func([]byte) (int, error) { return 0, failure }))
			} else {
				err = playAudioChat(ctx, client, runOptions{Prompt: "Question"}, func(ctx context.Context, wav []byte) error {
					if mode == "cancel" {
						cancel()
						return ctx.Err()
					}
					return failure
				})
			}
			want := failure
			if mode == "cancel" {
				want = context.Canceled
			}
			if !errors.Is(err, want) {
				t.Fatalf("got %v, want %v", err, want)
			}
			select {
			case <-closed:
			case <-time.After(5 * time.Second):
				t.Fatal("client did not disconnect")
			}
		})
	}
}
