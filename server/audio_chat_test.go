package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
)

func TestAudioChatValidation(t *testing.T) {
	for _, tt := range []struct{ name, fields, want string }{
		{"mode", `"output_modalities":["video"]`, "output_modalities"},
		{"empty mode", `"output_modalities":[]`, "output_modalities"},
		{"mixed", `"output_modalities":["text","audio"]`, "output_modalities"},
		{"format", `"audio":{"format":"mp3"}`, "pcm16"},
		{"voice", `"audio":{"voice":"female-german"}`, "voice selection is not supported"},
		{"default voice", `"audio":{"voice":"default"}`, "voice selection is not supported"},
		{"stream", `"stream":false`, "stream true"},
		{"tools", `"tools":[{}]`, "tools"},
		{"json", `"format":"json"`, "structured output"},
		{"thinking", `"think":true`, "thinking"},
		{"sampler", `"options":{"temperature":0.5}`, "temperature"},
		{"empty", `"messages":[]`, "user text"},
		{"blank", `"messages":[{"role":"user","content":" "}]`, "user text"},
		{"history", `"messages":[{"role":"assistant","content":"a"},{"role":"user","content":"b"}]`, "user text"},
		{"media", `"messages":[{"role":"user","content":"x","images":["AA=="]}]`, "text messages only"},
		{"runner", `"runner":"mlx"`, "local llama.cpp"},
		{"unknown runner", `"runner":"unknown"`, "unknown runner"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			request := map[string]any{
				"model":             "omni",
				"messages":          []api.Message{{Role: "user", Content: "why?"}},
				"output_modalities": []string{"audio"},
			}
			if err := json.Unmarshal([]byte("{"+tt.fields+"}"), &request); err != nil {
				t.Fatal(err)
			}
			payload, err := json.Marshal(request)
			if err != nil {
				t.Fatal(err)
			}
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request = httptest.NewRequest("POST", "/api/chat", bytes.NewReader(payload))
			// A nil scheduler proves invalid input is rejected before loading.
			(&Server{}).ChatHandler(c)
			if w.Code != 400 || !strings.Contains(w.Body.String(), tt.want) {
				t.Fatalf("%d %s, want %s", w.Code, w.Body, tt.want)
			}
		})
	}
	turn, err := validateAudioChat(api.ChatRequest{OutputModalities: []string{"audio"}, Messages: []api.Message{{Role: "system", Content: "Antworte kurz auf Deutsch"}, {Role: "user", Content: "Warum ist der Himmel blau?"}}})
	if err != nil || turn.System != "Antworte kurz auf Deutsch" || turn.Prompt != "Warum ist der Himmel blau?" {
		t.Fatalf("%+v %v", turn, err)
	}
}

func nativeAudioTestModel() *Model {
	return &Model{
		Config:         model.ConfigV2{ModelFormat: "gguf", Renderer: "lfm3-omni"},
		ModelPath:      "model.gguf",
		ProjectorPaths: []string{"mmproj.gguf"},
		metadata: ggufMetadata{KV: map[string]any{
			"general.architecture":    "lfm3",
			"tokenizer.chat_template": "template",
		}},
		projectorMetadata: []ggufMetadata{{KV: map[string]any{
			"general.architecture":       "clip",
			"clip.has_vision_encoder":    true,
			"clip.has_audio_encoder":     true,
			"clip.has_gen_audio_encoder": true,
		}}},
	}
}

func TestNativeAudioCapabilityAndLaunch(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func(*Model)
		want   bool
	}{
		{"omni", func(*Model) {}, true},
		{"missing projector", func(m *Model) { m.ProjectorPaths = nil }, false},
		{"missing codec", func(m *Model) { delete(m.projectorMetadata[0].KV, "clip.has_gen_audio_encoder") }, false},
		{"text model", func(m *Model) { m.metadata.KV["general.architecture"] = "llama" }, false},
		{"missing template", func(m *Model) { delete(m.metadata.KV, "tokenizer.chat_template") }, false},
		{"mlx", func(m *Model) { m.Config.ModelFormat = "safetensors" }, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			m := nativeAudioTestModel()
			tt.change(m)
			m.Config.Capabilities = []string{"audio_output"} // cannot spoof this capability
			if got := slices.Contains(m.Capabilities(), model.CapabilityAudioOutput); got != tt.want {
				t.Fatalf("capability %v, want %v", got, tt.want)
			}
			config := llamaServerConfigForModel(m)
			if config.NativeAudio != tt.want || config.DisableJinja == tt.want {
				t.Fatalf("launch %+v", config)
			}
		})
	}
}

type audioChatFunc func(context.Context, llm.AudioChatRequest, func(api.AudioChunk) error) error

func (f audioChatFunc) AudioChat(ctx context.Context, req llm.AudioChatRequest, emit func(api.AudioChunk) error) error {
	return f(ctx, req, emit)
}

func audioChatHTTPServer(t *testing.T, f audioChatFunc) *httptest.Server {
	t.Helper()
	r := gin.New()
	r.POST("/api/chat", func(c *gin.Context) {
		streamAudioChat(c, c.Request.Context(), f, "omni", llm.AudioChatRequest{Prompt: "why?"})
	})
	s := httptest.NewServer(r)
	t.Cleanup(s.Close)
	return s
}

func TestAudioChatHTTPStreaming(t *testing.T) {
	next := make(chan struct{})
	s := audioChatHTTPServer(t, func(ctx context.Context, req llm.AudioChatRequest, emit func(api.AudioChunk) error) error {
		if req.Prompt != "why?" {
			t.Error("lost prompt")
		}
		if err := emit(api.AudioChunk{Data: []byte{1, 0}, Format: "pcm16", SampleRate: 24000, Channels: 1}); err != nil {
			return err
		}
		select {
		case <-next:
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(3 * time.Second):
			return errors.New("first chunk was not flushed")
		}
		return emit(api.AudioChunk{Data: []byte{2, 0}, Format: "pcm16", SampleRate: 24000, Channels: 1})
	})
	resp, err := http.Post(s.URL+"/api/chat", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.Header.Get("Content-Type") != "application/x-ndjson" {
		t.Fatal(resp.Header)
	}
	dec := json.NewDecoder(resp.Body)
	var first api.ChatResponse
	if err := dec.Decode(&first); err != nil {
		t.Fatal(err)
	}
	if first.Audio == nil || !bytes.Equal(first.Audio.Data, []byte{1, 0}) || first.Done {
		t.Fatalf("%+v", first)
	}
	close(next)
	var second, done api.ChatResponse
	if err := dec.Decode(&second); err != nil {
		t.Fatal(err)
	}
	if err := dec.Decode(&done); err != nil {
		t.Fatal(err)
	}
	if second.Audio == nil || !done.Done || done.Audio != nil {
		t.Fatalf("second %+v, done %+v", second, done)
	}
}

func TestAudioChatHTTPFailure(t *testing.T) {
	for _, partial := range []bool{false, true} {
		s := audioChatHTTPServer(t, func(_ context.Context, _ llm.AudioChatRequest, emit func(api.AudioChunk) error) error {
			if partial {
				if err := emit(api.AudioChunk{Data: []byte{0, 0}}); err != nil {
					return err
				}
			}
			return errors.New("codec failed")
		})
		resp, err := http.Post(s.URL+"/api/chat", "application/json", nil)
		if err != nil {
			t.Fatal(err)
		}
		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			t.Fatal(err)
		}
		want := 500
		if partial {
			want = 200
		}
		if resp.StatusCode != want || !bytes.Contains(body, []byte("codec failed")) || bytes.Contains(body, []byte(`"done":true`)) {
			t.Fatalf("%d %s", resp.StatusCode, body)
		}
	}
}

func TestAudioChatHTTPDisconnect(t *testing.T) {
	stopped := make(chan struct{})
	s := audioChatHTTPServer(t, func(ctx context.Context, _ llm.AudioChatRequest, emit func(api.AudioChunk) error) error {
		defer close(stopped)
		if err := emit(api.AudioChunk{Data: []byte{0, 0}}); err != nil {
			return err
		}
		<-ctx.Done()
		return ctx.Err()
	})
	resp, err := http.Post(s.URL+"/api/chat", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	var first api.ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&first); err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	select {
	case <-stopped:
	case <-time.After(3 * time.Second):
		t.Fatal("disconnect did not cancel generation")
	}
}

type stalledAudioWriter struct {
	*httptest.ResponseRecorder
	deadline time.Time
}

func (w *stalledAudioWriter) SetWriteDeadline(deadline time.Time) error {
	if !deadline.IsZero() {
		w.deadline = deadline
	}
	return nil
}

func (w *stalledAudioWriter) FlushError() error { return context.DeadlineExceeded }

func TestAudioChatSlowClient(t *testing.T) {
	w := &stalledAudioWriter{ResponseRecorder: httptest.NewRecorder()}
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest("POST", "/api/chat", nil)
	called := false
	streamAudioChat(c, c.Request.Context(), audioChatFunc(func(_ context.Context, _ llm.AudioChatRequest, emit func(api.AudioChunk) error) error {
		called = true
		err := emit(api.AudioChunk{Data: []byte{0, 0}})
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("write failure not propagated: %v", err)
		}
		return err
	}), "omni", llm.AudioChatRequest{})
	if !called || time.Until(w.deadline) <= 0 || time.Until(w.deadline) > 30*time.Second {
		t.Fatalf("missing bounded write deadline: %v", w.deadline)
	}
	if bytes.Contains(w.Body.Bytes(), []byte(`"done":true`)) {
		t.Fatal("stalled client received success")
	}
}

type scheduledAudioRunner struct {
	mockRunner
	audioChatFunc
}

func TestAudioChatScheduling(t *testing.T) {
	for _, system := range []string{"", "Antworte kurz auf Deutsch"} {
		t.Run(system, func(t *testing.T) {
			t.Setenv("OLLAMA_MODELS", t.TempDir())
			requested := make(chan llm.AudioChatRequest, 1)
			released := make(chan struct{})
			runner := &scheduledAudioRunner{audioChatFunc: func(_ context.Context, req llm.AudioChatRequest, emit func(api.AudioChunk) error) error {
				requested <- req
				return emit(api.AudioChunk{Data: []byte{1, 0}, Format: "pcm16", SampleRate: 24000, Channels: 1})
			}}
			s := &Server{sched: &Scheduler{
				pendingReqCh:    make(chan *LlmRequest, 1),
				finishedReqCh:   make(chan *LlmRequest, 1),
				expiredCh:       make(chan *runnerRef, 1),
				unloadedCh:      make(chan any, 1),
				loaded:          make(map[string]*runnerRef),
				newServerFn:     newMockServer(&runner.mockRunner),
				getGpuFn:        getGpuFn,
				getSystemInfoFn: getSystemInfoFn,
				loadFn: func(req *LlmRequest, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
					if req.sessionDuration == nil || req.sessionDuration.Duration != time.Minute {
						t.Error("keep_alive was lost")
					}
					if !llamaServerConfigForModel(req.model).NativeAudio {
						t.Error("native templates not enabled during loading")
					}
					go func() { <-req.ctx.Done(); close(released) }()
					req.successCh <- &runnerRef{llama: runner}
					return false
				},
			}}
			go s.sched.Run(t.Context())
			_, digest := createBinFile(t, gguftest.KV{
				"general.architecture":      "lfm3",
				"lfm3.block_count":          uint32(1),
				"lfm3.context_length":       uint32(4096),
				"lfm3.embedding_length":     uint32(64),
				"lfm3.attention.head_count": uint32(1),
				"tokenizer.chat_template":   "{{ messages[0]['content'] }}",
				"tokenizer.ggml.tokens":     []string{""},
			}, []*gguftest.Tensor{{Name: "token_embd.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))}})
			_, projector := createBinFile(t, gguftest.KV{
				"general.architecture":       "clip",
				"clip.has_vision_encoder":    true,
				"clip.has_audio_encoder":     true,
				"clip.has_gen_audio_encoder": true,
			}, []*gguftest.Tensor{{Name: "v.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))}})
			noStream := false
			w := createRequest(t, s.CreateHandler, api.CreateRequest{Model: "native-omni", Files: map[string]string{"model.gguf": digest, "mmproj.gguf": projector}, Renderer: "lfm3-omni", System: "Model system prompt", Stream: &noStream})
			if w.Code != 200 {
				t.Fatalf("create: %d %s", w.Code, w.Body)
			}
			router := gin.New()
			router.POST("/api/chat", s.ChatHandler)
			srv := httptest.NewServer(router)
			defer srv.Close()
			messages := []api.Message{{Role: "user", Content: "Why is the sky blue?"}}
			if system != "" {
				messages = append([]api.Message{{Role: "system", Content: system}}, messages...)
			}
			body, _ := json.Marshal(api.ChatRequest{Model: "native-omni", Messages: messages, OutputModalities: []string{"audio"}, Options: map[string]any{"num_predict": 512}, KeepAlive: &api.Duration{Duration: time.Minute}})
			resp, err := http.Post(srv.URL+"/api/chat", "application/json", bytes.NewReader(body))
			if err != nil {
				t.Fatal(err)
			}
			data, err := io.ReadAll(resp.Body)
			resp.Body.Close()
			if err != nil || resp.StatusCode != 200 || !bytes.Contains(data, []byte(`"done":true`)) {
				t.Fatalf("%d %s %v", resp.StatusCode, data, err)
			}
			select {
			case req := <-requested:
				wantSystem := system
				if wantSystem == "" {
					wantSystem = "Model system prompt"
				}
				if req.Prompt != "Why is the sky blue?" || req.System != wantSystem || req.MaxTokens != 512 {
					t.Fatalf("request %+v", req)
				}
			case <-time.After(3 * time.Second):
				t.Fatal("runner not invoked")
			}
			select {
			case <-released:
			case <-time.After(3 * time.Second):
				t.Fatal("scheduler request context not released")
			}
		})
	}
}
