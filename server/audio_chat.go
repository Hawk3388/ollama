package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func (m *Model) supportsNativeAudio() bool {
	if !m.isGGUF() || m.metadata.String("general.architecture") != "lfm3" || m.metadata.String("tokenizer.chat_template") == "" {
		return false
	}
	// Restrict the first implementation to Omni, and require the decoder in
	// the projector actually loaded by llama-server (the first projector).
	if len(m.ProjectorPaths) == 0 || len(m.projectorMetadata) == 0 {
		return false
	}
	md := m.projectorMetadata[0]
	return projectorHasVision(md) && projectorHasAudio(md) && md.Bool("has_gen_audio_encoder")
}

func validateAudioChat(req api.ChatRequest) (llm.AudioChatRequest, error) {
	var turn llm.AudioChatRequest
	if !slices.Equal(req.OutputModalities, []string{"audio"}) {
		return turn, errors.New("native audio requires output_modalities [\"audio\"]")
	}
	if req.Audio != nil && req.Audio.Format != "" && req.Audio.Format != "pcm16" {
		return turn, errors.New("native audio supports only audio.format pcm16")
	}
	if req.Audio != nil && req.Audio.Voice != "" {
		return turn, errors.New("native audio voice selection is not supported by the current LFM3-Omni realtime backend; the listed voices are for text-to-speech only")
	}
	if req.Stream != nil && !*req.Stream {
		return turn, errors.New("native audio requires stream true")
	}
	if len(req.Tools) != 0 || len(req.Format) != 0 || req.Think != nil && req.Think.Bool() || req.Logprobs || req.TopLogprobs != 0 || req.DebugRenderOnly || req.Shift != nil || req.Truncate != nil {
		return turn, errors.New("native audio does not support tools, structured output, thinking, logprobs, debug rendering, shift or truncate")
	}
	for key := range req.Options {
		switch key {
		case "num_ctx", "num_batch", "num_gpu", "num_thread", "num_predict":
		default:
			return turn, fmt.Errorf("option %q is not supported for native audio; sampling uses the backend defaults", key)
		}
	}
	msgs := req.Messages
	if len(msgs) > 0 && msgs[0].Role == "system" {
		turn.System = msgs[0].Content
		msgs = msgs[1:]
	}
	if len(msgs) != 1 || msgs[0].Role != "user" || strings.TrimSpace(msgs[0].Content) == "" {
		return turn, errors.New("native audio requires one non-empty user text message, optionally preceded by a system message")
	}
	for _, msg := range req.Messages {
		if len(msg.Images) != 0 || len(msg.ToolCalls) != 0 || msg.Thinking != "" || msg.ToolCallID != "" || msg.ToolName != "" {
			return turn, errors.New("native audio accepts text messages only")
		}
		if len(msg.Content) > 64<<10 {
			return turn, errors.New("native audio messages must be at most 64 KiB each")
		}
	}
	turn.Prompt = msgs[0].Content
	return turn, nil
}

func (s *Server) audioChatHandler(c *gin.Context, req api.ChatRequest) {
	turn, err := validateAudioChat(req)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	ref, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusBadRequest, "model is required")
		return
	}
	runner, err := normalizeRunner(req.Runner)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if ref.Source == modelSourceCloud || runner != "" && runner != manifest.RunnerLlamaCPP && runner != manifest.RunnerGGML {
		c.JSON(http.StatusBadRequest, gin.H{"error": "native audio requires a local llama.cpp GGUF runner"})
		return
	}
	// Select the GGUF child even on hosts which otherwise prefer MLX.
	m, err := s.getExistingModelForRunner(ref.Name.String(), manifest.RunnerLlamaCPP)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}
	if !m.supportsNativeAudio() || m.Config.RemoteHost != "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "native audio requires LFM3-Omni GGUF with its audio codec projector and native chat templates"})
		return
	}
	if len(req.Messages) == 1 {
		turn.System = m.System
	}
	// The scheduler holds the model for this request context; cancellation also
	// releases that reference after the upstream websocket has been closed.
	ctx, cancel := context.WithCancel(c.Request.Context())
	defer cancel()
	r, _, opts, err := s.scheduleRunnerForModel(ctx, m, []model.Capability{model.CapabilityCompletion, model.CapabilityAudioOutput}, req.Options, req.KeepAlive, nil)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}
	chatter, ok := r.(llm.AudioChatter)
	if !ok {
		c.JSON(http.StatusBadRequest, gin.H{"error": "selected runner does not implement native audio chat"})
		return
	}
	turn.MaxTokens = opts.NumPredict
	if turn.MaxTokens <= 0 {
		turn.MaxTokens = 2048
	}
	streamAudioChat(c, ctx, chatter, req.Model, turn)
}

func streamAudioChat(c *gin.Context, ctx context.Context, chatter llm.AudioChatter, name string, turn llm.AudioChatRequest) {
	var writer http.ResponseWriter = c.Writer
	if w, ok := writer.(interface{ Unwrap() http.ResponseWriter }); ok {
		writer = w.Unwrap()
	}
	controller := http.NewResponseController(writer)
	defer controller.SetWriteDeadline(time.Time{})
	started, clientGone := false, false
	write := func(value any) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := controller.SetWriteDeadline(time.Now().Add(30 * time.Second)); err != nil {
			return err
		}
		c.Header("Content-Type", "application/x-ndjson")
		if err := json.NewEncoder(c.Writer).Encode(value); err != nil {
			return err
		}
		return controller.Flush()
	}
	response := func() api.ChatResponse {
		return api.ChatResponse{Model: name, CreatedAt: time.Now().UTC(), Message: api.Message{Role: "assistant"}}
	}
	err := chatter.AudioChat(ctx, turn, func(chunk api.AudioChunk) error {
		r := response()
		r.Audio = &chunk
		err := write(r)
		clientGone = err != nil
		started = true
		return err
	})
	if clientGone || ctx.Err() != nil {
		return
	}
	if err != nil {
		if started {
			// A terminal NDJSON error (no done:true) marks partial audio as failed.
			write(gin.H{"error": err.Error()})
		} else {
			status := http.StatusInternalServerError
			var se api.StatusError
			if errors.As(err, &se) {
				status = se.StatusCode
			}
			c.JSON(status, gin.H{"error": err.Error()})
		}
		return
	}
	r := response()
	r.Done, r.DoneReason = true, "stop"
	write(r)
}
