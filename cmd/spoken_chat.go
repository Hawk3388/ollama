package cmd

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"sync"

	"golang.org/x/term"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/renderers"
)

// runSpokenChat implements --speak. The same
// model streams its text answer while its TTS endpoint speaks complete sentences.
func runSpokenChat(ctx context.Context, client *api.Client, opts runOptions, voice string, interactive bool) error {
	if _, err := renderers.ParseLFM3Voice(voice); err != nil {
		return err
	}
	if term.IsTerminal(int(os.Stdout.Fd())) {
		if _, err := player(opts.Model+" --speak", ""); err != nil {
			return err
		}
	}
	respond := func(ctx context.Context, prompt string) error {
		opts.Prompt = prompt
		if term.IsTerminal(int(os.Stdout.Fd())) {
			fmt.Fprintln(os.Stderr, "Generating answer; complete sentences will be read aloud...")
		}
		return speakChatAnswer(ctx, client, opts, func(ctx context.Context, answer io.Reader) error {
			return speak(ctx, opts.Model, voice, answer)
		})
	}
	if interactive {
		return speechPrompt(ctx, opts.Model+" --speak", "Ask a question; the reply will be read aloud (/bye to exit)", respond)
	}
	ctx, stop := signal.NotifyContext(ctx, os.Interrupt)
	defer stop()
	return respond(ctx, opts.Prompt)
}

func speakChatAnswer(parent context.Context, client *api.Client, opts runOptions, speak func(context.Context, io.Reader) error) error {
	if strings.TrimSpace(opts.Prompt) == "" {
		return errors.New("a spoken answer requires a non-empty text prompt")
	}
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	answer := &chatAnswerReader{ctx: ctx, ready: make(chan struct{}, 1)}
	result := make(chan error, 1)
	go func() {
		err := streamChatText(ctx, client, opts, answer)
		answer.finish(err)
		if err != nil {
			cancel()
		}
		result <- err
	}()
	speechErr := speak(ctx, answer)
	cancel()
	textErr := <-result // Join the text request, including after playback failure.
	if err := parent.Err(); err != nil {
		return err
	}
	if textErr != nil && !errors.Is(textErr, context.Canceled) {
		return textErr
	}
	if speechErr != nil {
		return speechErr
	}
	return textErr
}

func streamChatText(ctx context.Context, client *api.Client, opts runOptions, answer *chatAnswerReader) error {
	stream := true
	req := &api.ChatRequest{
		Model: opts.Model, Runner: opts.Runner,
		Messages: []api.Message{{Role: "user", Content: opts.Prompt}},
		Stream:   &stream, Think: &api.ThinkValue{Value: false},
		OutputModalities: []string{"text"}, KeepAlive: opts.KeepAlive,
	}
	var size int
	var done, hasText bool
	err := client.Chat(ctx, req, func(res api.ChatResponse) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if done {
			return errors.New("received data after text answer completed")
		}
		if res.Audio != nil || len(res.Message.ToolCalls) != 0 {
			return errors.New("expected a text answer for speech synthesis")
		}
		if size+len(res.Message.Content) > 1<<20 {
			return errors.New("text answer exceeds the 1 MiB speech limit")
		}
		size += len(res.Message.Content)
		hasText = hasText || strings.TrimSpace(res.Message.Content) != ""
		answer.append(res.Message.Content)
		done = res.Done
		return nil
	})
	if err != nil {
		return err
	}
	if !done {
		return io.ErrUnexpectedEOF
	}
	if !hasText {
		return errors.New("model returned no text to speak")
	}
	return ctx.Err()
}

// The text request must drain independently of TTS: a one-slot runner can
// queue TTS until text finishes. streamChatText bounds the entire answer to
// 1 MiB, so slow speech never creates an unbounded queue or blocks that slot.
type chatAnswerReader struct {
	ctx   context.Context
	mu    sync.Mutex
	buf   bytes.Buffer
	err   error
	ready chan struct{}
}

func (r *chatAnswerReader) notify() {
	select {
	case r.ready <- struct{}{}:
	default:
	}
}

func (r *chatAnswerReader) append(text string) {
	r.mu.Lock()
	r.buf.WriteString(text)
	r.mu.Unlock()
	r.notify()
}

func (r *chatAnswerReader) finish(err error) {
	r.mu.Lock()
	if err != nil {
		r.buf.Reset()
		r.err = err
	} else {
		r.err = io.EOF
	}
	r.mu.Unlock()
	r.notify()
}

func (r *chatAnswerReader) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	for {
		if err := r.ctx.Err(); err != nil {
			return 0, err
		}
		r.mu.Lock()
		if r.buf.Len() > 0 {
			n, _ := r.buf.Read(p)
			r.mu.Unlock()
			return n, nil
		}
		err := r.err
		r.mu.Unlock()
		if err != nil {
			return 0, err
		}
		select {
		case <-r.ready:
		case <-r.ctx.Done():
			return 0, r.ctx.Err()
		}
	}
}
