package cmd

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/signal"
	"strings"

	"golang.org/x/term"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlxrunner/model/audio"
)

func runAudioChat(ctx context.Context, client *api.Client, opts runOptions, interactive bool) error {
	respond := func(ctx context.Context, text string) error {
		opts.Prompt = text
		if term.IsTerminal(int(os.Stdout.Fd())) {
			if _, err := player(opts.Model+" --audio", ""); err != nil {
				return err
			}
			fmt.Fprintln(os.Stderr, "Generating native audio answer...")
			return playAudioChat(ctx, client, opts, func(ctx context.Context, wav []byte) error {
				return play(ctx, opts.Model+" --audio", wav)
			})
		}
		return writeAudioChat(ctx, client, opts, os.Stdout)
	}
	if interactive {
		return speechPrompt(ctx, opts.Model+" --audio", "Ask for a native audio answer (/bye to exit)", respond)
	}
	ctx, stop := signal.NotifyContext(ctx, os.Interrupt)
	defer stop()
	return respond(ctx, opts.Prompt)
}

// streamAudioAnswer uses one chat request. The model's SYSTEM prompt is applied
// by the server; neither text completion nor speech synthesis is called here.
func streamAudioAnswer(ctx context.Context, client *api.Client, opts runOptions, emit func([]byte) error) error {
	if strings.TrimSpace(opts.Prompt) == "" {
		return fmt.Errorf("native audio requires a prompt. Usage: ollama run %s --audio \"your question here\"", opts.Model)
	}
	stream := true
	req := &api.ChatRequest{
		Model:            opts.Model,
		Runner:           opts.Runner,
		Messages:         []api.Message{{Role: "user", Content: opts.Prompt}},
		OutputModalities: []string{"audio"},
		Audio:            &api.AudioOptions{Format: "pcm16"},
		Stream:           &stream,
		KeepAlive:        opts.KeepAlive,
	}
	var done, received bool
	err := client.Chat(ctx, req, func(res api.ChatResponse) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if done {
			return errors.New("received data after native audio completed")
		}
		if a := res.Audio; a != nil {
			if a.Format != "pcm16" || a.SampleRate != 24000 || a.Channels != 1 || len(a.Data) == 0 || len(a.Data)%2 != 0 {
				return errors.New("invalid native audio chunk: expected PCM16, 24000 Hz, mono with complete samples")
			}
			if err := emit(a.Data); err != nil {
				return err
			}
			received = true
		}
		done = res.Done
		return nil
	})
	if err != nil {
		return err
	}
	if !done {
		return errors.New("native audio stream ended before completion")
	}
	if !received {
		return errors.New("native audio response contained no audio")
	}
	return nil
}

// writeAudioChat streams a WAV, fixing its lengths when output is seekable.
// Pipes keep unknown lengths, as with /v1/audio/speech streaming WAV output.
func writeAudioChat(ctx context.Context, client *api.Client, opts runOptions, out io.Writer) error {
	var start int64
	seeker, seekable := out.(io.WriteSeeker)
	if seekable {
		var err error
		start, err = seeker.Seek(0, io.SeekCurrent)
		seekable = err == nil
	}
	var size uint32
	err := streamAudioAnswer(ctx, client, opts, func(pcm []byte) error {
		if uint64(size)+uint64(len(pcm)) > math.MaxUint32-36 {
			return errors.New("native audio exceeds the WAV size limit")
		}
		if size == 0 {
			if _, err := out.Write(audio.WAVHeader(24000, math.MaxUint32)); err != nil {
				return err
			}
		}
		_, err := out.Write(pcm)
		size += uint32(len(pcm))
		return err
	})
	if err != nil || !seekable {
		return err
	}
	if _, err := seeker.Seek(start, io.SeekStart); err != nil {
		return err
	}
	if _, err := out.Write(audio.WAVHeader(24000, size)); err != nil {
		return err
	}
	_, err = seeker.Seek(start+44+int64(size), io.SeekStart)
	return err
}

// playAudioChat feeds short WAVs to the existing system player while generation
// continues. The bounded queue backpressures the HTTP stream for slow playback.
func playAudioChat(ctx context.Context, client *api.Client, opts runOptions, play func(context.Context, []byte) error) error {
	ctx, cancel := context.WithCancel(ctx)
	pieces := make(chan []byte, 1)
	result := make(chan error, 1)
	finished := make(chan struct{})
	go func() {
		defer close(finished)
		defer close(pieces)
		const pieceSize = 24000 * 2 // One second of PCM16 mono audio.
		pending := make([]byte, 0, pieceSize)
		send := func() error {
			wav := append(audio.WAVHeader(24000, uint32(len(pending))), pending...)
			select {
			case pieces <- wav:
				pending = pending[:0]
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		err := streamAudioAnswer(ctx, client, opts, func(pcm []byte) error {
			for len(pcm) > 0 {
				n := min(pieceSize-len(pending), len(pcm))
				pending = append(pending, pcm[:n]...)
				pcm = pcm[n:]
				if len(pending) == pieceSize {
					if err := send(); err != nil {
						return err
					}
				}
			}
			return nil
		})
		if err == nil && len(pending) > 0 {
			err = send()
		}
		result <- err
	}()
	defer func() {
		cancel()
		<-finished
	}()
	for wav := range pieces {
		if err := play(ctx, wav); err != nil {
			return err
		}
	}
	return <-result
}
