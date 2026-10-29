package llm

import (
	"context"

	"github.com/ollama/ollama/api"
)

// AudioChatRequest is one conversational turn, not text to be read aloud.
type AudioChatRequest struct {
	Prompt    string
	System    string
	MaxTokens int
}

// AudioChatter generates audio in the model's response decoding loop.
// The callback is synchronous: returning an error stops generation.
type AudioChatter interface {
	AudioChat(context.Context, AudioChatRequest, func(api.AudioChunk) error) error
}
