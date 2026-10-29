package cmd

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlxrunner/model/audio"
	"github.com/ollama/ollama/types/model"
)

func TestSpeechCut(t *testing.T) {
	long := strings.Repeat("This sentence is long enough to say. ", 3)
	for _, tc := range []struct {
		text string
		want int
	}{
		{"A short sentence. ", len("A short sentence.")},
		{"An unfinished sentence", 0},
		{"Heading\n\nBody", 7},
		{long + "And then", len(long) - 1},
		{"no sentence end yet " + strings.Repeat("words ", 20), 0},
	} {
		if got := speechCut(tc.text); got != tc.want {
			t.Errorf("speechCut(%q) = %d, want %d", tc.text, got, tc.want)
		}
	}
}

func TestRunPhonon(t *testing.T) {
	for _, tc := range []struct{ name, stdin, prompt string }{
		{"piped reply", "Owls have silent wings. They hunt at night.", ""},
		{"save WAV", "", "Save this to a file."},
	} {
		t.Run(tc.name, func(t *testing.T) {
			requests := make(chan string, 4)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/show":
					json.NewEncoder(w).Encode(api.ShowResponse{Capabilities: []model.Capability{model.CapabilitySpeech}})
				case "/v1/audio/speech":
					var req map[string]string
					if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
						t.Error(err)
					}
					if req["model"] != "phonon" || req["response_format"] != "wav" {
						t.Errorf("unexpected speech request: %v", req)
					}
					requests <- req["input"]
					w.Write(append(audio.WAVHeader(24000, ^uint32(0)), 1, 0))
				default:
					t.Errorf("unexpected request: %s", r.URL.Path)
					http.NotFound(w, r)
				}
			}))
			defer srv.Close()
			t.Setenv("OLLAMA_HOST", srv.URL)
			out, err := runAudioCLI(t, tc.stdin, "phonon", tc.prompt)
			if err != nil {
				t.Fatal(err)
			}
			var spoken []string
			for len(requests) > 0 {
				spoken = append(spoken, <-requests)
			}
			if got := strings.Join(spoken, " "); got != tc.stdin+tc.prompt {
				t.Fatalf("spoken text %q", got)
			}
			want := append(audio.WAVHeader(24000, uint32(2*len(spoken))), bytes.Repeat([]byte{1, 0}, len(spoken))...)
			if !bytes.Equal([]byte(out), want) {
				t.Fatalf("invalid saved speech WAV: %x", out)
			}
		})
	}
}

func TestSpeechRequestBeforeStdinEOF(t *testing.T) {
	requested := make(chan string, 2)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]string
		json.NewDecoder(r.Body).Decode(&req)
		requested <- req["input"]
		w.Write(append(audio.WAVHeader(24000, 2), 1, 0))
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)
	r, w := io.Pipe()
	defer r.Close()
	defer w.Close()
	go func() {
		defer w.Close()
		io.WriteString(w, "Owls have silent wings. ")
		select {
		case text := <-requested:
			if text != "Owls have silent wings." {
				t.Errorf("first request %q", text)
			}
		case <-time.After(5 * time.Second):
			t.Error("synthesis waited for stdin EOF")
		}
		io.WriteString(w, "They hunt at night.")
	}()
	captureStdout(t, func() {
		if err := runSpeech(t.Context(), "phonon", "", "", r, false); err != nil {
			t.Error(err)
		}
	})
}

// Speech starts after the first sentence, before the input ends.
func TestSpeechPiecesStream(t *testing.T) {
	r, w := io.Pipe()
	pieces := speechPieces(t.Context(), r)
	go w.Write([]byte("Owls have silent wings. Unfinished"))
	select {
	case p := <-pieces:
		if p != "Owls have silent wings." {
			t.Errorf("first piece = %q", p)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("speech waited for the end of the input")
	}
	go func() {
		w.Write([]byte(" tail"))
		w.Close()
	}()
	var rest []string
	for p := range pieces {
		rest = append(rest, p)
	}
	if !slices.Equal(rest, []string{"Unfinished tail"}) {
		t.Errorf("rest = %q", rest)
	}
}

func TestSpeechChunks(t *testing.T) {
	for _, tc := range []struct {
		text  string
		limit int
		want  []string
	}{
		{"Hello.", 100, []string{"Hello."}},
		{"One.\n\n Two.\r\n\r\nThree.", 100, []string{"One.", "Two.", "Three."}},
		{"One two. Three four. Five six.", 21, []string{"One two. Three four.", "Five six."}},
		{"one two three four", 10, []string{"one two", "three four"}},
		{"abcdefghijkl", 5, []string{"abcde", "fghij", "kl"}},
		{"ééééé", 2, []string{"éé", "éé", "é"}},
		{" \n\n ", 100, nil},
	} {
		if got := speechChunks(tc.text, tc.limit); !slices.Equal(got, tc.want) {
			t.Errorf("speechChunks(%q, %d) = %q, want %q", tc.text, tc.limit, got, tc.want)
		}
	}
}

func TestSynthesize(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct{ Input, Voice string }
		if r.URL.Path != "/v1/audio/speech" || json.NewDecoder(r.Body).Decode(&req) != nil || req.Input == "" || req.Voice != "Toby" {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"error":{"message":"input is required"}}`))
			return
		}
		wav := make([]byte, 48)
		binary.LittleEndian.PutUint32(wav[4:], 0xffffffff)
		binary.LittleEndian.PutUint32(wav[40:], 0xffffffff)
		copy(wav[44:], []byte{1, 0, 2, 0})
		w.Write(wav)
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	wav, err := synthesize(t.Context(), "phonon", "Toby", "hello")
	if err != nil || binary.LittleEndian.Uint32(wav[4:]) != 40 || binary.LittleEndian.Uint32(wav[40:]) != 4 {
		t.Fatalf("WAV lengths not recorded: %x, %v", wav, err)
	}
	if _, err := synthesize(t.Context(), "phonon", "Toby", ""); err == nil || err.Error() != "input is required" {
		t.Fatalf("server error lost: %v", err)
	}
}
