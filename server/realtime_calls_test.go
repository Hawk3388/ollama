package server

import (
	"bytes"
	"context"
	"encoding/json"
	"mime/multipart"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
	"golang.org/x/net/websocket"
)

func TestRealtimeOffer(t *testing.T) {
	const sdp = "v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 0\r\na=sendrecv\r\n"
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	form.WriteField("session", `{"type":"realtime","model":"lfm3-omni","instructions":"Speak German."}`)
	form.WriteField("sdp", sdp)
	form.Close()
	req := httptest.NewRequest("POST", "/v1/realtime/calls", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	offer, opts, err := readCallOffer(httptest.NewRecorder(), req)
	if err != nil || offer.SDP != sdp || opts.Instructions != "Speak German." || opts.Model != "lfm3-omni" {
		t.Fatalf("%+v %v", opts, err)
	}
	for _, tc := range []struct{ name, model, sdp string }{
		{"model required", "", sdp}, {"unsupported codec", "m", strings.Replace(sdp, "SAVPF 0", "SAVPF 111", 1)},
		{"no microphone", "m", strings.Replace(sdp, "sendrecv", "recvonly", 1)}, {"bad SDP", "m", "invalid"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest("POST", "/v1/realtime/calls?model="+tc.model, strings.NewReader(tc.sdp))
			req.Header.Set("Content-Type", "application/sdp")
			if _, _, err := readCallOffer(httptest.NewRecorder(), req); err == nil {
				t.Fatal("accepted invalid offer")
			}
		})
	}
}

func TestRealtimeVoiceConversation(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
	defer cancel()
	// Exercise real ICE/DTLS/RTP in both directions, with a deterministic model.
	var setting webrtc.SettingEngine
	setting.SetIncludeLoopbackCandidate(true)
	client, err := webrtc.NewAPI(webrtc.WithSettingEngine(setting)).NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	connected := make(chan struct{}, 1)
	client.OnConnectionStateChange(func(s webrtc.PeerConnectionState) {
		if s == webrtc.PeerConnectionStateConnected {
			select {
			case connected <- struct{}{}:
			default:
			}
		}
	})
	heard := make(chan byte, 4)
	client.OnTrack(func(track *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		for {
			p, _, err := track.ReadRTP()
			if err != nil {
				return
			}
			if len(p.Payload) == 160 {
				select {
				case heard <- p.Payload[0]:
				case <-ctx.Done():
					return
				}
			}
		}
	})
	mic, err := webrtc.NewTrackLocalStaticSample(webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypePCMU, ClockRate: 8000}, "mic", "test")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = client.AddTrack(mic); err != nil {
		t.Fatal(err)
	}
	offer, err := client.CreateOffer(nil)
	if err != nil {
		t.Fatal(err)
	}
	ready := webrtc.GatheringCompletePromise(client)
	if err := client.SetLocalDescription(offer); err != nil {
		t.Fatal(err)
	}
	select {
	case <-ready:
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	call, err := newVoiceCall(ctx, cancel, *client.LocalDescription())
	if err != nil {
		t.Fatal(err)
	}
	defer call.peer.Close()
	if err = client.SetRemoteDescription(*call.peer.LocalDescription()); err != nil {
		t.Fatal(err)
	}

	modelClosed := make(chan struct{})
	backend := httptest.NewServer(websocket.Handler(func(ws *websocket.Conn) {
		defer close(modelClosed)
		session := map[string]any{"audio": map[string]any{"input": map[string]any{"format": map[string]any{"type": "audio/pcm", "rate": 16000}}, "output": map[string]any{"format": map[string]any{"type": "audio/pcm", "rate": 24000}}}}
		websocket.JSON.Send(ws, map[string]any{"type": "session.created", "session": session})
		var update struct {
			Type    string
			Session struct{ Instructions string }
		}
		if err := websocket.JSON.Receive(ws, &update); err != nil || update.Type != "session.update" || update.Session.Instructions != "test system" {
			t.Error("lost session instructions")
			return
		}
		websocket.JSON.Send(ws, map[string]any{"type": "session.updated", "session": session})
		for i := range 2 {
			var committed bool
			var samples int
			for {
				var ev struct {
					Type  string
					Audio []byte
				}
				if err := websocket.JSON.Receive(ws, &ev); err != nil {
					return
				}
				switch ev.Type {
				case "input_audio_buffer.append":
					samples += len(ev.Audio) / 2
				case "input_audio_buffer.commit":
					committed = true
				case "response.create":
					if !committed || samples < 9600 {
						t.Errorf("invalid turn %d: %d samples, committed=%t", i, samples, committed)
					}
				default:
					t.Errorf("unexpected event %s", ev.Type)
				}
				if ev.Type == "response.create" {
					break
				}
			}
			// A full 20 ms RTP packet split across two backend sample chunks.
			pcm := bytes.Repeat([]byte{0, byte(16 + i*32)}, 480)
			for _, part := range [][]byte{pcm[:2], pcm[2:]} {
				websocket.JSON.Send(ws, map[string]any{"type": "response.output_audio.delta", "delta": part})
			}
			websocket.JSON.Send(ws, map[string]any{"type": "response.done", "response": map[string]string{"status": "completed"}})
		}
		websocket.JSON.Receive(ws, new(json.RawMessage))
	}))
	defer backend.Close()
	ws, err := websocket.Dial("ws"+strings.TrimPrefix(backend.URL, "http"), "", "http://localhost")
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	if err := configureVoice(ws, "test system", 256); err != nil {
		t.Fatal(err)
	}
	ended := make(chan struct{})
	go func() { defer close(ended); call.run(ctx, cancel, ws) }()
	select {
	case <-connected:
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	var previous byte
	for i := range 2 {
		for n := range 42 {
			v := byte(0xff)
			if n < 8 {
				v = 0x9f
			}
			if err := mic.WriteSample(media.Sample{Data: bytes.Repeat([]byte{v}, 160), Duration: 20 * time.Millisecond}); err != nil {
				t.Fatal(err)
			}
			time.Sleep(10 * time.Millisecond)
		}
		select {
		case value := <-heard:
			if value == 0xff || i > 0 && value == previous {
				t.Fatal("missing distinct voice reply")
			}
			previous = value
		case <-ctx.Done():
			t.Fatalf("no audio on turn %d", i)
		}
	}
	client.Close()
	select {
	case <-ctx.Done():
		if ctx.Err() != context.Canceled {
			t.Fatal("remote hangup failed to cancel call")
		}
	case <-time.After(8 * time.Second):
		t.Fatal("call still active after hangup")
	}
	ws.Close() // WithRealtime closes this on context cancellation in production.
	select {
	case <-modelClosed:
	case <-time.After(time.Second):
		t.Fatal("backend did not exit")
	}
	select {
	case <-ended:
	case <-time.After(time.Second):
		t.Fatal("call did not exit")
	}
}
