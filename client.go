// Package harness provides an SDK for building agentic AI features on top of Google Gemini.
package harness

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/Harshal1000/ai-harness/internal/google"
	"github.com/Harshal1000/ai-harness/internal/store"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"google.golang.org/genai"
)

// Agent is the top-level object for interacting with the Harness SDK.
// Create with NewAgent, register tools, then call GenerateStreamResponse or Chat.
type Agent struct {
	config   Config
	registry *toolRegistry
	store    *store.Service
	llm      *google.Client

	mu     sync.Mutex
	sealed bool
	closed bool
}

// NewAgent creates a new Agent with the given configuration.
// Connects to Postgres (if DatabaseURL is set) and initializes the GenAI client.
// New creates a new Agent with the given configuration.
// Connects to Postgres (if DatabaseURL is set) and initializes the GenAI client.
func New(cfg Config) (*Agent, error) {
	cfg.setDefaults()
	if err := cfg.validate(); err != nil {
		return nil, err
	}

	ctx := context.Background()

	var st *store.Service
	if cfg.DatabaseURL != "" {
		var err error
		st, err = store.New(ctx, cfg.DatabaseURL, cfg.DBOptions.MaxConns, cfg.DBOptions.TablePrefix)
		if err != nil {
			return nil, err
		}
	}

	lc, err := google.NewClient(ctx, cfg.GoogleAPIKey, cfg.ModelName)
	if err != nil {
		if st != nil {
			st.Close()
		}
		return nil, err
	}

	return &Agent{
		config:   cfg,
		registry: newToolRegistry(),
		store:    st,
		llm:      lc,
	}, nil
}

// RegisterTool adds a tool that the LLM can invoke during generation.
// Must be called before the first GenerateStreamResponse or Chat call.
func (a *Agent) RegisterTool(tool Tool) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.sealed {
		return ErrAgentSealed
	}
	return a.registry.Register(tool)
}

// GenerateStreamResponse makes a single streaming LLM call with the given content.
// No session management, no persistence, no agentic loop — just one request/response.
// Use this for lightweight, stateless operations like summarization or feedback.
func (a *Agent) GenerateStreamResponse(ctx context.Context, content *genai.Content) (<-chan Event, error) {
	a.mu.Lock()
	if a.closed {
		a.mu.Unlock()
		return nil, ErrAgentClosed
	}
	a.sealed = true
	a.mu.Unlock()

	eventCh := make(chan Event, 64)
	go a.runStream(ctx, content, eventCh)
	return eventCh, nil
}

// Chat handles a persistent, multi-turn conversation over a WebSocket connection.
// Loads session history from Postgres, runs the agentic loop (model → tools → model)
// for each message, persists every exchange, and streams events back over WebSocket.
// Returns when the WebSocket disconnects.
func (a *Agent) Chat(conn *websocket.Conn, sessionID uuid.UUID, userID string) error {
	if a.store == nil {
		return fmt.Errorf("%w: DatabaseURL is required for Chat", ErrConfigInvalid)
	}

	a.mu.Lock()
	if a.closed {
		a.mu.Unlock()
		return ErrAgentClosed
	}
	a.sealed = true
	a.mu.Unlock()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	conv, err := a.loadSession(ctx, sessionID, userID)
	if err != nil {
		return err
	}
	ctx = withConversation(ctx, conv)

	for {
		_, raw, err := conn.ReadMessage()
		if err != nil {
			return err
		}

		var msg wsIncoming
		if err := json.Unmarshal(raw, &msg); err != nil {
			if writeErr := conn.WriteJSON(wsEvent{Type: "error", Message: "invalid message format"}); writeErr != nil {
				return writeErr
			}
			continue
		}

		content := &genai.Content{Role: "user", Parts: msg.Parts}
		eventCh := make(chan Event, 64)
		go a.runLoop(ctx, content, eventCh, true)

		for event := range eventCh {
			if writeErr := conn.WriteJSON(eventToWS(event)); writeErr != nil {
				return writeErr
			}
		}
	}
}

// Close shuts down the agent, closing database and LLM connections.
func (a *Agent) Close() error {
	a.mu.Lock()
	a.closed = true
	a.mu.Unlock()

	if a.store != nil {
		a.store.Close()
	}
	a.llm.Close()
	return nil
}

// loadSession loads an existing session's history from Postgres, or creates a new one.
func (a *Agent) loadSession(ctx context.Context, sessionID uuid.UUID, userID string) (*conversation, error) {
	history, err := a.store.LoadEvents(ctx, sessionID)
	if err != nil {
		return nil, err
	}

	if err := a.store.EnsureSession(ctx, sessionID, userID, "", a.config.ModelName); err != nil {
		return nil, err
	}

	return &conversation{
		id:      sessionID,
		userID:  userID,
		history: history,
	}, nil
}

// --- WebSocket protocol ---

type wsIncoming struct {
	Parts []*genai.Part `json:"parts"`
}

type wsEvent struct {
	Type      string          `json:"type"`
	SessionID string          `json:"session_id,omitempty"`
	MessageID string          `json:"message_id,omitempty"`
	Response  json.RawMessage `json:"response,omitempty"`
	Message   string          `json:"message,omitempty"`
}

// eventToWS converts an internal Event to the WebSocket wire format.
func eventToWS(e Event) wsEvent {
	switch v := e.(type) {
	case Partial:
		raw, _ := json.Marshal(v.Response)
		return wsEvent{
			Type:      "partial",
			SessionID: v.SessionID.String(),
			MessageID: v.MessageID.String(),
			Response:  raw,
		}
	case Complete:
		raw, _ := json.Marshal(v.Response)
		return wsEvent{
			Type:      "complete",
			SessionID: v.SessionID.String(),
			MessageID: v.MessageID.String(),
			Response:  raw,
		}
	case Error:
		return wsEvent{Type: "error", Message: v.Err.Error()}
	default:
		return wsEvent{Type: "unknown"}
	}
}
