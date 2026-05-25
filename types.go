package harness

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"google.golang.org/genai"
)

// Sentinel errors returned by the SDK.
var (
	ErrMaxTurns      = errors.New("harness: maximum turns exceeded")
	ErrSessionBusy   = errors.New("harness: session is processing a message")
	ErrSessionClosed = errors.New("harness: session is closed")
	ErrAgentClosed   = errors.New("harness: agent is closed")
	ErrAgentSealed   = errors.New("harness: cannot register tools after first use")
	ErrToolExists    = errors.New("harness: tool already registered")
	ErrToolNotFound  = errors.New("harness: tool not found")
	ErrMissingParam  = errors.New("harness: missing required parameter")
	ErrInvalidParam  = errors.New("harness: invalid parameter type")
	ErrConfigInvalid = errors.New("harness: invalid configuration")
)

// Config holds all configuration for creating an Agent.
// GoogleAPIKey is required. DatabaseURL is optional — only needed for Chat.
type Config struct {
	GoogleAPIKey     string
	DatabaseURL      string
	ModelName        string
	SystemPrompt     string
	MaxTurns         int
	MaxOutputTokens  int
	MaxToolOutputLen int
	IdleTimeout      time.Duration
	DBOptions        DBOptions
}

// DBOptions configures the Postgres connection pool and table naming.
type DBOptions struct {
	MaxConns    int
	TablePrefix string
	Schema      string
}

func (c *Config) setDefaults() {
	if c.ModelName == "" {
		c.ModelName = "gemini-2.0-flash"
	}
	if c.MaxTurns == 0 {
		c.MaxTurns = 25
	}
	if c.MaxOutputTokens == 0 {
		c.MaxOutputTokens = 8192
	}
	if c.MaxToolOutputLen == 0 {
		c.MaxToolOutputLen = 10000
	}
	if c.IdleTimeout == 0 {
		c.IdleTimeout = 5 * time.Minute
	}
	if c.DBOptions.MaxConns == 0 {
		c.DBOptions.MaxConns = 10
	}
	if c.DBOptions.TablePrefix == "" {
		c.DBOptions.TablePrefix = "harness_"
	}
	if c.DBOptions.Schema == "" {
		c.DBOptions.Schema = "public"
	}
}

func (c *Config) validate() error {
	if c.GoogleAPIKey == "" {
		return fmt.Errorf("%w: GoogleAPIKey is required", ErrConfigInvalid)
	}
	return nil
}

// ParamType represents the JSON Schema type of a tool parameter.
type ParamType string

const (
	TypeString  ParamType = "string"
	TypeNumber  ParamType = "number"
	TypeInteger ParamType = "integer"
	TypeBoolean ParamType = "boolean"
	TypeArray   ParamType = "array"
	TypeObject  ParamType = "object"
)

// Param defines a single parameter in a Tool's schema.
type Param struct {
	Type        ParamType
	Description string
	Required    bool
	Enum        []string
	Items       *Param
	Properties  map[string]Param
}

// Tool defines a function the LLM can call during the agentic loop.
// Handler is invoked with the parsed arguments when the model calls this tool.
type Tool struct {
	Name        string
	Description string
	Parameters  map[string]Param
	Handler     func(ctx context.Context, params map[string]any) (string, error)
}

// toolRegistry is a concurrency-safe registry of tools, sealed after first use.
type toolRegistry struct {
	mu    sync.RWMutex
	tools map[string]Tool
}

func newToolRegistry() *toolRegistry {
	return &toolRegistry{tools: make(map[string]Tool)}
}

func (r *toolRegistry) Register(tool Tool) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.tools[tool.Name]; exists {
		return fmt.Errorf("%w: %s", ErrToolExists, tool.Name)
	}
	r.tools[tool.Name] = tool
	return nil
}

func (r *toolRegistry) Get(name string) (Tool, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	t, ok := r.tools[name]
	return t, ok
}

func (r *toolRegistry) All() []Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	tools := make([]Tool, 0, len(r.tools))
	for _, t := range r.tools {
		tools = append(tools, t)
	}
	return tools
}

// Event is the streaming event interface returned by GenerateStreamResponse and Chat.
type Event interface {
	eventType() string
}

// Partial is a streaming chunk from the model or a tool response result.
type Partial struct {
	Response  *genai.GenerateContentResponse `json:"response"`
	SessionID uuid.UUID                      `json:"session_id"`
	MessageID uuid.UUID                      `json:"message_id"`
}

func (Partial) eventType() string { return "partial" }

// Complete is the final assembled response with coalesced text and usage metadata.
type Complete struct {
	Response  *genai.GenerateContentResponse `json:"response"`
	SessionID uuid.UUID                      `json:"session_id"`
	MessageID uuid.UUID                      `json:"message_id"`
}

func (Complete) eventType() string { return "complete" }

// Error wraps a fatal error that terminates generation.
type Error struct {
	Err error `json:"-"`
}

func (Error) eventType() string { return "error" }

// conversation holds per-connection state, carried in context.Context via withConversation.
type conversation struct {
	id        uuid.UUID
	userID    string
	history   []*genai.Content
	messageID uuid.UUID
	title     string
}

type ctxKey struct{}

// withConversation stores a conversation pointer in the context.
func withConversation(ctx context.Context, conv *conversation) context.Context {
	return context.WithValue(ctx, ctxKey{}, conv)
}

// conversationFrom retrieves the conversation pointer from the context.
func conversationFrom(ctx context.Context) *conversation {
	return ctx.Value(ctxKey{}).(*conversation)
}

// toolResult holds the output of a single tool execution.
type toolResult struct {
	name    string
	result  string
	isError bool
}

// TextContent creates a user-role genai.Content from plain text.
func TextContent(text string) *genai.Content {
	return genai.NewContentFromText(text, "user")
}

// TextParts creates a single-element genai.Part slice from plain text.
func TextParts(text string) []*genai.Part {
	return []*genai.Part{{Text: text}}
}
