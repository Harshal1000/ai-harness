# ai-harness

Go SDK for building AI-powered applications. Drop-in agentic AI for any Go project — your backend, your API, your product.

There are plenty of AI harnesses out there for terminals and desktop apps. This one is for **your code**. Import it, wire it into your Go application, and ship AI features that your users interact with — not just you in a terminal.

## What it does

- **Single-shot generation** — one LLM call, streaming response. Summarize, classify, extract, generate feedback. No overhead.
- **Multi-turn chat with tools** — persistent conversations over WebSocket. The model calls your tools, gets results, reasons, and responds. Full agentic loop.
- **Session persistence** — conversation history stored in Postgres. Users disconnect, come back, pick up where they left off.
- **Your tools, your logic** — register any Go function as a tool. The model decides when to call it. You decide what it does.

## Install

```bash
go get github.com/Harshal1000/ai-harness
```

## Quick start

```go
package main

import (
    "context"
    "fmt"
    "log"
    "net/http"

    "github.com/Harshal1000/ai-harness"
    "github.com/google/uuid"
    "github.com/gorilla/websocket"
)

func main() {
    h, err := harness.New(harness.Config{
        GoogleAPIKey: "your-api-key",
        DatabaseURL:  "postgresql://user:pass@localhost:5432/mydb",
        SystemPrompt: "You are a helpful assistant.",
    })
    if err != nil {
        log.Fatal(err)
    }
    defer h.Close()

    // Register a tool the model can call
    h.RegisterTool(harness.Tool{
        Name:        "lookup_order",
        Description: "Look up an order by ID",
        Parameters: map[string]harness.Param{
            "order_id": {Type: harness.TypeString, Description: "The order ID", Required: true},
        },
        Handler: func(ctx context.Context, params map[string]any) (string, error) {
            id := params["order_id"].(string)
            return fmt.Sprintf(`{"order_id": "%s", "status": "shipped"}`, id), nil
        },
    })

    // Single-shot: no session, no persistence
    events, _ := h.GenerateStreamResponse(context.Background(), harness.TextContent("Summarize this quarter's metrics"))
    for e := range events {
        // stream to your HTTP response, queue, or anywhere
        _ = e
    }

    // Chat: persistent multi-turn over WebSocket
    upgrader := websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }}
    http.HandleFunc("/chat", func(w http.ResponseWriter, r *http.Request) {
        conn, _ := upgrader.Upgrade(w, r, nil)
        defer conn.Close()
        h.Chat(conn, uuid.New(), r.URL.Query().Get("user_id"))
    })

    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

## Two APIs, two use cases

### `GenerateStreamResponse` — fire and forget

Single streaming LLM call. No session, no database, no agentic loop. Just content in, response out.

Use it for:
- Summarization
- Classification
- Feedback generation
- Content extraction
- Anything that doesn't need back-and-forth

```go
events, err := h.GenerateStreamResponse(ctx, harness.TextContent("Rate this review: ..."))
for e := range events {
    if c, ok := e.(harness.Complete); ok {
        // final response
    }
}
```

Database not required. Works with just an API key.

### `Chat` — persistent multi-turn conversation

Full agentic session over WebSocket. The model can call tools, get results, reason, and respond across multiple turns. Every message persisted to Postgres.

```go
// Each WebSocket connection gets a session
h.Chat(conn, sessionID, userID)
// Disconnect → reconnect with same sessionID → full history restored
```

The agentic loop runs until the model produces a final text response:
```
user message → model → tool call → tool result → model → tool call → ... → final response
```

All persisted. All streamed.

## Configuration

```go
harness.Config{
    GoogleAPIKey:     "required",
    DatabaseURL:      "postgresql://... (optional, required for Chat)",
    ModelName:        "gemini-2.0-flash",   // default
    SystemPrompt:     "You are ...",
    MaxTurns:         25,                   // max agentic loop iterations
    MaxOutputTokens:  8192,
    MaxToolOutputLen: 10000,                // truncate long tool outputs
    DBOptions: harness.DBOptions{
        MaxConns:    10,
        TablePrefix: "harness_",            // table name prefix
    },
}
```

## WebSocket protocol

Client sends:
```json
{"parts": [{"text": "What's the status of order 12345?"}]}
```

Server streams back:
```json
{"type": "partial",  "session_id": "...", "message_id": "...", "response": {...}}
{"type": "complete", "session_id": "...", "message_id": "...", "response": {...}}
{"type": "error",    "message": "..."}
```

## Project structure

```
types.go                        — all types, errors, constants
client.go                       — public API (New, RegisterTool, GenerateStreamResponse, Chat, Close)
runner.go                       — agentic loop, tool execution
internal/google/genai.go        — Gemini streaming client
internal/google/convertor.go    — tool schema conversion
internal/store/service.go       — Postgres session/event persistence
internal/store/schema.sql       — auto-migrated DDL
```

## Requirements

- Go 1.23+
- Google Gemini API key
- Postgres (only for `Chat`)

## License

MIT
