package harness

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	"github.com/Harshal1000/ai-harness/internal/google"
	"github.com/google/uuid"
	"golang.org/x/sync/errgroup"
	"google.golang.org/genai"
)

// runStream performs a single streaming LLM call with no history, tools, or persistence.
func (a *Agent) runStream(ctx context.Context, content *genai.Content, eventCh chan<- Event) {
	defer close(eventCh)

	history := []*genai.Content{content}
	msgID, _ := uuid.NewV7()

	emitChunk := func(resp *genai.GenerateContentResponse) {
		eventCh <- Partial{Response: resp, MessageID: msgID}
	}

	result, err := a.llm.StreamChat(ctx, a.config.SystemPrompt, history, nil, a.config.MaxOutputTokens, emitChunk)
	if err != nil {
		eventCh <- Error{Err: err}
		return
	}

	eventCh <- Complete{
		Response: &genai.GenerateContentResponse{
			Candidates:    []*genai.Candidate{{Content: result.Content}},
			UsageMetadata: result.Usage,
		},
		MessageID: msgID,
	}
}

// runLoop is the core agentic loop used by Chat. It prepares the user message,
// calls the LLM, executes any tool calls, and repeats until the model produces
// a final text response or MaxTurns is reached. Each exchange is persisted to
// Postgres when persist is true. Conversation state is carried in ctx.
func (a *Agent) runLoop(ctx context.Context, content *genai.Content, eventCh chan<- Event, persist bool) {
	defer close(eventCh)

	conv := conversationFrom(ctx)
	conv.messageID, _ = uuid.NewV7()

	if persist && conv.title == "" {
		for _, p := range content.Parts {
			if p.Text != "" {
				conv.title = truncate(p.Text, 80)
				break
			}
		}
	}

	conv.history = append(conv.history, content)
	if persist && a.store != nil {
		if err := a.store.AppendEvent(ctx, conv.id, conv.messageID, content, nil); err != nil {
			log.Printf("harness: failed to store user event: %v", err)
		}
	}

	effectiveMaxTokens := a.config.MaxOutputTokens
	tools := a.buildTools()

	for turn := 0; turn < a.config.MaxTurns; turn++ {
		if ctx.Err() != nil {
			return
		}

		emitChunk := func(resp *genai.GenerateContentResponse) {
			eventCh <- Partial{
				Response:  resp,
				SessionID: conv.id,
				MessageID: conv.messageID,
			}
		}

		result, err := a.llm.StreamChat(ctx, a.config.SystemPrompt, conv.history, tools, effectiveMaxTokens, emitChunk)

		if err != nil {
			var le *google.Error
			if !errors.As(err, &le) {
				eventCh <- Error{Err: err}
				return
			}

			if le.Code == 400 && !le.Fatal {
				effectiveMaxTokens = effectiveMaxTokens * 3 / 4
				if effectiveMaxTokens < 1024 {
					eventCh <- Error{Err: err}
					return
				}
				turn--
				continue
			}

			if le.Retryable {
				var retryResult *google.StreamResult
				for attempt := 1; attempt <= 3; attempt++ {
					select {
					case <-ctx.Done():
						return
					case <-time.After(time.Duration(attempt*attempt) * time.Second):
					}
					retryResult, err = a.llm.StreamChat(ctx, a.config.SystemPrompt, conv.history, tools, effectiveMaxTokens, emitChunk)
					if err == nil {
						result = retryResult
						break
					}
				}
				if result == nil {
					eventCh <- Error{Err: err}
					return
				}
			} else {
				eventCh <- Error{Err: err}
				return
			}
		}

		conv.history = append(conv.history, result.Content)
		if persist && a.store != nil {
			if err := a.store.AppendEvent(ctx, conv.id, conv.messageID, result.Content, result.Usage); err != nil {
				log.Printf("harness: failed to store model event: %v", err)
			}
		}

		var calls []*genai.FunctionCall
		for _, p := range result.Content.Parts {
			if p.FunctionCall != nil {
				calls = append(calls, p.FunctionCall)
			}
		}

		if len(calls) == 0 {
			fullResp := &genai.GenerateContentResponse{
				Candidates:    []*genai.Candidate{{Content: result.Content}},
				UsageMetadata: result.Usage,
			}
			eventCh <- Complete{
				Response:  fullResp,
				SessionID: conv.id,
				MessageID: conv.messageID,
			}
			return
		}

		toolResults := a.executeToolsConcurrently(ctx, calls, a.config.MaxToolOutputLen)

		var responseParts []*genai.Part
		for _, r := range toolResults {
			resp := map[string]any{"result": r.result}
			if r.isError {
				resp["error"] = true
			}
			responseParts = append(responseParts, &genai.Part{
				FunctionResponse: &genai.FunctionResponse{
					Name:     r.name,
					Response: resp,
				},
			})
		}

		toolContent := &genai.Content{Role: "user", Parts: responseParts}
		conv.history = append(conv.history, toolContent)
		if persist && a.store != nil {
			if err := a.store.AppendEvent(ctx, conv.id, conv.messageID, toolContent, nil); err != nil {
				log.Printf("harness: failed to store tool response event: %v", err)
			}
		}

		eventCh <- Partial{
			Response: &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{{Content: toolContent}},
			},
			SessionID: conv.id,
			MessageID: conv.messageID,
		}
	}

	eventCh <- Error{Err: fmt.Errorf("%w: reached %d turns", ErrMaxTurns, a.config.MaxTurns)}
}

// executeTool runs a single registered tool and returns the result.
func (a *Agent) executeTool(ctx context.Context, fc *genai.FunctionCall, maxOutputLen int) toolResult {
	tool, ok := a.registry.Get(fc.Name)
	if !ok {
		return toolResult{
			name:    fc.Name,
			result:  fmt.Sprintf("error: tool %q not found", fc.Name),
			isError: true,
		}
	}

	for name, param := range tool.Parameters {
		if param.Required {
			if _, ok := fc.Args[name]; !ok {
				return toolResult{
					name:    fc.Name,
					result:  fmt.Sprintf("error: %s: %s", ErrMissingParam, name),
					isError: true,
				}
			}
		}
	}

	output, err := tool.Handler(ctx, fc.Args)
	if err != nil {
		return toolResult{
			name:    fc.Name,
			result:  fmt.Sprintf("error: %s", err.Error()),
			isError: true,
		}
	}

	if len(output) > maxOutputLen {
		output = output[:maxOutputLen] + "\n...[truncated]"
	}

	return toolResult{name: fc.Name, result: output}
}

// executeToolsConcurrently runs multiple tool calls in parallel using errgroup.
func (a *Agent) executeToolsConcurrently(ctx context.Context, calls []*genai.FunctionCall, maxOutputLen int) []toolResult {
	results := make([]toolResult, len(calls))

	if len(calls) == 1 {
		results[0] = a.executeTool(ctx, calls[0], maxOutputLen)
		return results
	}

	g, gctx := errgroup.WithContext(ctx)
	for i, fc := range calls {
		g.Go(func() error {
			results[i] = a.executeTool(gctx, fc, maxOutputLen)
			return nil
		})
	}
	g.Wait()

	return results
}

// buildTools converts registered Tool definitions to genai.Tool format via the google convertor.
func (a *Agent) buildTools() []*genai.Tool {
	all := a.registry.All()
	if len(all) == 0 {
		return nil
	}
	defs := make([]google.ToolDef, len(all))
	for i, t := range all {
		defs[i] = google.ToolDef{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  convertParamMap(t.Parameters),
		}
	}
	return google.BuildTools(defs)
}

func convertParamMap(params map[string]Param) map[string]google.ParamDef {
	result := make(map[string]google.ParamDef, len(params))
	for name, p := range params {
		result[name] = convertParam(p)
	}
	return result
}

func convertParam(p Param) google.ParamDef {
	def := google.ParamDef{
		Type:        string(p.Type),
		Description: p.Description,
		Required:    p.Required,
		Enum:        p.Enum,
	}
	if p.Items != nil {
		items := convertParam(*p.Items)
		def.Items = &items
	}
	if len(p.Properties) > 0 {
		def.Properties = convertParamMap(p.Properties)
	}
	return def
}

func truncate(str string, maxLen int) string {
	if len(str) <= maxLen {
		return str
	}
	return str[:maxLen]
}
