// Package google wraps the Google GenAI (Gemini) SDK for streaming chat completions.
package google

import (
	"context"
	"fmt"
	"strings"

	"google.golang.org/genai"
)

// Client wraps the Google GenAI client for a specific model.
type Client struct {
	client *genai.Client
	model  string
}

// StreamResult holds the coalesced model response and usage metadata from a streaming call.
type StreamResult struct {
	Content *genai.Content
	Usage   *genai.GenerateContentResponseUsageMetadata
}

// NewClient creates a new GenAI client using the Gemini API backend.
func NewClient(ctx context.Context, apiKey, model string) (*Client, error) {
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, fmt.Errorf("harness: failed to create genai client: %w", err)
	}
	return &Client{client: client, model: model}, nil
}

// StreamChat sends a streaming chat request to the model and calls emitChunk for each response chunk.
// Returns the coalesced result with all text parts merged.
func (c *Client) StreamChat(
	ctx context.Context,
	systemPrompt string,
	history []*genai.Content,
	tools []*genai.Tool,
	maxOutputTokens int,
	emitChunk func(*genai.GenerateContentResponse),
) (*StreamResult, error) {
	config := &genai.GenerateContentConfig{
		MaxOutputTokens: genai.Ptr(int32(maxOutputTokens)),
	}
	if systemPrompt != "" {
		config.SystemInstruction = &genai.Content{
			Parts: []*genai.Part{{Text: systemPrompt}},
		}
	}
	if len(tools) > 0 {
		config.Tools = tools
	}

	result := c.client.Models.GenerateContentStream(ctx, c.model, history, config)

	var allParts []*genai.Part
	var lastUsage *genai.GenerateContentResponseUsageMetadata

	for resp, err := range result {
		if err != nil {
			return nil, ClassifyError(err)
		}
		emitChunk(resp)

		if resp.UsageMetadata != nil {
			lastUsage = resp.UsageMetadata
		}
		for _, candidate := range resp.Candidates {
			if candidate.Content == nil {
				continue
			}
			allParts = append(allParts, candidate.Content.Parts...)
		}
	}

	if len(allParts) == 0 {
		return nil, &Error{Code: 0, Message: "empty response from model", Retryable: false, Fatal: true}
	}

	return &StreamResult{
		Content: &genai.Content{Role: "model", Parts: coalesceParts(allParts)},
		Usage:   lastUsage,
	}, nil
}

// coalesceParts merges adjacent text-only parts into single parts.
func coalesceParts(parts []*genai.Part) []*genai.Part {
	if len(parts) <= 1 {
		return parts
	}
	var result []*genai.Part
	var buf strings.Builder
	for _, p := range parts {
		isTextOnly := p.Text != "" && p.FunctionCall == nil && p.FunctionResponse == nil && p.InlineData == nil
		if isTextOnly {
			buf.WriteString(p.Text)
			continue
		}
		if buf.Len() > 0 {
			result = append(result, &genai.Part{Text: buf.String()})
			buf.Reset()
		}
		result = append(result, p)
	}
	if buf.Len() > 0 {
		result = append(result, &genai.Part{Text: buf.String()})
	}
	return result
}

// Close is a no-op; the underlying genai client does not require explicit cleanup.
func (c *Client) Close() {}

// Error represents a classified GenAI API error with retry and severity hints.
type Error struct {
	Code      int
	Message   string
	Retryable bool
	Fatal     bool
}

func (e *Error) Error() string {
	return fmt.Sprintf("harness: genai error (%d): %s", e.Code, e.Message)
}

// ClassifyError maps a raw GenAI error to a structured Error with retry/severity hints.
func ClassifyError(err error) *Error {
	msg := err.Error()
	switch {
	case strings.Contains(msg, "429") || strings.Contains(msg, "RESOURCE_EXHAUSTED"):
		return &Error{Code: 429, Message: msg, Retryable: true}
	case strings.Contains(msg, "500") || strings.Contains(msg, "INTERNAL"):
		return &Error{Code: 500, Message: msg, Retryable: true}
	case strings.Contains(msg, "503") || strings.Contains(msg, "UNAVAILABLE"):
		return &Error{Code: 503, Message: msg, Retryable: true}
	case strings.Contains(msg, "400") || strings.Contains(msg, "INVALID_ARGUMENT"):
		if strings.Contains(msg, "token") || strings.Contains(msg, "length") || strings.Contains(msg, "too long") {
			return &Error{Code: 400, Message: msg, Retryable: false, Fatal: false}
		}
		return &Error{Code: 400, Message: msg, Retryable: false, Fatal: true}
	case strings.Contains(msg, "401") || strings.Contains(msg, "UNAUTHENTICATED"):
		return &Error{Code: 401, Message: msg, Retryable: false, Fatal: true}
	case strings.Contains(msg, "403") || strings.Contains(msg, "PERMISSION_DENIED"):
		return &Error{Code: 403, Message: msg, Retryable: false, Fatal: true}
	default:
		return &Error{Code: 0, Message: msg, Retryable: false, Fatal: true}
	}
}
