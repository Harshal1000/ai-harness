// Package store handles Postgres persistence for chat sessions and events.
package store

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"google.golang.org/genai"
)

//go:embed schema.sql
var schemaDDL string

// Service manages the Postgres connection pool and provides CRUD operations
// for sessions and events. Schema is auto-migrated on creation.
type Service struct {
	pool   *pgxpool.Pool
	prefix string
}

// New creates a new store Service, connects to Postgres, and runs schema migration.
func New(ctx context.Context, databaseURL string, maxConns int, tablePrefix string) (*Service, error) {
	cfg, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		return nil, fmt.Errorf("harness: invalid database URL: %w", err)
	}
	cfg.MaxConns = int32(maxConns)

	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("harness: failed to connect to database: %w", err)
	}

	s := &Service{pool: pool, prefix: tablePrefix}

	if err := s.migrate(ctx); err != nil {
		pool.Close()
		return nil, err
	}

	return s, nil
}

func (s *Service) migrate(ctx context.Context) error {
	ddl := strings.ReplaceAll(schemaDDL, "{prefix}", s.prefix)
	_, err := s.pool.Exec(ctx, ddl)
	if err != nil {
		return fmt.Errorf("harness: migration failed: %w", err)
	}
	return nil
}

// EnsureSession creates a session row or touches its updated_at timestamp.
func (s *Service) EnsureSession(ctx context.Context, sessionID uuid.UUID, userID, title, model string) error {
	query := fmt.Sprintf(`
		INSERT INTO %ssessions (id, user_id, title, model)
		VALUES ($1, $2, $3, $4)
		ON CONFLICT (id) DO UPDATE SET updated_at = NOW()`,
		s.prefix,
	)
	_, err := s.pool.Exec(ctx, query, sessionID, userID, title, model)
	if err != nil {
		return fmt.Errorf("harness: failed to upsert session: %w", err)
	}
	return nil
}

// LoadEvents loads all conversation messages for a session, ordered by creation time.
func (s *Service) LoadEvents(ctx context.Context, sessionID uuid.UUID) ([]*genai.Content, error) {
	query := fmt.Sprintf(
		`SELECT role, content FROM %sevents WHERE session_id = $1 ORDER BY id ASC`,
		s.prefix,
	)
	rows, err := s.pool.Query(ctx, query, sessionID)
	if err != nil {
		return nil, fmt.Errorf("harness: failed to load events: %w", err)
	}
	defer rows.Close()

	var contents []*genai.Content
	for rows.Next() {
		var role string
		var raw []byte
		if err := rows.Scan(&role, &raw); err != nil {
			return nil, fmt.Errorf("harness: failed to scan event: %w", err)
		}
		var c genai.Content
		if err := json.Unmarshal(raw, &c); err != nil {
			return nil, fmt.Errorf("harness: failed to unmarshal content: %w", err)
		}
		c.Role = role
		contents = append(contents, &c)
	}
	return contents, rows.Err()
}

// AppendEvent persists a single conversation event (user message, model response, or tool result).
func (s *Service) AppendEvent(ctx context.Context, sessionID, messageID uuid.UUID, content *genai.Content, usage *genai.GenerateContentResponseUsageMetadata) error {
	eventID, _ := uuid.NewV7()
	raw, err := json.Marshal(content)
	if err != nil {
		return fmt.Errorf("harness: failed to marshal content: %w", err)
	}

	var usageRaw []byte
	if usage != nil {
		usageRaw, _ = json.Marshal(usage)
	}

	query := fmt.Sprintf(
		`INSERT INTO %sevents (id, session_id, message_id, role, content, usage) VALUES ($1, $2, $3, $4, $5, $6)`,
		s.prefix,
	)
	_, err = s.pool.Exec(ctx, query, eventID, sessionID, messageID, content.Role, raw, usageRaw)
	if err != nil {
		return fmt.Errorf("harness: failed to insert event: %w", err)
	}
	return nil
}

// Close shuts down the Postgres connection pool.
func (s *Service) Close() {
	s.pool.Close()
}
