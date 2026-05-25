CREATE TABLE IF NOT EXISTS {prefix}sessions (
    id            UUID PRIMARY KEY,
    user_id       TEXT NOT NULL,
    title         TEXT,
    model         TEXT NOT NULL,
    system_prompt TEXT,
    metadata      JSONB NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {prefix}events (
    id          UUID PRIMARY KEY,
    session_id  UUID NOT NULL REFERENCES {prefix}sessions(id) ON DELETE CASCADE,
    message_id  UUID NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user', 'model')),
    content     JSONB NOT NULL,
    usage       JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_{prefix}events_session ON {prefix}events(session_id, id);
CREATE INDEX IF NOT EXISTS idx_{prefix}sessions_user ON {prefix}sessions(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_{prefix}events_message ON {prefix}events(message_id);
CREATE INDEX IF NOT EXISTS idx_{prefix}sessions_updated ON {prefix}sessions(updated_at);
