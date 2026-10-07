-- Esquema inicial. Postgres lo ejecuta una sola vez, al crear el volumen de datos
-- (docker-entrypoint-initdb.d). Para reiniciar desde cero: `docker compose down -v`.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename    TEXT        NOT NULL,
    sha256      CHAR(64)    NOT NULL UNIQUE,   -- evita ingestar dos veces el mismo archivo
    pages       INTEGER     NOT NULL,
    chunk_count INTEGER     NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID        NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index INTEGER     NOT NULL,
    page_start  INTEGER     NOT NULL,
    page_end    INTEGER     NOT NULL,
    content     TEXT        NOT NULL,
    -- Debe coincidir con EMBEDDING_DIM (1536 = text-embedding-3-small).
    embedding   vector(1536) NOT NULL,
    -- Columna léxica para la búsqueda híbrida (BM25-like con ts_rank_cd).
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('spanish', content)) STORED,
    UNIQUE (document_id, chunk_index)
);

-- Búsqueda vectorial aproximada por coseno.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
-- Búsqueda léxica.
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);

CREATE TABLE IF NOT EXISTS chat_messages (
    id         BIGSERIAL PRIMARY KEY,          -- también define el orden cronológico
    session_id TEXT        NOT NULL,
    question   TEXT        NOT NULL,
    answer     TEXT        NOT NULL,
    grounded   BOOLEAN     NOT NULL,
    sources    JSONB       NOT NULL DEFAULT '[]'::jsonb,
    meta       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chat_messages_session_idx ON chat_messages (session_id, id);
