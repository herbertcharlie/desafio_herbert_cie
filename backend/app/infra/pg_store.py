"""Almacén de conocimiento sobre PostgreSQL + pgvector (SQL explícito con SQLAlchemy Core)."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.domain.errors import DuplicateDocumentError
from app.domain.models import ChatMessage, ChunkDraft, Document, RetrievedChunk

_WORD = re.compile(r"\w+", re.UNICODE)

_CHUNK_SELECT = """
    SELECT c.id, c.document_id, d.filename, c.page_start, c.page_end, c.content
"""


def create_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, pool_size=10, max_overflow=10, pool_pre_ping=True)


def to_vector_literal(embedding: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in embedding) + "]"


def to_tsquery(question: str) -> str:
    """Convierte una pregunta en una tsquery OR segura (solo caracteres \\w)."""
    terms = {w.lower() for w in _WORD.findall(question) if len(w) > 2}
    return " | ".join(sorted(terms))


def _json(value: Any) -> Any:
    """asyncpg puede devolver JSONB como str según el codec configurado."""
    return json.loads(value) if isinstance(value, str) else value


def _document(row: Any) -> Document:
    return Document(
        id=row.id,
        filename=row.filename,
        sha256=row.sha256,
        pages=row.pages,
        chunk_count=row.chunk_count,
        created_at=row.created_at,
    )


def _chunk(row: Any, similarity: float | None) -> RetrievedChunk:
    return RetrievedChunk(
        id=row.id,
        document_id=row.document_id,
        filename=row.filename,
        page_start=row.page_start,
        page_end=row.page_end,
        content=row.content,
        similarity=similarity,
    )


class PgKnowledgeStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def ping(self) -> None:
        async with self._engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    async def find_document_by_hash(self, sha256: str) -> Document | None:
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    text("SELECT * FROM documents WHERE sha256 = :sha"), {"sha": sha256}
                )
            ).first()
        return _document(row) if row else None

    async def add_document(
        self,
        *,
        filename: str,
        sha256: str,
        pages: int,
        chunks: Sequence[ChunkDraft],
        embeddings: Sequence[Sequence[float]],
    ) -> Document:
        try:
            async with self._engine.begin() as conn:  # transacción: todo o nada
                row = (
                    await conn.execute(
                        text(
                            "INSERT INTO documents (filename, sha256, pages, chunk_count) "
                            "VALUES (:filename, :sha, :pages, :count) RETURNING *"
                        ),
                        {
                            "filename": filename,
                            "sha": sha256,
                            "pages": pages,
                            "count": len(chunks),
                        },
                    )
                ).one()
                await conn.execute(
                    text(
                        "INSERT INTO chunks (document_id, chunk_index, page_start, page_end, "
                        "content, embedding) VALUES (:doc, :idx, :ps, :pe, :content, "
                        "CAST(:emb AS vector))"
                    ),
                    [
                        {
                            "doc": row.id,
                            "idx": chunk.index,
                            "ps": chunk.page_start,
                            "pe": chunk.page_end,
                            "content": chunk.content,
                            "emb": to_vector_literal(embedding),
                        }
                        for chunk, embedding in zip(chunks, embeddings, strict=True)
                    ],
                )
        except IntegrityError as exc:
            raise DuplicateDocumentError("El documento ya existe.") from exc
        return _document(row)

    async def list_documents(self) -> list[Document]:
        async with self._engine.connect() as conn:
            rows = (await conn.execute(text("SELECT * FROM documents ORDER BY created_at"))).all()
        return [_document(r) for r in rows]

    async def delete_document(self, document_id: UUID) -> bool:
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text("DELETE FROM documents WHERE id = :id"), {"id": document_id}
            )
        return result.rowcount > 0

    async def search_vector(self, embedding: Sequence[float], k: int) -> list[RetrievedChunk]:
        sql = (
            _CHUNK_SELECT + ", 1 - (c.embedding <=> CAST(:q AS vector)) AS similarity "
            "FROM chunks c JOIN documents d ON d.id = c.document_id "
            "ORDER BY c.embedding <=> CAST(:q AS vector) LIMIT :k"
        )
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(text(sql), {"q": to_vector_literal(embedding), "k": k})
            ).all()
        return [_chunk(r, float(r.similarity)) for r in rows]

    async def search_text(self, query: str, k: int) -> list[RetrievedChunk]:
        tsquery = to_tsquery(query)
        if not tsquery:
            return []
        sql = (
            _CHUNK_SELECT + " FROM chunks c JOIN documents d ON d.id = c.document_id, "
            "to_tsquery('spanish', :tsq) AS q WHERE c.tsv @@ q "
            "ORDER BY ts_rank_cd(c.tsv, q) DESC LIMIT :k"
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(text(sql), {"tsq": tsquery, "k": k})).all()
        return [_chunk(r, None) for r in rows]


class PgChatHistory:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def add(self, message: ChatMessage) -> ChatMessage:
        async with self._engine.begin() as conn:
            row = (
                await conn.execute(
                    text(
                        "INSERT INTO chat_messages (session_id, question, answer, grounded, "
                        "sources, meta) VALUES (:sid, :q, :a, :g, CAST(:sources AS jsonb), "
                        "CAST(:meta AS jsonb)) RETURNING id, created_at"
                    ),
                    {
                        "sid": message.session_id,
                        "q": message.question,
                        "a": message.answer,
                        "g": message.grounded,
                        "sources": json.dumps(message.sources),
                        "meta": json.dumps(message.meta),
                    },
                )
            ).one()
        return ChatMessage(
            id=row.id,
            created_at=row.created_at,
            session_id=message.session_id,
            question=message.question,
            answer=message.answer,
            grounded=message.grounded,
            sources=message.sources,
            meta=message.meta,
        )

    async def list(self, session_id: str, limit: int = 100) -> list[ChatMessage]:
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT * FROM (SELECT * FROM chat_messages WHERE session_id = :sid "
                        "ORDER BY id DESC LIMIT :limit) recent ORDER BY id"
                    ),
                    {"sid": session_id, "limit": limit},
                )
            ).all()
        return [
            ChatMessage(
                id=r.id,
                session_id=r.session_id,
                question=r.question,
                answer=r.answer,
                grounded=r.grounded,
                sources=_json(r.sources),
                meta=_json(r.meta),
                created_at=r.created_at,
            )
            for r in rows
        ]
