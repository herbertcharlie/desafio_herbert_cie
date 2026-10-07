"""Puertos (interfaces) que la lógica de aplicación necesita.

La lógica de negocio solo conoce estos Protocols; las implementaciones concretas
(OpenAI, Postgres/pgvector...) viven en `app/infra` y se inyectan en `app/container.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from app.domain.models import (
    ChatMessage,
    ChunkDraft,
    Document,
    LLMResponse,
    RetrievedChunk,
)


class Embedder(Protocol):
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class LLMProvider(Protocol):
    async def complete(self, *, system: str, user: str, json_mode: bool = False) -> LLMResponse: ...


class KnowledgeStore(Protocol):
    """Documentos + fragmentos + índices de búsqueda."""

    async def ping(self) -> None: ...

    async def find_document_by_hash(self, sha256: str) -> Document | None: ...

    async def add_document(
        self,
        *,
        filename: str,
        sha256: str,
        pages: int,
        chunks: Sequence[ChunkDraft],
        embeddings: Sequence[Sequence[float]],
    ) -> Document:
        """Persiste documento y fragmentos de forma atómica.

        Lanza `DuplicateDocumentError` si el hash ya existe.
        """
        ...

    async def list_documents(self) -> list[Document]: ...

    async def delete_document(self, document_id: UUID) -> bool: ...

    async def search_vector(self, embedding: Sequence[float], k: int) -> list[RetrievedChunk]: ...

    async def search_text(self, query: str, k: int) -> list[RetrievedChunk]: ...


class ChatHistory(Protocol):
    async def add(self, message: ChatMessage) -> ChatMessage: ...

    async def list(self, session_id: str, limit: int = 100) -> list[ChatMessage]: ...
