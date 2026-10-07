"""Dobles de prueba en memoria que implementan los puertos del dominio."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

from app.domain.errors import DuplicateDocumentError
from app.domain.models import (
    ChatMessage,
    ChunkDraft,
    Document,
    LLMResponse,
    RetrievedChunk,
)


def make_chunk(
    content: str = "Texto de ejemplo.",
    *,
    similarity: float | None = 0.8,
    filename: str = "guia.pdf",
    page: int = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        id=uuid4(),
        document_id=uuid4(),
        filename=filename,
        page_start=page,
        page_end=page,
        content=content,
        similarity=similarity,
    )


class FakeEmbedder:
    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [1.0, 0.0]


class FakeLLM:
    """Devuelve una respuesta JSON preprogramada y registra las llamadas."""

    def __init__(
        self,
        *,
        sufficient: bool = True,
        answer: str = "Respuesta fundamentada [1].",
        citations: list[int] | None = None,
        raw: str | None = None,
        delay: float = 0.0,
    ) -> None:
        self._raw = raw or json.dumps(
            {
                "sufficient": sufficient,
                "answer": answer,
                "citations": [1] if citations is None else citations,
            }
        )
        self._delay = delay
        self.calls: list[dict] = []
        self.active = 0
        self.max_active = 0

    async def complete(self, *, system: str, user: str, json_mode: bool = False) -> LLMResponse:
        self.calls.append({"system": system, "user": user, "json_mode": json_mode})
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
        finally:
            self.active -= 1
        return LLMResponse(text=self._raw, model="fake-llm", prompt_tokens=10, completion_tokens=5)


class FakeStore:
    def __init__(
        self,
        vector_hits: list[RetrievedChunk] | None = None,
        text_hits: list[RetrievedChunk] | None = None,
    ) -> None:
        self.vector_hits = vector_hits or []
        self.text_hits = text_hits or []
        self.documents: dict[str, Document] = {}
        self.added: list[dict] = []
        self.text_searches = 0
        self.raise_duplicate_on_add = False

    async def ping(self) -> None:
        return None

    async def find_document_by_hash(self, sha256: str) -> Document | None:
        return self.documents.get(sha256)

    async def add_document(
        self,
        *,
        filename: str,
        sha256: str,
        pages: int,
        chunks: Sequence[ChunkDraft],
        embeddings: Sequence[Sequence[float]],
    ) -> Document:
        if self.raise_duplicate_on_add:
            self.raise_duplicate_on_add = False
            self.documents[sha256] = self._doc(filename, sha256, pages, len(chunks))
            raise DuplicateDocumentError("duplicado")
        doc = self._doc(filename, sha256, pages, len(chunks))
        self.documents[sha256] = doc
        self.added.append({"doc": doc, "chunks": list(chunks), "embeddings": list(embeddings)})
        return doc

    @staticmethod
    def _doc(filename: str, sha256: str, pages: int, chunk_count: int) -> Document:
        return Document(
            id=uuid4(),
            filename=filename,
            sha256=sha256,
            pages=pages,
            chunk_count=chunk_count,
            created_at=datetime.now(UTC),
        )

    async def list_documents(self) -> list[Document]:
        return list(self.documents.values())

    async def delete_document(self, document_id: UUID) -> bool:
        for key, doc in list(self.documents.items()):
            if doc.id == document_id:
                del self.documents[key]
                return True
        return False

    async def search_vector(self, embedding: Sequence[float], k: int) -> list[RetrievedChunk]:
        return self.vector_hits[:k]

    async def search_text(self, query: str, k: int) -> list[RetrievedChunk]:
        self.text_searches += 1
        return self.text_hits[:k]


class FakeHistory:
    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    async def add(self, message: ChatMessage) -> ChatMessage:
        saved = ChatMessage(
            id=len(self.messages) + 1,
            created_at=datetime.now(UTC),
            session_id=message.session_id,
            question=message.question,
            answer=message.answer,
            grounded=message.grounded,
            sources=message.sources,
            meta=message.meta,
        )
        self.messages.append(saved)
        return saved

    async def list(self, session_id: str, limit: int = 100) -> list[ChatMessage]:
        return [m for m in self.messages if m.session_id == session_id][-limit:]
