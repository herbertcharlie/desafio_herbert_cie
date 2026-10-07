"""Modelos de dominio: estructuras puras, sin dependencias de infraestructura."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass(frozen=True)
class PageText:
    """Texto extraído de una página del PDF (numeración desde 1)."""

    number: int
    text: str


@dataclass(frozen=True)
class ChunkDraft:
    """Fragmento aún sin embedding ni identificador."""

    index: int
    page_start: int
    page_end: int
    content: str


@dataclass(frozen=True)
class Document:
    id: UUID
    filename: str
    sha256: str
    pages: int
    chunk_count: int
    created_at: datetime


@dataclass(frozen=True)
class RetrievedChunk:
    id: UUID
    document_id: UUID
    filename: str
    page_start: int
    page_end: int
    content: str
    # Similitud coseno (0..1 aprox.). None si el fragmento solo apareció en la búsqueda léxica.
    similarity: float | None = None


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class SourceRef:
    """Fuente citada en una respuesta. `ref` es el número [n] que aparece en el texto."""

    ref: int
    chunk_id: UUID
    document_id: UUID
    filename: str
    page_start: int
    page_end: int
    similarity: float | None
    snippet: str


@dataclass(frozen=True)
class AnswerMeta:
    model: str | None
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    candidates: int = 0
    best_similarity: float | None = None


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    grounded: bool
    sources: list[SourceRef]
    meta: AnswerMeta


@dataclass(frozen=True)
class IngestResult:
    document: Document
    duplicate: bool


@dataclass(frozen=True)
class ChatMessage:
    session_id: str
    question: str
    answer: str
    grounded: bool
    sources: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    id: int | None = None
    created_at: datetime | None = None
