"""Esquemas Pydantic de entrada/salida de la API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.domain.models import AnswerResult, ChatMessage, Document, IngestResult

SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class DocumentOut(BaseModel):
    id: UUID
    filename: str
    pages: int
    chunk_count: int
    created_at: datetime

    @classmethod
    def from_domain(cls, doc: Document) -> DocumentOut:
        return cls(
            id=doc.id,
            filename=doc.filename,
            pages=doc.pages,
            chunk_count=doc.chunk_count,
            created_at=doc.created_at,
        )


class UploadResult(BaseModel):
    filename: str
    status: Literal["created", "duplicate", "failed"]
    document: DocumentOut | None = None
    error: ErrorBody | None = None

    @classmethod
    def from_ingest(cls, filename: str, result: IngestResult) -> UploadResult:
        return cls(
            filename=filename,
            status="duplicate" if result.duplicate else "created",
            document=DocumentOut.from_domain(result.document),
        )


class UploadResponse(BaseModel):
    results: list[UploadResult]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000, examples=["¿Qué es un Sprint?"])
    session_id: str = Field(pattern=SESSION_ID_PATTERN, examples=["demo-1"])

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("La pregunta no puede estar vacía.")
        return value


class SourceOut(BaseModel):
    ref: int = Field(description="Número [n] con el que se cita esta fuente en la respuesta.")
    chunk_id: UUID
    document_id: UUID
    filename: str
    page_start: int
    page_end: int
    similarity: float | None
    snippet: str


class MetaOut(BaseModel):
    model: str | None
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    prompt_tokens: int
    completion_tokens: int
    candidates: int
    best_similarity: float | None


class AskResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    grounded: bool = Field(
        description="False si los documentos no contienen información suficiente (rechazo)."
    )
    sources: list[SourceOut]
    meta: MetaOut

    @classmethod
    def from_domain(cls, session_id: str, question: str, result: AnswerResult) -> AskResponse:
        return cls(
            session_id=session_id,
            question=question,
            answer=result.answer,
            grounded=result.grounded,
            sources=[SourceOut(**vars(s)) for s in result.sources],
            meta=MetaOut(**vars(result.meta)),
        )


class HistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None
    question: str
    answer: str
    grounded: bool
    sources: list[dict[str, Any]]
    meta: dict[str, Any]
    created_at: datetime | None

    @classmethod
    def from_domain(cls, message: ChatMessage) -> HistoryItem:
        return cls.model_validate(message)


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryItem]
