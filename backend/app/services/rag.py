"""Caso de uso: responder una pregunta con RAG y grounding verificable."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.domain.errors import LLMResponseError
from app.domain.models import (
    AnswerMeta,
    AnswerResult,
    ChatMessage,
    LLMResponse,
    RetrievedChunk,
    SourceRef,
)
from app.domain.ports import ChatHistory, Embedder, KnowledgeStore, LLMProvider
from app.services.prompts import NO_INFO_MESSAGE, SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

RRF_K = 60  # constante estándar de Reciprocal Rank Fusion
SNIPPET_CHARS = 400


@dataclass(frozen=True)
class RAGConfig:
    candidates: int = 10
    context_top_k: int = 6
    min_similarity: float = 0.30
    hybrid: bool = True
    vector_anchor: int = 2  # mejores resultados vectoriales que siempre entran al contexto
    max_llm_concurrency: int = 4


@dataclass(frozen=True)
class _Parsed:
    sufficient: bool
    answer: str
    citations: list[int]


def reciprocal_rank_fusion(*rankings: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    """Fusiona varios rankings sumando 1/(k+rango). Conserva la similitud vectorial si existe."""
    scores: dict[Any, float] = {}
    best: dict[Any, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (RRF_K + rank)
            current = best.get(chunk.id)
            if current is None or (current.similarity is None and chunk.similarity is not None):
                best[chunk.id] = chunk
    return [best[cid] for cid in sorted(scores, key=scores.__getitem__, reverse=True)]


def select_context(
    vector_hits: Sequence[RetrievedChunk],
    lexical_hits: Sequence[RetrievedChunk],
    top_k: int,
    anchor: int,
) -> list[RetrievedChunk]:
    """Elige los fragmentos que verá el LLM: híbrido anclado en el retriever vectorial.

    Los `anchor` mejores resultados vectoriales entran siempre; el resto de plazas se rellena
    con el orden RRF. Sin el ancla, una búsqueda léxica poco selectiva (muchas palabras
    genéricas) da doble voto a fragmentos mediocres y desplaza al mejor resultado vectorial.
    """
    anchored = list(vector_hits[: min(anchor, top_k)])
    taken = {chunk.id for chunk in anchored}
    rest = [c for c in reciprocal_rank_fusion(vector_hits, lexical_hits) if c.id not in taken]
    return (anchored + rest)[:top_k]


def parse_llm_answer(raw: str) -> _Parsed:
    try:
        data = json.loads(raw)
        sufficient = bool(data["sufficient"])
        answer = str(data.get("answer", "")).strip()
        citations = [int(c) for c in data.get("citations", [])]
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise LLMResponseError("El modelo devolvió una respuesta con formato inválido.") from exc
    return _Parsed(sufficient=sufficient, answer=answer, citations=citations)


class RAGService:
    def __init__(
        self,
        *,
        embedder: Embedder,
        store: KnowledgeStore,
        llm: LLMProvider,
        history: ChatHistory,
        config: RAGConfig,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._history = history
        self._config = config
        # Limita las llamadas simultáneas al LLM (control de concurrencia y de costos).
        self._llm_slots = asyncio.Semaphore(config.max_llm_concurrency)

    async def ask(self, question: str, session_id: str) -> AnswerResult:
        started = time.perf_counter()
        result = await self._answer(question, started)
        await self._history.add(self._to_history(session_id, question, result))
        return result

    async def _answer(self, question: str, started: float) -> AnswerResult:
        cfg = self._config

        # 1) Recuperación
        query_vec = await self._embedder.embed_query(question)
        vector_hits = await self._store.search_vector(query_vec, cfg.candidates)
        lexical_hits: list[RetrievedChunk] = []
        if cfg.hybrid and vector_hits:
            lexical_hits = await self._store.search_text(question, cfg.candidates)
        retrieval_ms = (time.perf_counter() - started) * 1000

        best_similarity = max((h.similarity or 0.0 for h in vector_hits), default=None)

        # 2) Puerta de relevancia: sin evidencia suficiente NO se llama al LLM.
        if best_similarity is None or best_similarity < cfg.min_similarity:
            logger.info("Rechazo por baja similitud (best=%s)", best_similarity)
            return self._refusal(started, retrieval_ms, len(vector_hits), best_similarity)

        context = select_context(vector_hits, lexical_hits, cfg.context_top_k, cfg.vector_anchor)

        # 3) Generación (acotada por semáforo)
        generation_started = time.perf_counter()
        async with self._llm_slots:
            response = await self._llm.complete(
                system=SYSTEM_PROMPT,
                user=build_user_prompt(question, context),
                json_mode=True,
            )
        generation_ms = (time.perf_counter() - generation_started) * 1000
        parsed = parse_llm_answer(response.text)

        # 4) Verificación de grounding: las citas deben existir en el contexto enviado.
        valid = sorted({c for c in parsed.citations if 1 <= c <= len(context)})
        if not parsed.sufficient or not parsed.answer or not valid:
            logger.info(
                "Rechazo por grounding (sufficient=%s, citas válidas=%s)", parsed.sufficient, valid
            )
            return self._refusal(
                started,
                retrieval_ms,
                len(vector_hits),
                best_similarity,
                response=response,
                generation_ms=generation_ms,
            )

        sources = [self._to_source(ref, context[ref - 1]) for ref in valid]
        return AnswerResult(
            answer=parsed.answer,
            grounded=True,
            sources=sources,
            meta=AnswerMeta(
                model=response.model,
                retrieval_ms=round(retrieval_ms, 1),
                generation_ms=round(generation_ms, 1),
                total_ms=round((time.perf_counter() - started) * 1000, 1),
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                candidates=len(vector_hits),
                best_similarity=best_similarity,
            ),
        )

    @staticmethod
    def _refusal(
        started: float,
        retrieval_ms: float,
        candidates: int,
        best_similarity: float | None,
        *,
        response: LLMResponse | None = None,
        generation_ms: float = 0.0,
    ) -> AnswerResult:
        return AnswerResult(
            answer=NO_INFO_MESSAGE,
            grounded=False,
            sources=[],
            meta=AnswerMeta(
                model=response.model if response else None,
                retrieval_ms=round(retrieval_ms, 1),
                generation_ms=round(generation_ms, 1),
                total_ms=round((time.perf_counter() - started) * 1000, 1),
                prompt_tokens=response.prompt_tokens if response else 0,
                completion_tokens=response.completion_tokens if response else 0,
                candidates=candidates,
                best_similarity=best_similarity,
            ),
        )

    @staticmethod
    def _to_source(ref: int, chunk: RetrievedChunk) -> SourceRef:
        return SourceRef(
            ref=ref,
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            filename=chunk.filename,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            similarity=chunk.similarity,
            snippet=chunk.content[:SNIPPET_CHARS],
        )

    @staticmethod
    def _to_history(session_id: str, question: str, result: AnswerResult) -> ChatMessage:
        return ChatMessage(
            session_id=session_id,
            question=question,
            answer=result.answer,
            grounded=result.grounded,
            sources=[
                {
                    "ref": s.ref,
                    "chunk_id": str(s.chunk_id),
                    "document_id": str(s.document_id),
                    "filename": s.filename,
                    "page_start": s.page_start,
                    "page_end": s.page_end,
                    "similarity": s.similarity,
                    "snippet": s.snippet,
                }
                for s in result.sources
            ],
            meta={
                "model": result.meta.model,
                "total_ms": result.meta.total_ms,
                "retrieval_ms": result.meta.retrieval_ms,
                "generation_ms": result.meta.generation_ms,
                "prompt_tokens": result.meta.prompt_tokens,
                "completion_tokens": result.meta.completion_tokens,
            },
        )
