"""Composition root: único lugar donde se eligen e instancian las implementaciones concretas."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from app.core.config import Settings
from app.domain.ports import ChatHistory, KnowledgeStore
from app.infra.openai_provider import OpenAIEmbedder, OpenAILLM, build_client
from app.infra.pdf import extract_pages
from app.infra.pg_store import PgChatHistory, PgKnowledgeStore, create_engine
from app.services.ingestion import IngestionService
from app.services.rag import RAGConfig, RAGService


async def _noop() -> None:
    return None


@dataclass
class Services:
    settings: Settings
    store: KnowledgeStore
    history: ChatHistory
    ingestion: IngestionService
    rag: RAGService
    aclose: Callable[[], Awaitable[None]] = field(default=_noop)


def build_services(settings: Settings) -> Services:
    engine = create_engine(settings.database_url)
    store = PgKnowledgeStore(engine)
    history = PgChatHistory(engine)

    client = build_client(settings)
    embedder = OpenAIEmbedder(
        client, model=settings.openai_embedding_model, dimensions=settings.embedding_dim
    )
    llm = OpenAILLM(client, model=settings.openai_chat_model, temperature=settings.llm_temperature)

    return Services(
        settings=settings,
        store=store,
        history=history,
        ingestion=IngestionService(
            extract_pages=extract_pages,
            embedder=embedder,
            store=store,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        rag=RAGService(
            embedder=embedder,
            store=store,
            llm=llm,
            history=history,
            config=RAGConfig(
                candidates=settings.retrieval_candidates,
                context_top_k=settings.context_top_k,
                min_similarity=settings.min_similarity,
                hybrid=settings.hybrid_search,
                max_llm_concurrency=settings.llm_max_concurrency,
            ),
        ),
        aclose=engine.dispose,
    )
