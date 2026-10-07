"""Integración contra Postgres+pgvector real. Se omite si TEST_DATABASE_URL no está definida.

Ver el README (sección "Pruebas") para el comando que las ejecuta dentro de la red de Docker.
"""

from __future__ import annotations

import os
import uuid

import pytest

from app.domain.errors import DuplicateDocumentError
from app.domain.models import ChatMessage, ChunkDraft
from app.infra.pg_store import PgChatHistory, PgKnowledgeStore, create_engine

DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL no definida")

DIM = 1536


def vec(*head: float) -> list[float]:
    return [*head, *([0.0] * (DIM - len(head)))]


@pytest.fixture
async def engine():
    engine = create_engine(DATABASE_URL)
    yield engine
    await engine.dispose()


@pytest.fixture
async def stored_document(engine):
    store = PgKnowledgeStore(engine)
    sha = uuid.uuid4().hex + uuid.uuid4().hex  # 64 caracteres
    chunks = [
        ChunkDraft(0, 1, 1, "El Scrum Master es responsable de la efectividad del equipo."),
        ChunkDraft(1, 2, 3, "El Sprint Backlog contiene los elementos seleccionados."),
    ]
    doc = await store.add_document(
        filename="it.pdf",
        sha256=sha,
        pages=3,
        chunks=chunks,
        embeddings=[vec(1.0, 0.0), vec(0.0, 1.0)],
    )
    yield store, doc
    await store.delete_document(doc.id)


async def test_vector_search_orders_by_cosine_similarity(stored_document):
    store, doc = stored_document
    hits = await store.search_vector(vec(1.0, 0.1), k=5)
    mine = [h for h in hits if h.document_id == doc.id]
    assert mine[0].content.startswith("El Scrum Master")
    assert mine[0].similarity > 0.9
    assert (mine[0].page_start, mine[0].page_end) == (1, 1)
    assert mine[0].similarity > mine[1].similarity


async def test_text_search_uses_spanish_stemming_and_or_semantics(stored_document):
    store, doc = stored_document
    hits = await store.search_text("¿Qué elementos contiene el backlog del sprint?", k=5)
    assert any(h.document_id == doc.id and "Sprint Backlog" in h.content for h in hits)
    assert all(h.similarity is None for h in hits)


async def test_duplicate_hash_raises_domain_error(stored_document):
    store, doc = stored_document
    with pytest.raises(DuplicateDocumentError):
        await store.add_document(
            filename="otro.pdf",
            sha256=doc.sha256,
            pages=1,
            chunks=[ChunkDraft(0, 1, 1, "x")],
            embeddings=[vec(1.0)],
        )
    assert (await store.find_document_by_hash(doc.sha256)).id == doc.id


async def test_delete_cascades_to_chunks(engine):
    store = PgKnowledgeStore(engine)
    doc = await store.add_document(
        filename="tmp.pdf",
        sha256=uuid.uuid4().hex * 2,
        pages=1,
        chunks=[ChunkDraft(0, 1, 1, "fragmento temporal único zzqq")],
        embeddings=[vec(0.5, 0.5)],
    )
    assert await store.delete_document(doc.id) is True
    assert await store.delete_document(doc.id) is False
    assert await store.search_text("zzqq", k=3) == []


async def test_history_roundtrip_preserves_json_and_order(engine):
    history = PgChatHistory(engine)
    session = "it-" + uuid.uuid4().hex[:8]
    source = {"ref": 1, "filename": "it.pdf", "page_start": 2}
    await history.add(ChatMessage(session, "p1", "r1", True, [source], {"total_ms": 12.5}))
    await history.add(ChatMessage(session, "p2", "r2", False))

    messages = await history.list(session)

    assert [m.question for m in messages] == ["p1", "p2"]
    assert messages[0].sources == [source]
    assert messages[0].meta == {"total_ms": 12.5}
    assert messages[1].grounded is False
    assert messages[0].created_at is not None
