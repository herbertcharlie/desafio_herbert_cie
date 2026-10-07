import pytest

from app.domain.errors import InvalidDocumentError, NoTextExtractedError
from app.infra.pdf import extract_pages
from tests.conftest import make_pdf


def test_extract_pages_reads_text_per_page_with_accents():
    pages = extract_pages(make_pdf(["Hola Scrum", "El equipo es autoorganizado"]))
    assert [p.number for p in pages] == [1, 2]
    assert "Hola Scrum" in pages[0].text
    assert "autoorganizado" in pages[1].text


def test_extract_pages_rejects_garbage():
    with pytest.raises(InvalidDocumentError):
        extract_pages(b"%PDF-1.4 esto no es un pdf real")


async def test_ingest_extracts_chunks_embeds_and_stores(services, store, embedder):
    pdf = make_pdf(["El Scrum Master ayuda al equipo a entender la teoría de Scrum."])

    result = await services.ingestion.ingest("scrum.pdf", pdf)

    assert result.duplicate is False
    assert result.document.filename == "scrum.pdf"
    assert result.document.pages == 1
    saved = store.added[0]
    assert len(saved["chunks"]) == len(saved["embeddings"]) == result.document.chunk_count
    assert embedder.document_calls[0] == [c.content for c in saved["chunks"]]


async def test_same_file_twice_is_detected_as_duplicate_without_reembedding(services, embedder):
    pdf = make_pdf(["Contenido repetido para comprobar la deduplicación."])
    first = await services.ingestion.ingest("a.pdf", pdf)
    second = await services.ingestion.ingest("copia.pdf", pdf)

    assert second.duplicate is True
    assert second.document.id == first.document.id
    assert len(embedder.document_calls) == 1


async def test_concurrent_duplicate_race_returns_existing_document(services, store):
    store.raise_duplicate_on_add = True
    result = await services.ingestion.ingest("a.pdf", make_pdf(["Texto de prueba para carrera."]))
    assert result.duplicate is True


async def test_pdf_without_text_raises(services):
    with pytest.raises(NoTextExtractedError):
        await services.ingestion.ingest("vacio.pdf", make_pdf(["   "]))
