from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.container import Services
from app.core.config import Settings
from app.main import create_app
from app.services.ingestion import IngestionService
from app.services.rag import RAGConfig, RAGService
from tests.fakes import FakeEmbedder, FakeHistory, FakeLLM, FakeStore


def make_pdf(pages: list[str]) -> bytes:
    """Genera un PDF mínimo válido con una línea de texto por página (solo para pruebas)."""
    objects: list[bytes] = []
    n = len(pages)
    page_ids = [3 + 2 * i for i in range(n)]
    font_id = 3 + 2 * n
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    for i, text in enumerate(pages):
        content_id = page_ids[i] + 1
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode()
        )
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 50 700 Td ({escaped}) Tj ET".encode("cp1252")
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


@pytest.fixture
def settings() -> Settings:
    return Settings(chunk_size=200, chunk_overlap=40, max_upload_mb=1)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def history() -> FakeHistory:
    return FakeHistory()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def services(settings, embedder, store, history, llm) -> Services:
    from app.infra.pdf import extract_pages

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
            config=RAGConfig(min_similarity=0.3, context_top_k=3),
        ),
    )


@pytest.fixture
def client(services, settings):
    with TestClient(create_app(services=services, settings=settings)) as test_client:
        yield test_client
