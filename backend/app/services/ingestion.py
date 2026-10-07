"""Caso de uso: ingesta de un PDF (extraer -> trocear -> embeddings -> almacenar)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Callable, Sequence

from app.domain.errors import DuplicateDocumentError, NoTextExtractedError
from app.domain.models import IngestResult, PageText
from app.domain.ports import Embedder, KnowledgeStore
from app.services.chunking import chunk_pages

logger = logging.getLogger(__name__)

PdfExtractor = Callable[[bytes], Sequence[PageText]]


class IngestionService:
    def __init__(
        self,
        *,
        extract_pages: PdfExtractor,
        embedder: Embedder,
        store: KnowledgeStore,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._extract_pages = extract_pages
        self._embedder = embedder
        self._store = store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    async def ingest(self, filename: str, data: bytes) -> IngestResult:
        sha256 = hashlib.sha256(data).hexdigest()

        existing = await self._store.find_document_by_hash(sha256)
        if existing:
            return IngestResult(document=existing, duplicate=True)

        # La extracción es CPU-bound: fuera del event loop para no bloquear otras peticiones.
        pages = await asyncio.to_thread(self._extract_pages, data)
        chunks = chunk_pages(pages, self._chunk_size, self._chunk_overlap)
        if not chunks:
            raise NoTextExtractedError(
                "El PDF no contiene texto extraíble (¿es un documento escaneado?)."
            )

        embeddings = await self._embedder.embed_documents([c.content for c in chunks])

        try:
            document = await self._store.add_document(
                filename=filename,
                sha256=sha256,
                pages=len(pages),
                chunks=chunks,
                embeddings=embeddings,
            )
        except DuplicateDocumentError:  # dos cargas simultáneas del mismo archivo
            existing = await self._store.find_document_by_hash(sha256)
            if existing is None:
                raise
            return IngestResult(document=existing, duplicate=True)

        logger.info(
            "Documento ingestado: %s (%d páginas, %d chunks)", filename, len(pages), len(chunks)
        )
        return IngestResult(document=document, duplicate=False)
