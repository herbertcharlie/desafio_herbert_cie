"""Chunking consciente de páginas.

Estrategia: se normaliza el texto de cada página, se parte en oraciones y se agrupan
oraciones consecutivas hasta `size` caracteres. Entre chunks contiguos se arrastran las
últimas oraciones (hasta `overlap` caracteres) para no perder contexto en los bordes.
Cada chunk recuerda la página donde empieza y donde termina, base de la trazabilidad.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from app.domain.models import ChunkDraft, PageText

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_WHITESPACE = re.compile(r"\s+")

Unit = tuple[int, str]  # (número de página, texto de la oración)


def normalize_text(text: str) -> str:
    """Une palabras cortadas por guion de fin de línea y colapsa espacios/saltos.

    NFKC deshace ligaduras tipográficas (ﬁ, ﬂ...) que algunos PDF usan y que romperían la
    búsqueda léxica ("deﬁnición" no coincidiría con "definición").
    """
    text = unicodedata.normalize("NFKC", text).replace("­", "")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    return _WHITESPACE.sub(" ", text).strip()


def _split_long(sentence: str, size: int) -> list[str]:
    return [sentence[i : i + size] for i in range(0, len(sentence), size)]


def _units(pages: Sequence[PageText], size: int) -> list[Unit]:
    units: list[Unit] = []
    for page in pages:
        text = normalize_text(page.text)
        if not text:
            continue
        for sentence in _SENTENCE_SPLIT.split(text):
            if sentence:
                units.extend((page.number, piece) for piece in _split_long(sentence, size))
    return units


def _carry_over(units: list[Unit], overlap: int) -> tuple[list[Unit], int]:
    carried: list[Unit] = []
    length = 0
    for unit in reversed(units):
        cost = len(unit[1]) + 1
        if length + cost > overlap:
            break
        carried.insert(0, unit)
        length += cost
    return carried, length


def _make_chunk(index: int, units: list[Unit]) -> ChunkDraft:
    return ChunkDraft(
        index=index,
        page_start=units[0][0],
        page_end=units[-1][0],
        content=" ".join(text for _, text in units),
    )


def chunk_pages(pages: Sequence[PageText], size: int, overlap: int) -> list[ChunkDraft]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("size debe ser > 0 y 0 <= overlap < size")

    chunks: list[ChunkDraft] = []
    current: list[Unit] = []
    current_len = 0
    fresh = 0  # unidades nuevas desde el último chunk emitido (evita un chunk solo de solape)

    for unit in _units(pages, size):
        cost = len(unit[1]) + 1
        if current and current_len + cost > size:
            chunks.append(_make_chunk(len(chunks), current))
            current, current_len = _carry_over(current, overlap)
            if current_len + cost > size:  # el solape no cabe junto a esta oración
                current, current_len = [], 0
            fresh = 0
        current.append(unit)
        current_len += cost
        fresh += 1

    if current and fresh:
        chunks.append(_make_chunk(len(chunks), current))
    return chunks
