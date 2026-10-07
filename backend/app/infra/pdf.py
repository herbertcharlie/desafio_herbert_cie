"""Extracción de texto por página con pypdf."""

from __future__ import annotations

import io

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app.domain.errors import InvalidDocumentError
from app.domain.models import PageText


def extract_pages(data: bytes) -> list[PageText]:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise InvalidDocumentError("El PDF está protegido con contraseña.")
        return [
            PageText(number=index, text=page.extract_text() or "")
            for index, page in enumerate(reader.pages, start=1)
        ]
    except InvalidDocumentError:
        raise
    except (PyPdfError, ValueError, KeyError, OSError) as exc:
        raise InvalidDocumentError("No se pudo leer el archivo como PDF válido.") from exc
