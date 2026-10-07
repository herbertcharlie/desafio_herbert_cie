"""Endpoints HTTP. Solo adaptan peticiones/respuestas; la lógica vive en `app/services`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File
from fastapi import Path as PathParam
from fastapi import Request, UploadFile, status

from app.api.schemas import (
    SESSION_ID_PATTERN,
    AskRequest,
    AskResponse,
    DocumentOut,
    ErrorBody,
    ErrorResponse,
    HistoryItem,
    HistoryResponse,
    UploadResponse,
    UploadResult,
)
from app.container import Services
from app.domain.errors import (
    AppError,
    ConfigurationError,
    DocumentNotFoundError,
    InvalidDocumentError,
    UploadTooLargeError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
)

router = APIRouter()

PDF_MAGIC = b"%PDF-"
# Si falla el proveedor externo, no tiene sentido seguir con el resto de archivos.
_ABORT_BATCH = (ConfigurationError, UpstreamUnavailableError, UpstreamTimeoutError)

ERROR_RESPONSES: dict[int | str, dict] = {
    422: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
    504: {"model": ErrorResponse},
}


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


@router.get("/health", tags=["system"])
async def health(services: ServicesDep) -> dict[str, str]:
    await services.store.ping()
    return {"status": "ok"}


async def _read_pdf(file: UploadFile, max_bytes: int) -> tuple[str, bytes]:
    filename = Path(file.filename or "documento.pdf").name
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise UploadTooLargeError(f"'{filename}' supera el límite de {max_bytes // 2**20} MB.")
    if not data.startswith(PDF_MAGIC):
        raise InvalidDocumentError(f"'{filename}' no es un archivo PDF.")
    return filename, data


@router.post(
    "/documents",
    response_model=UploadResponse,
    tags=["documents"],
    summary="Carga y procesa uno o más PDF",
    responses={413: {"model": ErrorResponse}, **ERROR_RESPONSES},
)
async def upload_documents(
    services: ServicesDep,
    files: Annotated[list[UploadFile], File(description="Uno o más archivos PDF")],
) -> UploadResponse:
    """Extrae el texto, lo divide en chunks, genera embeddings y los indexa.

    Devuelve un resultado por archivo (`created`, `duplicate` o `failed`). Si **todos** los
    archivos fallan, responde con el error HTTP del primero.
    """
    results: list[UploadResult] = []
    first_error: AppError | None = None

    for file in files:
        filename = Path(file.filename or "documento.pdf").name
        try:
            filename, data = await _read_pdf(file, services.settings.max_upload_bytes)
            outcome = await services.ingestion.ingest(filename, data)
            results.append(UploadResult.from_ingest(filename, outcome))
        except _ABORT_BATCH:
            raise
        except AppError as exc:
            first_error = first_error or exc
            results.append(
                UploadResult(
                    filename=filename,
                    status="failed",
                    error=ErrorBody(code=exc.code, message=exc.message),
                )
            )

    if first_error and all(r.status == "failed" for r in results):
        raise first_error
    return UploadResponse(results=results)


@router.get("/documents", response_model=list[DocumentOut], tags=["documents"])
async def list_documents(services: ServicesDep) -> list[DocumentOut]:
    return [DocumentOut.from_domain(d) for d in await services.store.list_documents()]


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["documents"],
    responses={404: {"model": ErrorResponse}},
)
async def delete_document(document_id: UUID, services: ServicesDep) -> None:
    if not await services.store.delete_document(document_id):
        raise DocumentNotFoundError("El documento no existe.")


@router.post(
    "/ask",
    response_model=AskResponse,
    tags=["rag"],
    summary="Pregunta sobre los documentos cargados",
    responses=ERROR_RESPONSES,
)
async def ask(body: AskRequest, services: ServicesDep) -> AskResponse:
    """Recupera fragmentos relevantes y genera una respuesta fundamentada con fuentes.

    Si los documentos no contienen información suficiente, `grounded` es `false`,
    `sources` está vacío y la respuesta lo indica explícitamente.
    """
    result = await services.rag.ask(body.question, body.session_id)
    return AskResponse.from_domain(body.session_id, body.question, result)


@router.get("/history/{session_id}", response_model=HistoryResponse, tags=["history"])
async def get_history(
    session_id: Annotated[str, PathParam(pattern=SESSION_ID_PATTERN)],
    services: ServicesDep,
) -> HistoryResponse:
    messages = await services.history.list(session_id)
    return HistoryResponse(
        session_id=session_id, messages=[HistoryItem.from_domain(m) for m in messages]
    )
