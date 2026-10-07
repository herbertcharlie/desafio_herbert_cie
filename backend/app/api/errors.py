"""Manejo de errores consistente: toda respuesta de error usa {"error": {code, message}}."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.domain.errors import (
    AppError,
    ConfigurationError,
    DocumentNotFoundError,
    InvalidDocumentError,
    LLMResponseError,
    NoTextExtractedError,
    UploadTooLargeError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
)

logger = logging.getLogger(__name__)

STATUS_BY_ERROR: dict[type[AppError], int] = {
    InvalidDocumentError: 422,
    NoTextExtractedError: 422,
    UploadTooLargeError: 413,
    DocumentNotFoundError: 404,
    ConfigurationError: 503,
    UpstreamUnavailableError: 503,
    UpstreamTimeoutError: 504,
    LLMResponseError: 502,
}


def status_for(error: AppError) -> int:
    for error_type, status in STATUS_BY_ERROR.items():
        if isinstance(error, error_type):
            return status
    return 500


def error_payload(code: str, message: str, details: object | None = None) -> dict:
    body: dict = {"code": code, "message": message}
    if details is not None:
        body["details"] = details
    return {"error": body}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(error_payload(exc.code, exc.message), status_for(exc))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]}
            for e in exc.errors()
        ]
        return JSONResponse(
            error_payload("validation_error", "La solicitud no es válida.", details), 422
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(error_payload("http_error", str(exc.detail)), exc.status_code)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Error no controlado", exc_info=exc)
        return JSONResponse(error_payload("internal_error", "Error interno del servidor."), 500)
