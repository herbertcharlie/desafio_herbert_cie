"""Errores de dominio. La capa API los traduce a códigos HTTP (ver app/api/errors.py)."""


class AppError(Exception):
    """Base de todos los errores controlados de la aplicación."""

    code = "app_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidDocumentError(AppError):
    code = "invalid_document"


class UploadTooLargeError(AppError):
    code = "upload_too_large"


class NoTextExtractedError(AppError):
    code = "no_text_extracted"


class DocumentNotFoundError(AppError):
    code = "document_not_found"


class DuplicateDocumentError(AppError):
    """El almacén detectó un documento con el mismo hash (condición de carrera)."""

    code = "duplicate_document"


class ConfigurationError(AppError):
    """Falta o es inválida una credencial/configuración necesaria."""

    code = "configuration_error"


class UpstreamUnavailableError(AppError):
    """El proveedor externo (LLM/embeddings) falló de forma temporal."""

    code = "upstream_unavailable"


class UpstreamTimeoutError(AppError):
    code = "upstream_timeout"


class LLMResponseError(AppError):
    """El LLM devolvió algo que no cumple el formato esperado."""

    code = "invalid_llm_response"
