"""Traducción de errores del SDK de OpenAI a errores de dominio (sin red)."""

import httpx
import openai
import pytest

from app.domain.errors import (
    ConfigurationError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
)
from app.infra.openai_provider import _translate_errors

REQUEST = httpx.Request("POST", "https://api.openai.com/v1/embeddings")


def status_error(cls, status: int, body: dict | None = None):
    response = httpx.Response(status, request=REQUEST)
    return cls("error", response=response, body=body)


@pytest.mark.parametrize(
    ("exc", "expected", "text"),
    [
        (openai.APITimeoutError(request=REQUEST), UpstreamTimeoutError, "tardó"),
        (status_error(openai.AuthenticationError, 401), ConfigurationError, "inválida"),
        (
            status_error(openai.PermissionDeniedError, 403, {"message": "sin acceso al modelo X"}),
            ConfigurationError,
            "sin acceso al modelo X",
        ),
        (
            status_error(openai.RateLimitError, 429, {"code": "insufficient_quota"}),
            ConfigurationError,
            "crédito",
        ),
        (
            status_error(openai.RateLimitError, 429, {"code": "rate_limit"}),
            UpstreamUnavailableError,
            "tasa",
        ),
        (openai.APIConnectionError(request=REQUEST), UpstreamUnavailableError, "disponible"),
        (status_error(openai.InternalServerError, 500), UpstreamUnavailableError, "disponible"),
    ],
)
def test_sdk_errors_are_translated(exc, expected, text):
    with pytest.raises(expected) as caught, _translate_errors():
        raise exc
    assert text in caught.value.message
