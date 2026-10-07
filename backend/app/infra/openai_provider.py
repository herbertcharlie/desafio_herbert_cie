"""Implementaciones de `Embedder` y `LLMProvider` sobre la API de OpenAI.

Es el ÚNICO módulo que conoce el SDK de OpenAI. Traduce sus excepciones a errores de dominio.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import openai
from openai import AsyncOpenAI

from app.core.config import Settings
from app.domain.errors import (
    ConfigurationError,
    LLMResponseError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
)
from app.domain.models import LLMResponse


@contextmanager
def _translate_errors() -> Iterator[None]:
    try:
        yield
    except openai.APITimeoutError as exc:
        raise UpstreamTimeoutError("El proveedor de IA tardó demasiado en responder.") from exc
    except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
        raise ConfigurationError(
            "La credencial de OpenAI es inválida o no tiene permisos."
        ) from exc
    except openai.OpenAIError as exc:  # rate limit, conexión, 5xx, 4xx restantes
        raise UpstreamUnavailableError("El proveedor de IA no está disponible ahora.") from exc


def build_client(settings: Settings) -> AsyncOpenAI | None:
    if settings.openai_api_key is None:
        return None
    return AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,  # reintentos con backoff ante 429/5xx/timeouts
    )


def _require(client: AsyncOpenAI | None) -> AsyncOpenAI:
    if client is None:
        raise ConfigurationError("OPENAI_API_KEY no está configurada.")
    return client


class OpenAIEmbedder:
    def __init__(
        self, client: AsyncOpenAI | None, *, model: str, dimensions: int, batch_size: int = 96
    ) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        client = _require(self._client)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            with _translate_errors():
                response = await client.embeddings.create(**self._params(batch))
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda i: i.index))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    def _params(self, batch: list[str]) -> dict:
        params: dict = {"model": self._model, "input": batch}
        if self._model.startswith("text-embedding-3"):
            params["dimensions"] = self._dimensions
        return params


class OpenAILLM:
    def __init__(self, client: AsyncOpenAI | None, *, model: str, temperature: float) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    async def complete(self, *, system: str, user: str, json_mode: bool = False) -> LLMResponse:
        client = _require(self._client)
        kwargs: dict = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        with _translate_errors():
            response = await client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **kwargs,
            )
        content = response.choices[0].message.content
        if not content:
            raise LLMResponseError("El modelo devolvió una respuesta vacía.")
        usage = response.usage
        return LLMResponse(
            text=content,
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
