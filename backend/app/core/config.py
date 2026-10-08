"""Configuración centralizada. Todo se lee de variables de entorno (nada de secretos en código)."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # Infraestructura
    database_url: str = "postgresql+asyncpg://rag:rag@db:5432/rag"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # Proveedor de IA (OpenAI)
    openai_api_key: SecretStr | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536  # Debe coincidir con vector(N) en db/init/001_schema.sql
    llm_temperature: float = 0.0
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2
    llm_max_concurrency: int = 4  # llamadas simultáneas al LLM por proceso

    # Ingesta
    chunk_size: int = 1000  # caracteres (~250 tokens en español)
    chunk_overlap: int = 150
    max_upload_mb: int = 25

    # Recuperación y grounding
    retrieval_candidates: int = 10  # candidatos por cada método (vectorial y léxico)
    context_top_k: int = 8  # fragmentos que se envían al LLM
    min_similarity: float = 0.30  # por debajo, no se llama al LLM
    hybrid_search: bool = True
    vector_anchor: int = Field(default=2, ge=0)  # top vectoriales que siempre van al contexto

    # Entrada
    max_question_chars: int = 1000

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def _blank_key_is_none(cls, value: object) -> object:
        # docker compose pasa OPENAI_API_KEY="" cuando no está definida.
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _check_chunking(self) -> Settings:
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP debe ser >= 0 y menor que CHUNK_SIZE")
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
