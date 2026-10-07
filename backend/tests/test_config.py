import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.domain.errors import ConfigurationError
from app.infra.openai_provider import OpenAILLM, build_client


def test_blank_api_key_is_treated_as_missing():
    settings = Settings(openai_api_key="  ")
    assert settings.openai_api_key is None
    assert build_client(settings) is None


async def test_missing_key_fails_only_when_the_llm_is_used():
    llm = OpenAILLM(None, model="m", temperature=0)
    with pytest.raises(ConfigurationError):
        await llm.complete(system="s", user="u")


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError):
        Settings(chunk_size=100, chunk_overlap=100)


def test_secret_is_not_leaked_in_repr():
    assert "sk-secret" not in repr(Settings(openai_api_key="sk-secret"))
