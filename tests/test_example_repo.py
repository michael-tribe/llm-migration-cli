import pytest

from src.settings import settings

from .example_repo.main import generate_examples


LLM_API_KEYS_SET = settings.llm.anthropic_api_key and settings.llm.openai_api_key


@pytest.mark.skipif(not LLM_API_KEYS_SET, reason="LLM API keys not set")
def test_generate_examples() -> None:
    generate_examples()
