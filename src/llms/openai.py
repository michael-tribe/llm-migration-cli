from string import Template

import backoff
import tiktoken
from httpx import RequestError
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError

from src.llms.base import BaseLLM, llm_registry
from src.llms.enums import FinishReason
from src.logger import root_logger
from src.settings import settings


backoff_exceptions = (RequestError, APIConnectionError, APITimeoutError, RateLimitError)
log = root_logger.getChild(__name__)


class OpenAiLLM(BaseLLM):
    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.llm.openai_api_key)
        self._tokenizer = tiktoken.encoding_for_model(self.llm_name)

    @backoff.on_exception(backoff.expo, backoff_exceptions, base=2, factor=2, max_value=120, max_tries=50)
    async def _generate(
        self,
        template: Template,
        template_params: dict[str, str],
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        **extra_options: dict,
    ) -> tuple[str | None, FinishReason]:
        messages = self._build_messages(template=template, template_params=template_params)
        response = await self._client.chat.completions.create(
            messages=messages,
            model=self.llm_name,
            temperature=temperature,
            max_tokens=max_output_tokens,
            user="llm-migration-evaluation",
            **extra_options,
        )
        finish_reason = self._convert_to_finish_reason(reason=response.choices[0].finish_reason)
        return response.choices[0].message.content, finish_reason

    def _get_num_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def _trim_to_max_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self._tokenizer.encode(text)
        return self._tokenizer.decode(tokens[:max_tokens])

    def _convert_to_finish_reason(self, reason: str | None) -> FinishReason:
        match reason:
            case "stop" | "tool_calls" | "content_filter" | "function_call":
                return FinishReason.STOP_SEQUENCE
            case "length":
                return FinishReason.MAX_LENGTH
            case _:
                return FinishReason.SUCCESS


@llm_registry.register("gpt-3.5-turbo")
class Gpt3Turbo(OpenAiLLM):
    llm_name = "gpt-3.5-turbo-1106"
    context_window_size = 16_385
    max_output_tokens = 4096
    input_token_cost_usd: float = 1 / 1_000_000
    output_token_cost_usd: float = 2 / 1_000_000


@llm_registry.register("gpt-4")
class Gpt4(OpenAiLLM):
    llm_name = "gpt-4"
    context_window_size = 8192
    max_output_tokens = 4096
    input_token_cost_usd: float = 30 / 1_000_000
    output_token_cost_usd: float = 60 / 1_000_000


@llm_registry.register("gpt-4-turbo")
class Gpt4Turbo(OpenAiLLM):
    llm_name = "gpt-4-turbo"
    context_window_size = 128_000
    max_output_tokens = 4096
    input_token_cost_usd: float = 10 / 1_000_000
    output_token_cost_usd: float = 30 / 1_000_000
