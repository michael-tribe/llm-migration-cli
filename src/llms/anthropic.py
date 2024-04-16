from string import Template

import backoff
from anthropic import (
    Anthropic,
    APIConnectionError,
    APIStatusError,
    AsyncAnthropic,
    RateLimitError,
)
from httpx import RequestError

from src.llms.base import BaseLLM, llm_registry
from src.llms.enums import FinishReason
from src.logger import root_logger
from src.settings import settings


backoff_exceptions = (RequestError, APIConnectionError, APIStatusError, RateLimitError)
log = root_logger.getChild(__name__)


class AnthropicLLM(BaseLLM):
    def __init__(self) -> None:
        self._async_client = AsyncAnthropic(api_key=settings.llm.anthropic_api_key)
        self._sync_client = Anthropic(api_key=settings.llm.anthropic_api_key)

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
        message = await self._async_client.messages.create(
            messages=messages,
            model=self.llm_name,
            max_tokens=max_output_tokens,
            temperature=temperature,
            **extra_options,
        )
        finish_reason = self._convert_to_finish_reason(reason=message.stop_reason)
        return message.content[0].text, finish_reason

    def _get_num_tokens(self, text: str) -> int:
        return self._sync_client.count_tokens(text=text)

    def _trim_to_max_tokens(self, text: str, max_tokens: int) -> str:
        tokenizer = self._sync_client.get_tokenizer()
        encoded_text = tokenizer.encode(text)
        return tokenizer.decode(encoded_text.ids[:max_tokens]).strip()

    def _convert_to_finish_reason(self, reason: str) -> FinishReason:
        match reason:
            case "max_tokens":
                return FinishReason.MAX_LENGTH
            case "stop_sequence":
                return FinishReason.STOP_SEQUENCE
            case _:
                return FinishReason.SUCCESS


class Claude3(AnthropicLLM):
    context_window_size = 200_000
    max_output_tokens = 4096


@llm_registry.register("claude-opus")
class Claude3Opus(Claude3):
    llm_name = "claude-3-opus-20240229"
    input_token_cost_usd: float = 15 / 1_000_000
    output_token_cost_usd: float = 75 / 1_000_000


@llm_registry.register("claude-sonnet")
class Claude3Sonnet(Claude3):
    llm_name = "claude-3-sonnet-20240229"
    input_token_cost_usd: float = 3 / 1_000_000
    output_token_cost_usd: float = 15 / 1_000_000


@llm_registry.register("claude-haiku")
class Claude3Haiku(Claude3):
    llm_name = "claude-3-haiku-20240307"
    input_token_cost_usd: float = 0.25 / 1_000_000
    output_token_cost_usd: float = 1.25 / 1_000_000
