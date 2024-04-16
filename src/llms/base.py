from abc import ABC, abstractmethod
from functools import lru_cache
from string import Template
from typing import TypeVar

from class_registry import ClassRegistry
from pydantic import BaseModel, ValidationError

from src.example_logger import ExampleLogger
from src.llms.enums import FinishReason
from src.logger import root_logger
from src.settings import settings


log = root_logger.getChild(__name__)
llm_registry = ClassRegistry()
T = TypeVar("T", bound=BaseModel)


class BaseLLM(ABC):
    llm_name: str
    context_window_size: int
    input_token_cost_usd: float
    output_token_cost_usd: float
    example_logger: ExampleLogger | None = None

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __init__(self, log_examples: bool = False) -> None:
        self.log_examples = log_examples
        if self.log_examples:
            self.example_logger = ExampleLogger()

    @abstractmethod
    async def _generate(
        self,
        template: Template,
        template_params: dict[str, str],
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        **extra_options: dict,
    ) -> tuple[str | None, FinishReason]:
        pass

    @abstractmethod
    def _get_num_tokens(self, text: str) -> int:
        pass

    @abstractmethod
    def _trim_to_max_tokens(self, text: str, max_tokens: int) -> str:
        pass

    async def generate(
        self,
        template: Template,
        template_params: dict[str, str],
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        **extra_options: dict,
    ) -> tuple[str, FinishReason]:
        self._validate_template_params(
            template=template,
            template_params=template_params,
            max_output_tokens=max_output_tokens,
        )
        input_text = template.substitute(template_params=template_params)

        log.debug(f"{self.name} generating response for input with {self.get_num_tokens(input_text)} tokens")
        output_text, finish_reason = await self._generate(template=template, template_params=template_params)
        if output_text is None:
            raise ValueError(f"{self.name} failed to generate a response: {finish_reason}")

        log.debug(f"{self.name} generated response with {len(output_text)} characters")

        self._maybe_log_verbose(input_text=input_text, output_text=output_text)
        if self.log_examples and self.example_logger is not None:
            await self.example_logger.a_log(
                llm_name=self.name,
                template=str(template),
                template_params=template_params,
                output_text=output_text,
                inference_params={"temperature": temperature, "max_output_tokens": max_output_tokens, **extra_options},
            )

        return output_text, finish_reason

    async def generate_with_output_model(
        self,
        template: Template,
        template_params: dict[str, str],
        output_model: type[T],
        max_attempts: int = 3,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        **extra_options: dict,
    ) -> T:
        attempts = 0
        while attempts < max_attempts:
            output_text, finish_reason = await self._generate(
                template=template,
                template_params=template_params,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                **extra_options,
            )
            if output_text is not None:
                output_text = output_text[output_text.find("{") : output_text.rfind("}") + 1]
                try:
                    return output_model.model_validate_json(output_text)
                except ValidationError as e:
                    log.error(f"Error validating output model: {e}")

            attempts += 1

        raise ValueError(f"Failed to generate response after {attempts} attempts")

    @lru_cache(maxsize=1000)  # noqa: B019
    def get_num_tokens(self, text: str) -> int:
        return self._get_num_tokens(text=text)

    def trim_to_max_tokens(self, text: str, max_tokens: int) -> str:
        return self._trim_to_max_tokens(text=text, max_tokens=max_tokens)

    def get_cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return input_tokens * self.input_token_cost_usd + output_tokens * self.output_token_cost_usd

    def exceeds_max_tokens(self, template: Template, template_params: dict[str, str], max_output_tokens: int) -> bool:
        input_text = template.substitute(template_params=template_params)
        num_input_tokens = self.get_num_tokens(text=input_text)
        return num_input_tokens + max_output_tokens > self.context_window_size

    def _build_messages(self, template: Template, template_params: dict[str, str]) -> list[dict[str, str]]:
        return [{"role": "user", "content": template.substitute(**template_params)}]

    def _validate_template_params(
        self,
        template: Template,
        template_params: dict[str, str],
        max_output_tokens: int,
    ) -> None:
        if self.exceeds_max_tokens(
            template=template,
            template_params=template_params,
            max_output_tokens=max_output_tokens,
        ):
            input_text = template.substitute(template_params=template_params)
            num_input_tokens = self.get_num_tokens(text=input_text)
            raise ValueError(
                f"Input tokens ({num_input_tokens}) + max output tokens ({max_output_tokens}) "
                f"exceeds context window size ({self.context_window_size})",
            )

    def __repr__(self) -> str:
        return f"{self.name}({self.llm_name})"

    def _maybe_log_verbose(self, input_text: str, output_text: str) -> None:
        if settings.llm.echo:
            log.info(f"{self.name} verbose mode:\n\nPROMPT\n\n{input_text}\n\nCOMPLETION\n\n{output_text}\n")
