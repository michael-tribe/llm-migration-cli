from enum import Enum

from anthropic import Anthropic
from openai import OpenAI

from src.example_logger import ExampleLogger
from src.settings import settings


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class InstrumentedLlmClient:
    default_anthropic_model = "claude-3-haiku-20240307"
    default_openai_model = "gpt-3.5-turbo"

    def __init__(self, provider: Provider | None = None, model: str | None = None) -> None:
        self.provider = provider or Provider.ANTHROPIC
        if self.provider == Provider.ANTHROPIC:
            self.client = Anthropic(api_key=settings.llm.anthropic_api_key)
            self.model = model or self.default_anthropic_model
        else:
            self.client = OpenAI(api_key=settings.llm.openai_api_key)
            self.model = model or self.default_openai_model

        self.example_logger = ExampleLogger()

    def generate(
        self,
        template: str,
        variables: dict[str, str],
        model: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 100,
    ) -> str:
        if self.provider == Provider.ANTHROPIC:
            return self._generate_anthropic(
                template=template,
                variables=variables,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        return self._generate_openai(
            template=template,
            variables=variables,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _generate_openai(
        self,
        template: str,
        variables: dict[str, str],
        model: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 100,
    ) -> str:
        model = model or self.model
        filled_prompt = template.format(**variables)
        completion = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": filled_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        output_text = completion.choices[0].message.content
        self.example_logger.log(
            llm_name=model,
            template=template,
            template_params=variables,
            output_text=output_text,
            inference_params={"temperature": temperature, "max_tokens": max_tokens},
        )
        return output_text

    def _generate_anthropic(
        self,
        template: str,
        variables: dict[str, str],
        model: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 100,
    ) -> str:
        model = model or self.model
        filled_prompt = template.format(**variables)
        message = self.client.messages.create(
            messages=[{"role": "user", "content": filled_prompt}],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        output_text = message.content[0].text
        self.example_logger.log(
            llm_name=model,
            template=template,
            template_params=variables,
            output_text=output_text,
            inference_params={"temperature": temperature, "max_tokens": max_tokens},
        )
        return output_text
