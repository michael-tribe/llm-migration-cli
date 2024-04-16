from string import Template

from src.llms.base import BaseLLM
from src.llms.enums import FinishReason
from src.logger import root_logger


log = root_logger.getChild(__name__)


class MockLLM(BaseLLM):
    llm_name = "mock"
    max_tokens = 1000
    input_token_cost_usd: float = 0.0001
    output_token_cost_usd: float = 0.0002

    async def _generate(
        self,
        template: Template,
        template_params: dict[str, str],
        temperature: float = 0.0,
        max_output_tokens: int = 0,
        **extra_options: dict,
    ) -> tuple[str | None, FinishReason]:
        return self._build_mock_response(template=template, template_params=template_params), FinishReason.SUCCESS

    def _get_num_tokens(self, text: str) -> int:
        return len(text.split(" "))

    def _build_mock_response(self, template: Template, template_params: dict[str, str]) -> str:
        return f"Mock response for input: \n\n```{template.substitute(**template_params)}```"
