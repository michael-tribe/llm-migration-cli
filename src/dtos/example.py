from hashlib import md5
from string import Template
from uuid import UUID, uuid4

import jinja2
from pydantic import BaseModel, Field, model_validator

from src.enums import TemplateStyle


# TODO: find better pattern for prompt_name being optional to pass in, but always being a string due to model_validator
class Example(BaseModel):
    uuid: UUID = Field(default_factory=uuid4)

    prompt_name: str | None = None
    llm_name: str

    template: str
    template_params: dict[str, str] = Field(default_factory=dict)

    output_text: str

    inference_params: dict | None = None
    metadata: dict | None = None

    input_hash: str | None = None
    template_hash: str | None = None

    @model_validator(mode="after")  # type: ignore
    def set_computed_fields(self) -> None:
        input_text = self.render()
        self.input_hash = md5(input_text.encode()).hexdigest()  # noqa: S324
        self.template_hash = md5(self.template.encode()).hexdigest()[:8]  # noqa: S324
        if not self.prompt_name:
            self.prompt_name = f"prompt-{self.template_hash}"

    @property
    def template_style(self) -> TemplateStyle:
        # TODO: make more robust with regex
        if "${" in self.template:
            return TemplateStyle.PYTHON_TEMPLATE

        if "{{" in self.template:
            return TemplateStyle.JINJA

        if "{" in self.template:
            return TemplateStyle.PYTHON_STRING

        raise ValueError("Unknown template style")

    @property
    def input_text(self) -> str:
        return self.render()

    def render(self) -> str:
        if not self.template_params:
            return self.template

        if self.template_style == TemplateStyle.JINJA:
            return jinja2.Template(self.template).render(**self.template_params)

        if self.template_style == TemplateStyle.PYTHON_STRING:
            return self.template.format_map(self.template_params)

        if self.template_style == TemplateStyle.PYTHON_TEMPLATE:
            return Template(self.template).substitute(**self.template_params)

        raise ValueError("Unknown template style")
