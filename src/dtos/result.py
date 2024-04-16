from uuid import UUID

from pydantic import BaseModel, field_validator


class Result(BaseModel):
    evaluation_name: str
    evaluation_question: str
    evaluator_name: str

    prompt_name: str
    llm_name: str

    example_uuid: UUID

    result: float
    reasoning: str | None = None

    comparison_example_uuid: UUID | None = None
    comparison_llm_name: str | None = None

    # NOTE: this is done to cover reading in empty string values from CSVs
    @field_validator("comparison_example_uuid", mode="before")
    def _validate_comparison_example_uuid(cls, v: UUID | str | None) -> UUID | None:
        if not v:
            return None

        if isinstance(v, str):
            try:
                return UUID(v)
            except ValueError:
                raise ValueError("comparison_example_uuid must be a valid UUID") from None

        return v
