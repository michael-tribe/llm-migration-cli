from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# NOTE: BaseSettings loads all settings from environment variables case-insensitive
class LLMSettings(BaseSettings):
    anthropic_api_key: str | None = Field(None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(None, alias="OPENAI_KEY")
    echo: bool = False

    model_config = SettingsConfigDict(env_file="secrets.env", extra="ignore")


class Paths(BaseSettings):
    default_output_dir: Path = Path("data/evaluations")
    evaluation_config_filename: str = "evals.yaml"
    evaluation_results_filename: str = "results.csv"
    evaluation_examples_filename: str = "examples.jsonl"
    flagged_examples_filename: str = "flagged.jsonl"
    removed_examples_filename: str = "removed.jsonl"

    model_config = SettingsConfigDict(env_file="vars.env", extra="ignore")


class GlobalSettings(BaseSettings):
    llm: LLMSettings = LLMSettings()
    paths: Paths = Paths()

    model_config = SettingsConfigDict(
        env_file=("secrets.env", "vars.env"),
        env_nested_delimiter="__",
        extra="ignore",
    )


settings = GlobalSettings()
