import logging

from .client import InstrumentedLlmClient, Provider
from .prompts.summarize import summarize_text
from .prompts.write_poem import write_a_poem


task_map = {"write_a_poem": write_a_poem, "summarize_text": summarize_text}


def generate_examples(
    task: str | None = None,
    provider: Provider | None = None,
) -> None:
    if task and task not in task_map:
        raise ValueError(f"Invalid task: {task}")

    tasks = [task] if task else list(task_map)
    providers = [provider] if provider else list(Provider)
    for provider in providers:
        client = InstrumentedLlmClient(provider=provider)
        for task in tasks:
            logging.info(f"Generating examples for task: {task} (provider: {provider})")
            task_map[task](client=client)
