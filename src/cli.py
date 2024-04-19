import json
from pathlib import Path

from typer import Typer, secho

from src.dtos.example import Example
from src.evals.base import evals_registry
from src.llms.base import llm_registry
from src.settings import settings


cli = Typer(no_args_is_help=True)


@cli.command("list")
def list_command(items: str) -> None:
    if items == "evals":
        secho("\nAvailable evaluations:", bold=True)
        for evaluation in evals_registry:
            secho(f"  - {evaluation}")

    elif items == "llms":
        secho("\nAvailable LLMs:", bold=True)
        for llm in llm_registry:
            secho(f"  - {llm}")

    elif items == "prompts":
        secho("\nAvailable prompts:", bold=True)
        for prompt_dir in settings.paths.default_output_dir.iterdir():
            if prompt_dir.is_dir():
                secho(f"  - {prompt_dir.name}")

    else:
        secho(f"Unknown item: {items} (available: evals, llms, prompts)", fg="red")
    
    secho("")


@cli.command()
def filter_examples(prompt_name: str) -> None:
    # filter a large amount of examples by a certain criteria
    #  - manually discard examples that are not useful
    #  - filter by cluster based on embeddings
    #  - filter by metadata
    #  - filter by score from llm evaluation
    secho("Filtering examples not yet implemented", fg="red")
    raise SystemExit


@cli.command()
def evaluate(evaluation: str, prompt_name: str, llm: str | None = None) -> None:
    if evaluation not in evals_registry:
        secho(f"Unknown evaluation: {evaluation} (available: {', '.join(evals_registry.keys())})", fg="red")
        raise SystemExit

    prompt_dir = settings.paths.default_output_dir / prompt_name
    if not prompt_dir.exists():
        available_prompts = [p.name for p in settings.paths.default_output_dir.iterdir() if p.is_dir()]
        secho(f"Prompt not found: {prompt_dir} (available: {', '.join(available_prompts)})", fg="red")
        raise SystemExit

    evaluation_obj = evals_registry[evaluation]
    examples = _load_examples(prompt_dir=prompt_dir)

    if llm is None:
        evaluation_obj.evaluate_examples_by_human(examples=examples)
        return

    if llm not in llm_registry:
        secho(f"Unknown LLM: {llm} (available: {', '.join(llm_registry.keys())})", fg="red")
        raise SystemExit

    llm_obj = llm_registry[llm]
    evaluation_obj.evaluate_examples_by_llm(examples=examples, llm=llm_obj)


@cli.command()
def results(evaluation: str, prompt_name: str) -> None:
    if evaluation not in evals_registry:
        secho(f"Unknown evaluation: {evaluation} (available: {', '.join(evals_registry.keys())})", fg="red")
        raise SystemExit

    prompt_dir = settings.paths.default_output_dir / prompt_name
    if not prompt_dir.exists():
        available_prompts = [p.name for p in settings.paths.default_output_dir.iterdir() if p.is_dir()]
        secho(f"Prompt not found: {prompt_dir} (available: {', '.join(available_prompts)})", fg="red")
        raise SystemExit

    evaluation_obj = evals_registry[evaluation]
    results = evaluation_obj.load_results(prompt_name=prompt_name)
    evaluation_obj.display_results(results=results)


def _load_examples(prompt_dir: Path) -> list[Example]:
    if not prompt_dir.exists():
        secho(f"prompt_dir does not exist: {prompt_dir}", fg="red")
        raise SystemExit

    if not prompt_dir.is_dir():
        secho(f"prompt_dir is not a directory: {prompt_dir}", fg="red")
        raise SystemExit

    examples_path = prompt_dir / settings.paths.evaluation_examples_filename
    if not examples_path.exists():
        secho(f"Examples file not found: {examples_path}", fg="red")
        raise SystemExit

    example_lines = examples_path.read_text().splitlines()
    examples = [Example(**json.loads(line)) for line in example_lines]
    if not examples:
        secho(f"No examples found in {examples_path}", fg="red")
        raise SystemExit

    return examples
