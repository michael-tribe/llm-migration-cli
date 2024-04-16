import json
from abc import abstractmethod
from csv import DictReader, DictWriter
from pathlib import Path

from class_registry import ClassRegistry

from src.dtos.example import Example
from src.dtos.result import Result
from src.llms.base import BaseLLM
from src.logger import root_logger
from src.settings import settings


log = root_logger.getChild(__name__)
evals_registry = ClassRegistry()


class BaseEvaluation:
    question: str | None = None
    llm_evaluation_template: str | None = None
    default_template_path: Path | None = None

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __init__(self, question: str | None, template: str | None = None, output_dir: str | None = None):
        self.question = question if question else self.question
        self.llm_evaluation_template = template if template else self._load_default_template()
        self.output_dir = Path(output_dir) if output_dir else settings.paths.default_output_dir

    @abstractmethod
    def _evaluate_examples_by_human(self, examples: list[Example], evaluator_name: str) -> list[Result]:
        pass

    @abstractmethod
    def _evaluate_examples_by_llm(self, examples: list[Example], llm: BaseLLM) -> list[Result]:
        pass

    @abstractmethod
    def _display_results(self, results: list[Result]) -> None:
        pass

    def evaluate_examples_by_human(self, examples: list[Example]) -> None:
        if not examples:
            log.info("No examples to evaluate")
            return

        log.info(f"Evaluating {len(examples)} examples by human")
        self._evaluate_examples_by_human(examples=examples, evaluator_name="human")
        all_results = self.load_results(prompt_name=examples[0].prompt_name)  # type: ignore
        if not all_results:
            log.warning("No results to display")
            return

        self.display_results(results=all_results)

    def evaluate_examples_by_llm(self, examples: list[Example], llm: BaseLLM) -> None:
        if not examples:
            log.info("No examples to evaluate")
            return

        log.info(f"Evaluating {len(examples)} examples by LLM")
        self._evaluate_examples_by_llm(examples=examples, llm=llm)
        all_results = self.load_results(prompt_name=examples[0].prompt_name)  # type: ignore
        if not all_results:
            log.warning("No results to display")
            return

        self.display_results(results=all_results)

    def display_results(self, results: list[Result]) -> None:
        if not results:
            log.info("No results to display")
            return

        self._display_results(results=results)

    def log_result(self, result: Result) -> None:
        output_path = self._build_evaluation_results_output_path(prompt_name=result.prompt_name)
        if not output_path.exists():
            with output_path.open("w") as f:
                writer = DictWriter(f, fieldnames=result.model_dump().keys())
                writer.writeheader()

        log.debug(f"Logging evaluation result to {output_path}")
        with output_path.open("a") as f:
            data = result.model_dump()
            writer = DictWriter(f, fieldnames=data.keys())
            writer.writerow(data)

    def load_examples(self, prompt_name: str) -> list[Example]:
        examples_path = self.output_dir / prompt_name / settings.paths.evaluation_examples_filename
        if not examples_path.exists():
            return []

        with examples_path.open("r") as f:
            example_lines = f.read().splitlines()

        return [Example(**json.loads(line)) for line in example_lines]

    def load_results(self, prompt_name: str, evaluator_name: str | None = None) -> list[Result]:
        output_path = self._build_evaluation_results_output_path(prompt_name=prompt_name)
        if not output_path.exists():
            return []

        with output_path.open("r") as f:
            reader = DictReader(f)
            results = [Result(**row) for row in reader]  # type: ignore

        results = [
            result
            for result in results
            if result.evaluation_name == self.name and result.evaluation_question == self.question
        ]
        if evaluator_name:
            return [result for result in results if result.evaluator_name == evaluator_name]

        return results

    def _filter_evaluated_examples(self, examples: list[Example], evaluator_name: str) -> list[Example]:
        if not examples:
            return []

        results = self.load_results(prompt_name=examples[0].prompt_name)  # type: ignore
        evaluated_example_uuids = [r.example_uuid for r in results if r.evaluator_name == evaluator_name]
        return [example for example in examples if example.uuid not in evaluated_example_uuids]

    def _build_evaluation_results_output_path(self, prompt_name: str) -> Path:
        output_path = self.output_dir / prompt_name / self.name / settings.paths.evaluation_results_filename
        output_path.parent.mkdir(exist_ok=True, parents=True)
        return output_path

    def _load_default_template(self) -> str | None:
        if not self.default_template_path:
            return None

        return self.default_template_path.read_text()
