import random
from pathlib import Path
from string import Template

from pydantic import BaseModel, Field, ValidationError
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel

from src.dtos.example import Example
from src.dtos.result import Result
from src.evals.base import BaseEvaluation, evals_registry
from src.llms.base import BaseLLM
from src.logger import root_logger
from src.utils.asyncio import run_async_tasks
from src.utils.cli import create_progress_bar_panel


log = root_logger.getChild(__name__)


class RatingResponse(BaseModel):
    reasoning: str | None = None
    rating: float = Field(..., ge=1, le=5)


@evals_registry.register("rating")
class RatingEvaluation(BaseEvaluation):
    llm_evaluation_template: str | None = None
    default_template_path: Path | None = Path(__file__).parent / "rating_template.txt"

    question: str = "How would you rate this response?"
    response_instructions: str = (
        "Answer 1 for poor, 2 for below average, 3 for average, 4 for above average, 5 for excellent."
    )
    min_value: float = 1
    max_value: float = 5

    def __init__(self, question: str | None = None, template: str | None = None, output_dir: str | None = None):
        super().__init__(question=question, template=template, output_dir=output_dir)

    def _evaluate_examples_by_human(self, examples: list[Example], evaluator_name: str = "human") -> list[Result]:
        filtered_examples = self._filter_evaluated_examples(examples=examples, evaluator_name=evaluator_name)
        if not filtered_examples:
            log.info(f"All examples have been evaluated by {evaluator_name}")
            return self.load_results(prompt_name=examples[0].prompt_name, evaluator_name=evaluator_name)  # type: ignore

        log.info(f"Evaluating {len(filtered_examples)} examples by human")
        console = Console()
        layout = self.build_cli_evaluation_layout()

        results = []
        random.shuffle(filtered_examples)
        for idx, example in enumerate(filtered_examples):
            progress_bar = create_progress_bar_panel(n_done=idx, total=len(filtered_examples), width=console.width - 20)
            layout["progress"].update(progress_bar)
            layout["content_left"].update(Panel(example.input_text, title="Input", border_style="blue"))
            layout["content_right"].update(Panel(example.output_text, title="Output", border_style="blue"))
            console.clear()
            console.print(layout)

            rating = self._get_rating_from_user(console)
            reasoning = (
                console.input("Provide your reasoning for this rating (optional, press enter to skip): ") or None
            )

            result = Result(
                evaluation_name=self.name,
                evaluation_question=self.question,
                evaluator_name=evaluator_name,
                prompt_name=example.prompt_name,  # type: ignore
                llm_name=example.llm_name,
                example_uuid=example.uuid,
                result=rating.rating,
                reasoning=reasoning,
            )
            self.log_result(result)
            results.append(result)
            log.debug(f"{evaluator_name} rated example {example.uuid} {rating.rating}")

        return results

    def _get_rating_from_user(self, console: Console) -> RatingResponse:
        while True:
            rating = console.input(f"Rate this example {self.min_value}-{self.max_value}: ")
            try:
                return RatingResponse(rating=rating)
            except ValidationError:
                console.print("Invalid rating. Please enter a number between 0 and 5.")

    def _evaluate_examples_by_llm(self, examples: list[Example], llm: BaseLLM) -> list[Result]:
        filtered_examples = self._filter_evaluated_examples(examples=examples, evaluator_name=llm.name)
        if not filtered_examples:
            log.info(f"All examples have been evaluated by {llm.name}")
            return self.load_results(prompt_name=examples[0].prompt_name, evaluator_name=llm.name)  # type: ignore

        log.info(f"Evaluating {len(examples)} examples by LLM")
        return run_async_tasks([self._evaluate_example_by_human(llm, example) for example in filtered_examples])

    async def _evaluate_example_by_human(self, llm: BaseLLM, example: Example) -> Result:
        log.info(f"Evaluating example {example.uuid} by {llm.name}")
        template_params: dict[str, str] = {
            "input_text": example.input_text,
            "output_text": example.output_text,
            "question": self.question,
            "min_value": str(self.min_value),
            "max_value": str(self.max_value),
        }

        rating = await llm.generate_with_output_model(
            template=Template(self.llm_evaluation_template),  # type: ignore
            template_params=template_params,
            max_output_tokens=200,
            temperature=0.5,
            output_model=RatingResponse,
        )

        log.debug(f"Rating: {rating.rating}, Reasoning: {rating.reasoning}")
        result = Result(
            evaluation_name=self.name,
            evaluation_question=self.question,
            evaluator_name=llm.name,
            prompt_name=example.prompt_name,  # type: ignore
            llm_name=example.llm_name,
            example_uuid=example.uuid,
            result=rating.rating,
            reasoning=rating.reasoning,
        )

        self.log_result(result)
        log.debug(f"{llm.name} rated example {example.uuid} {rating.rating}")
        return result

    def _display_results(self, results: list[Result]) -> None:
        llm_results_text, evaluator_results_text, worst_examples_text = self._render_results(results=results)

        console = Console()
        layout = self._build_cli_results_layout()

        layout["title"].update(
            Panel(
                Align.center(f"Results for {self.name}", vertical="middle"),
                title="Evaluation",
                border_style="yellow",
            ),
        )
        layout["results"]["llm_results"].update(
            Panel(Align.center(llm_results_text, vertical="middle"), title="Results", border_style="blue"),
        )
        layout["results"]["evaluator_results"].update(
            Panel(Align.center(evaluator_results_text, vertical="middle"), title="Results", border_style="blue"),
        )
        layout["worst_examples"].update(Panel(worst_examples_text, title="Worst Examples", border_style="red"))

        console.clear()
        console.print(layout)
        console.input("Press enter to continue...")
        console.clear()

    def _render_results(self, results: list[Result]) -> tuple[str, str, str]:
        llm_names = {r.llm_name for r in results}
        results_by_llm = {llm_name: [r for r in results if r.llm_name == llm_name] for llm_name in llm_names}

        llm_results_text = self._render_llm_results_text(results_by_llm=results_by_llm)
        evaluator_results_text = self._render_evaluator_results_text(results=results, results_by_llm=results_by_llm)
        worst_examples_text = self._render_worst_examples_text(results, results_by_llm)
        return llm_results_text, evaluator_results_text, worst_examples_text

    def _render_llm_results_text(self, results_by_llm: dict[str, list[Result]]) -> str:
        llm_results_text = ""
        llm_average_ratings = {
            llm: sum(r.result for r in results) / len(results) for llm, results in results_by_llm.items()
        }
        best_llm = max(llm_average_ratings, key=lambda llm: llm_average_ratings[llm])

        for llm_name, llm_results in sorted(
            results_by_llm.items(),
            key=lambda item: llm_average_ratings[item[0]],
            reverse=True,
        ):
            if llm_name == best_llm:
                extra_text = "[green] (Winner)[/green]"
                result_color = "green"
            else:
                extra_text = ""
                result_color = "yellow"

            llm_results_text += f"LLM: [{result_color}]{llm_name}[/{result_color}] {extra_text}\n"

            average_rating = round(sum(r.result for r in llm_results) / len(llm_results), 2)
            llm_results_text += f"Average Rating: [{result_color}]{average_rating}[/{result_color}]\n"

            distribution = {i: 0 for i in range(6)}
            for r in llm_results:
                distribution[int(r.result)] += 1

            llm_results_text += f"Rating Distribution:\n"
            for i, count in distribution.items():
                llm_results_text += f"{i}: {count} {'█' * count}\n"

            llm_results_text += "\n"

        return llm_results_text

    def _render_evaluator_results_text(self, results: list[Result], results_by_llm: dict[str, list[Result]]) -> str:
        evaluator_results_text = ""
        evaluators = {r.evaluator_name for r in results}
        results_by_evaluator = {
            evaluator: [r for r in results if r.evaluator_name == evaluator] for evaluator in evaluators
        }
        for evaluator, results in results_by_evaluator.items():
            evaluator_results_text += f"[blue]Evaluator: {evaluator}[/blue]\n"

            for llm_name in results_by_llm:
                llm_results = [r for r in results if r.llm_name == llm_name]
                average_rating = round(sum(r.result for r in llm_results) / len(llm_results), 2)
                distribution = {i: 0 for i in range(6)}
                for r in llm_results:
                    distribution[int(r.result)] += 1

                evaluator_results_text += f"[green]LLM: {llm_name}, Average Rating: {average_rating}[/green]\n"
                evaluator_results_text += f"[]Rating Distribution:\n"
                for i, count in distribution.items():
                    evaluator_results_text += f"{i}: {count} {'█' * count}\n"

                evaluator_results_text += "\n"

        return evaluator_results_text

    def _render_worst_examples_text(self, results: list[Result], results_by_llm: dict[str, list[Result]]) -> str:
        worst_examples_text = ""
        examples = self.load_examples(prompt_name=results[0].prompt_name)  # type: ignore
        examples_by_uuid = {example.uuid: example for example in examples}
        worst_examples_text += "\n\n"
        for llm, llm_results in results_by_llm.items():
            worst_examples = sorted(llm_results, key=lambda r: r.result)[:3]
            worst_examples_text += f"[red]Worst Examples for LLM {llm}:[/red]\n"
            for r in worst_examples:
                example = examples_by_uuid[r.example_uuid]
                worst_examples_text += f"[yellow]Example UUID[/yellow]: {r.example_uuid}\n"
                worst_examples_text += f"[yellow]Rating[/yellow]: {r.result}\n"
                worst_examples_text += f"[yellow]Reasoning[/yellow]: {r.reasoning}\n"
                worst_examples_text += f"[yellow]Input (truncated)[/yellow]: {example.input_text[:100]}\n"
                worst_examples_text += f"[yellow]Output (truncated)[/yellow]: {example.output_text[:100]}\n"
                worst_examples_text += "\n"

            worst_examples_text += "\n"
        return worst_examples_text

    def build_cli_evaluation_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="overscan", size=1),
            Layout(name="title", size=5),
            Layout(name="progress", size=3),
            Layout(name="content", ratio=2),
            Layout(name="instructions", size=5),
        )
        layout["content"].split_row(Layout(name="content_left"), Layout(name="content_right"))

        layout["title"].update(
            Panel(
                Align.center(f"Rate Examples {self.min_value}-{self.max_value}", vertical="middle"),
                title="Evaluation",
                border_style="yellow",
            ),
        )
        layout["instructions"].update(
            Panel(
                Align.center(self.question + "\n" + self.response_instructions, vertical="middle"),
                title="Instructions",
                border_style="green",
            ),
        )
        return layout

    def _build_cli_results_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="overscan", size=1),
            Layout(name="title", size=5),
            Layout(name="results"),
            Layout(name="worst_examples", size=10),
        )
        layout["results"].split_row(Layout(name="llm_results"), Layout(name="evaluator_results"))
        return layout
