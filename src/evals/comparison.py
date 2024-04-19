import random
from enum import Enum
from pathlib import Path
from string import Template

from pydantic import BaseModel, ValidationError, field_validator
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


class ComparisonResponse(Enum):
    FIRST_BETTER = 1
    SECOND_BETTER = 2
    EQUALLY_GOOD = 3
    ABSTAIN = 4


class RatingResponse(BaseModel):
    reasoning: str | None = None
    rating: ComparisonResponse

    @field_validator("rating", mode="before")
    def convert_rating(cls, value: int) -> ComparisonResponse:
        try:
            value = int(value)
        except ValueError:
            raise ValueError("Rating must be an integer") from None

        try:
            return ComparisonResponse(value)
        except ValueError:
            raise ValueError("Rating must be one of 1, 2, 3, 4") from None


@evals_registry.register("comparison")
class ComparisonEvaluation(BaseEvaluation):
    llm_evaluation_template: str | None = None
    default_template_path: Path | None = Path(__file__).parent / "comparison_template.txt"

    question: str = "Which of the outputs is better?"
    response_instructions: str = (
        "Answer '1' for the first output, '2' for the second output, '3' if they are equally good, '4' to abstain."
    )
    allowed_values: list[int] = [1, 2, 3, 4, 5, 6]

    def __init__(self, question: str | None = None, template: str | None = None, output_dir: str | None = None):
        super().__init__(question=question, template=template, output_dir=output_dir)

    # TODO: simplify this method
    def _evaluate_examples_by_human(self, examples: list[Example], evaluator_name: str = "human") -> list[Result]:
        main_llm_name, comparison_llm_name, main_examples, comparison_examples = self._split_examples_by_llm_name(
            examples=examples,
        )

        filtered_examples = self._filter_evaluated_examples(examples=main_examples, evaluator_name=evaluator_name)
        if not filtered_examples:
            log.info("All examples have been evaluated by human")
            return self.load_results(prompt_name=examples[0].prompt_name, evaluator_name=evaluator_name)  # type: ignore

        pairs = self._build_example_pairs(main_examples=filtered_examples, comparison_examples=comparison_examples)
        if not pairs:
            log.info("No valid pairs found for comparison evaluation")
            return self.load_results(prompt_name=examples[0].prompt_name, evaluator_name=evaluator_name)  # type: ignore

        log.info(f"Evaluating {len(pairs)} examples by human")
        console = Console()
        layout = self._build_cli_layout()

        results = []
        random.shuffle(pairs)
        for idx, (main_example, comparison_example) in enumerate(pairs):
            progress_bar = create_progress_bar_panel(n_done=idx, total=len(pairs), width=console.width - 20)
            first_example = random.choice([main_example, comparison_example])  # noqa: S311
            second_example = main_example if first_example == comparison_example else comparison_example

            layout["progress"].update(progress_bar)
            layout["input_text"].update(Panel(main_example.input_text, title="Context", border_style="blue"))
            layout["content"]["first_example"].update(
                Panel(first_example.output_text, title="First", border_style="blue")
            )
            layout["content"]["second_example"].update(
                Panel(second_example.output_text, title="Second", border_style="blue")
            )
            console.clear()
            console.print(layout)

            rating = self._get_rating_from_user(console)
            reasoning = (
                console.input("Provide your reasoning for this rating (optional, press enter to skip): ") or None
            )

            if rating.rating == ComparisonResponse.FIRST_BETTER and first_example == comparison_example:
                rating.rating = ComparisonResponse.SECOND_BETTER
            elif rating.rating == ComparisonResponse.SECOND_BETTER and second_example == main_example:
                rating.rating = ComparisonResponse.FIRST_BETTER

            result = Result(
                evaluation_name=self.name,
                evaluation_question=self.question,
                evaluator_name=evaluator_name,
                prompt_name=main_example.prompt_name,  # type: ignore
                example_uuid=main_example.uuid,
                llm_name=main_llm_name,
                comparison_example_uuid=comparison_example.uuid,
                comparison_llm_name=comparison_llm_name,
                result=rating.rating.value,
                reasoning=reasoning,
            )
            self.log_result(result)
            results.append(result)

            preferred_llm = main_llm_name if rating.rating == 1 else comparison_llm_name
            preferred_llm = "both" if rating.rating == 3 else preferred_llm
            log.debug(f"{evaluator_name} preferred {preferred_llm} output")

        return results

    def _evaluate_examples_by_llm(self, examples: list[Example], llm: BaseLLM) -> list[Result]:
        main_llm_name, comparison_llm_name, main_examples, comparison_examples = self._split_examples_by_llm_name(
            examples=examples,
        )

        filtered_examples = self._filter_evaluated_examples(examples=main_examples, evaluator_name=llm.name)
        if not filtered_examples:
            log.info("All examples have been evaluated by human")
            return self.load_results(prompt_name=examples[0].prompt_name, evaluator_name=llm.name)  # type: ignore

        pairs = self._build_example_pairs(main_examples=filtered_examples, comparison_examples=comparison_examples)
        if not pairs:
            log.info("No valid pairs found for comparison evaluation")
            return self.load_results(prompt_name=examples[0].prompt_name, evaluator_name=llm.name)  # type: ignore

        log.info(f"Evaluating {len(pairs)} examples by LLM")
        tasks = []
        for main_example, comparison_example in pairs:
            tasks.append(
                self._evaluate_example_by_llm(
                    llm=llm,
                    main_llm_name=main_llm_name,
                    comparison_llm_name=comparison_llm_name,
                    main_example=main_example,
                    comparison_example=comparison_example,
                ),
            )
        return run_async_tasks(tasks)

    async def _evaluate_example_by_llm(
        self,
        llm: BaseLLM,
        main_llm_name: str,
        comparison_llm_name: str,
        main_example: Example,
        comparison_example: Example,
    ) -> Result:
        log.info(f"Evaluating examples {main_example.uuid} and {comparison_example.uuid} by {llm.name}")
        template_params: dict[str, str] = {
            "input_text": main_example.input_text,
            "first_output": main_example.output_text,
            "second_output": comparison_example.output_text,
            "question": self.question,
            "response_instructions": self.response_instructions,
        }

        rating = await llm.generate_with_output_model(
            template=Template(self.llm_evaluation_template),  # type: ignore
            template_params=template_params,
            max_output_tokens=200,
            temperature=0.5,
            output_model=RatingResponse,
        )

        result = Result(
            evaluation_name=self.name,
            evaluation_question=self.question,
            evaluator_name=llm.name,
            prompt_name=main_example.prompt_name,  # type: ignore
            example_uuid=main_example.uuid,
            llm_name=main_example.llm_name,
            comparison_example_uuid=comparison_example.uuid,
            comparison_llm_name=comparison_example.llm_name,
            result=float(rating.rating.value),
            reasoning=rating.reasoning,
        )
        self.log_result(result)

        if rating.rating == ComparisonResponse.FIRST_BETTER:
            log.debug(f"{llm.name} preferred the {main_llm_name} output")
        elif rating.rating == ComparisonResponse.SECOND_BETTER:
            log.debug(f"{llm.name} preferred the {comparison_llm_name} output")
        elif rating.rating == ComparisonResponse.EQUALLY_GOOD:
            log.debug(f"{llm.name} found both outputs equally good")
        elif rating.rating == ComparisonResponse.ABSTAIN:
            log.debug(f"{llm.name} abstained from rating the outputs")
        return result

    def _build_example_pairs(
        self, main_examples: list[Example], comparison_examples: list[Example]
    ) -> list[tuple[Example, Example]]:
        comparison_examples_by_input_hash = {e.input_hash: e for e in comparison_examples}
        pairs: list[tuple[Example, Example]] = []
        for main_example in sorted(main_examples, key=lambda e: e.input_hash):  # type: ignore
            comparison_example = comparison_examples_by_input_hash.get(main_example.input_hash, None)
            if not comparison_example:
                log.warning(f"No comparison example found for input hash {main_example.input_hash}, skipping...")
                continue

            pairs.append((main_example, comparison_example))

        log.info(f"Built {len(pairs)} example pairs for comparison evaluation")
        return pairs

    def _split_examples_by_llm_name(self, examples: list[Example]) -> tuple[str, str, list[Example], list[Example]]:
        examples_by_llm_name: dict[str, list[Example]] = {}
        for example in examples:
            examples_by_llm_name.setdefault(example.llm_name, []).append(example)

        if len(examples_by_llm_name) > 2:
            raise ValueError(
                f"Comparison evaluation can only handle two LLM's at a time, got {len(examples_by_llm_name)}",
            )

        llm_names = list(examples_by_llm_name.keys())
        main_llm_name, comparison_llm_name = sorted(llm_names)
        main_examples = examples_by_llm_name[main_llm_name]
        comparison_examples = examples_by_llm_name[comparison_llm_name]
        return main_llm_name, comparison_llm_name, main_examples, comparison_examples

    def _get_rating_from_user(self, console: Console) -> RatingResponse:
        while True:
            rating = console.input(f"Rate the outputs (1/2/3/4): ")
            try:
                return RatingResponse(rating=rating)
            except ValidationError as e:
                log.error(f"Invalid input: {e}")
                console.print("Invalid input, please try again.")

    def _display_results(self, results: list[Result]) -> None:
        llm_results_text, evaluator_results_text = self._render_results(results=results)

        console = Console()
        layout = self._build_cli_results_layout()

        layout["title"].update(
            Panel(Align.center(f"Results for {self.name}", vertical="middle"), title="Title", border_style="blue"),
        )
        layout["results"]["llm_results"].update(
            Panel(Align.center(llm_results_text, vertical="middle"), title="LLM Results", border_style="blue"),
        )
        layout["results"]["evaluator_results"].update(
            Panel(
                Align.center(evaluator_results_text, vertical="middle"),
                title="Evaluator Results",
                border_style="blue",
            ),
        )

        console.clear()
        console.print(layout)
        console.input("Press enter to continue...")
        console.clear()

    def _render_results(self, results: list[Result]) -> tuple[str, str]:
        # TODO: handle case of multiple comparison evaluations stored in the same results file
        main_llm = results[0].llm_name
        comparison_llm = results[0].comparison_llm_name

        llm_results_text = self._render_llm_results_text(results=results, main_llm=main_llm, comparison_llm=comparison_llm)  # type: ignore
        evaluator_results_text = self._render_evaluator_results_text(
            results=results, main_llm=main_llm, comparison_llm=comparison_llm  # type: ignore
        )

        return llm_results_text, evaluator_results_text

    def _render_llm_results_text(self, results: list[Result], main_llm: str, comparison_llm: str) -> str:
        main_llm_preferred = sum(1 for result in results if result.result == 1)
        comparison_llm_preferred = sum(1 for result in results if result.result == 2)
        equally_good = sum(1 for result in results if result.result == 3)
        abstained_main_llm = sum(1 for result in results if result.result == 4 and result.llm_name == main_llm)
        abstained_comparison_llm = sum(
            1 for result in results if result.result == 4 and result.llm_name == comparison_llm
        )
        abstained = abstained_main_llm + abstained_comparison_llm
        total = len(results)

        if main_llm_preferred > comparison_llm_preferred:
            winner = main_llm
        elif comparison_llm_preferred > main_llm_preferred:
            winner = comparison_llm  # type: ignore
        else:
            winner = "DRAW"

        draw_text = "[yellow]No overall winner[/yellow]\n\n"
        winner_text = f"[green]Winner overall: {winner}[/green]" if winner != "DRAW" else draw_text
        winner_preferred = main_llm_preferred if winner == main_llm else comparison_llm_preferred
        total_with_preference = main_llm_preferred + comparison_llm_preferred
        if winner != "DRAW":
            winner_text += f"\n{winner} was preferred: {winner_preferred} out of {total_with_preference} "
            winner_text += f"({winner_preferred / total_with_preference:.2%})\n\n"

        llm_results_text = winner_text
        llm_results_text += f"{main_llm} preferred: {main_llm_preferred} ({main_llm_preferred / total:.2%})\n\n"
        llm_results_text += (
            f"{comparison_llm} preferred: {comparison_llm_preferred} ({comparison_llm_preferred / total:.2%})\n\n"
        )
        llm_results_text += f"Equally good: {equally_good} ({equally_good / total:.2%})\n\n"
        llm_results_text += f"Abstained: {abstained} ({abstained / total:.2%})\n\n"
        if abstained:
            llm_results_text += f"({main_llm} abstained: {abstained_main_llm}, {comparison_llm} abstained: {abstained_comparison_llm})\n\n"

        llm_results_text += f"Total: {total}\n"
        return llm_results_text

    def _render_evaluator_results_text(self, results: list[Result], main_llm: str, comparison_llm: str) -> str:
        evaluator_results_text = ""
        evaluators = {result.evaluator_name for result in results}
        for evaluator in evaluators:
            evaluator_results_text += f"[blue]{evaluator} results:[/blue]\n"
            evaluator_results = [result for result in results if result.evaluator_name == evaluator]
            main_llm_preferred = sum(1 for result in evaluator_results if result.result == 1)
            comparison_llm_preferred = sum(1 for result in evaluator_results if result.result == 2)
            equally_good = sum(1 for result in evaluator_results if result.result == 3)
            abstained_main_llm = sum(
                1 for result in evaluator_results if result.result == 4 and result.llm_name == main_llm
            )
            abstained_comparison_llm = sum(
                1 for result in evaluator_results if result.result == 4 and result.llm_name == comparison_llm
            )
            abstained = abstained_main_llm + abstained_comparison_llm
            total = len(evaluator_results)

            if main_llm_preferred > comparison_llm_preferred:
                winner = main_llm
            elif comparison_llm_preferred > main_llm_preferred:
                winner = comparison_llm
            else:
                winner = "DRAW"

            draw_text = "[yellow]No overall winner[/yellow]\n\n"
            winner_text = f"[green]Winner overall: {winner}[/green]" if winner != "DRAW" else draw_text
            winner_preferred = main_llm_preferred if winner == main_llm else comparison_llm_preferred
            total_with_preference = main_llm_preferred + comparison_llm_preferred
            if winner != "DRAW":
                winner_text += f"\n{winner} was preferred: {winner_preferred} out of {total_with_preference} "
                winner_text += f"({winner_preferred / total_with_preference:.2%})\n\n"

            evaluator_results_text += winner_text
            evaluator_results_text += f"{main_llm} preferred: {main_llm_preferred} ({main_llm_preferred / total:.2%})\n"
            evaluator_results_text += (
                f"{comparison_llm} preferred: {comparison_llm_preferred} ({comparison_llm_preferred / total:.2%})\n"
            )
            evaluator_results_text += f"Equally good: {equally_good} ({equally_good / total:.2%})\n"
            evaluator_results_text += f"Abstained: {abstained} ({abstained / total:.2%})\n"
            if abstained:
                evaluator_results_text += f"({main_llm} abstained: {abstained_main_llm}, {comparison_llm} abstained: {abstained_comparison_llm})\n"

            evaluator_results_text += f"Total: {total}\n\n"
        return evaluator_results_text

    def _build_cli_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="overscan", size=1),
            Layout(name="title", size=5),
            Layout(name="progress", size=3),
            Layout(name="input_text"),
            Layout(name="content", ratio=2),
            Layout(name="instructions", size=5),
        )
        layout["content"].split_row(Layout(name="first_example"), Layout(name="second_example"))

        layout["title"].update(
            Panel(
                Align.center(f"Comparison Evaluation: {self.name}", vertical="middle"),
                title="Title",
                border_style="blue",
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
        )
        layout["results"].split_row(Layout(name="llm_results"), Layout(name="evaluator_results"))
        return layout
