# LLM Migration CLI

The purpose of this library is to provide a simple solution to compare the performance of two LLM models/providers on a given task.

It is not intended to be a comprehensive solution, but rather the simplest and fastest way to compare models and reach a decision on which is preferred. If found useful, more features can be added over time.

## Workflow

- `pip install git+https://github.com/michael-tribe/llm-migration-cli.git`
- log examples from inside your code using `ExampleLogger`
- run your application with the model(s) you want to compare
- run `llm_migrate evaluate <evaluation_type> <path_to_prompts_dir>`


## Logging Examples

- log examples from inside your code using `ExampleLogger`
```python
from llm_migrate import ExampleLogger


example_logger = ExampleLogger()


def process_prompt(template: str, variables: dict[str, Any], model: str) -> str:
    filled_prompt = template.format(**variables)
    response = openai.Completion.create(model=model, prompt=filled_prompt)
    processed = process_result(response.choices[0].text.strip())

    # insert this line
    example_logger.log(
        llm_name=model,
        template=template,
        template_params=variables,
        output_text=processed,
        inference_params={"temperature": temperature, "max_tokens": max_tokens},
    )

    return processed
```
- only one example will be logged per input text per model to avoid duplicates


## Evaluating Examples

- the two types of evaluations currently implemented are `rating` and `comparison`
- `rating` is used to evaluate the quality of the output, but can configured with a different question
- `comparison` is used to compare the output of two models on the same prompt
  - examples by different models are matched using a hash of the input text
- evaluations can be run using the CLI, either performed by a human or by an LLM
  - the idea is to use LLM evaluation for large numbers of samples to get an early signal, and human evaluation on a subset for a more accurate result
  - filtering / subsetting examples is TODO
- after going through all examples you will be presented with the results
  - result presentation has room for improvement
- if you want to see the current results of an evaluation, you can run `llm_migrate results <evaluation_type> <path_to_prompts_dir>`


## Usage

- for LLM evaluation, the values from `secrets.env.sample` should be filled in and the file renamed to `secrets.env`

```bash
# run without args for help
llm_migrate

# list available objects
llm_migrate list {evals, llms, prompts}

# run evaluation by human
llm_migrate evaluate <evaluation_type> <path_to_prompts_dir>

# run evaluation by llm
llm_migrate evaluate <evaluation_type> <path_to_prompts_dir> --llm <llm_model_name>

# view results
llm_migrate results <evaluation_type> <path_to_prompts_dir>
```

- settings are defined in `src/settings.py` and can be overridden by environment variables in `vars.env` or `secrets.env`


## Design Goals

- simple, extensible, and easy to use
- labeling interface: currently uses a simple CLI, can be extended to a more advanced TUI or web interface for better UX


## Development

- environment setup is done with Conda and has been captured in `scripts/setup.sh`
- `pip` requirements are captured in `requirements.txt` instead of `environment.yml` for better compatibility with other tools
- pre-commit hooks are used heavily to ensure code quality

```bash
# environment setup
./scripts/setup.sh

# activate environment
conda activate llm-migration-cli
# or
source ./scripts/activate.sh

# install pre-commit hooks
pre-commit install -t pre-commit -t pre-push -t commit-msg --overwrite

# run tests
./scripts/test.sh
```
