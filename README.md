# successat

Reusable benchmarking utilities for large language models (LLMs) with a
focus on testable, provider-agnostic client integrations.

## Project overview

The project packages reusable LLM client abstractions under
`successat.llm.clients`. Two concrete clients are included:

* `OpenAIClient` – connects to OpenAI's API using the official SDK.
* `OpenRouterClient` – connects to the OpenRouter API while reusing the same
  SDK.

Both clients share a common base class, default to the
`gpt-5-nano` model, and identify themselves with the application name
`successat`. The module is designed so downstream benchmarks can swap providers
without changing business logic.

## Prerequisites

* [uv](https://docs.astral.sh/uv/) (package and project manager for Python)
* Python 3.12 (managed automatically when using `uv`)

## Environment configuration

Credentials are read from environment variables and the repository ships with
an `.env` file that defines the keys:

* `OPENAI_API_KEY`
* `OPENROUTER_API_KEY`

Do **not** commit secret keys to version control. Populate the `.env` file
locally or export the variables in your shell. The clients also expose
`from_env()` helpers to read these values at runtime.

## Installing dependencies

Run the following command once to install (or update) the project dependencies
defined in `pyproject.toml` and `uv.lock`:

```bash
uv sync
```

This will create a local virtual environment under `.venv/` if one does not
already exist.

## Running the test suite

Unit and integration tests are managed with `pytest`. By default the suite will
exercise real OpenAI and OpenRouter endpoints when valid credentials are
available.

```bash
uv run --env-file .env pytest
```

The `--env-file` flag ensures the `.env` file is loaded, keeping the test
environment aligned with production usage when developing locally. Continuous
integration loads provider credentials from GitHub secrets and runs `uv run
pytest` directly without relying on a `.env` file.

Integration tests that call the external APIs are marked with the `integtest`
pytest marker. To focus on unit tests locally you can exclude them:

```bash
uv run --env-file .env pytest -m "not integtest"
```

If a provider-specific model should be used instead of the default
`gpt-5-nano`, set one of the following environment variables before running the
tests:

* `OPENAI_INTEG_MODEL`
* `OPENROUTER_INTEG_MODEL`

Tests are skipped automatically when a required API key is not available in the
environment.

## Example usage

```python
from successat.benchmarks import run_benchmark
from successat.llm.clients import OpenAIClient, OpenRouterClient

# Instantiate using environment variables defined in .env
openai_client = OpenAIClient.from_env()
router_client = OpenRouterClient.from_env()

print(openai_client.chat("Explain reusable LLM benchmarking."))
print(router_client.chat("Explain reusable LLM benchmarking."))

# Execute a reusable benchmark example (here the GSM8K math task)
result = run_benchmark(openai_client, "gsm8k", identifier=0, split="test")

print(
    "Model:", result.model,
    "Prompt:", result.prompt,
    "Expected:", result.metadata["expected"],
    "Response:", result.response_text,
    "Correct:", result.correct,
)
```

Remember that invoking the clients requires valid API keys; the example is
intended for environments where the necessary credentials are available. The
`run_benchmark` helper returns a `BenchmarkResult` containing the raw response
object, extracted text, correctness flag, and metadata describing the
evaluation.

## Command line usage

The package also installs a `successat` console script that provides a thin
wrapper around the benchmark runner. This is useful for quickly sanity-checking
models without writing a Python harness. For example:

```bash
uv run successat --list-benchmarks
uv run successat --benchmark gsm8k --client openai --split test --identifier 0 --param temperature=0.2
```

Pass `--client-option` arguments to forward extra keyword arguments to the
client constructor and `--param` to control the chat completion call (for
example, `temperature`, `max_tokens`, or provider-specific toggles). Credentials
are read from environment variables by default, but an explicit `--api-key`
value can be supplied for ad-hoc testing.

## Available benchmarks

Each benchmark pulls real evaluation data from the Hugging Face Hub. The first
execution will download the corresponding dataset artefacts to the local cache.

* **GSM8K** – loads the `gsm8k` dataset (`main` configuration) and supports the
  `train` and `test` splits.
* **MMLU** – uses the `cais/mmlu` dataset with the `all` configuration. Supported
  splits include `train` (mapped to `auxiliary_train`), `dev`, `validation`, and
  `test`.
* **HumanEval** – evaluates generated code against the official
  `openai_humaneval` test harness.
* **TriviaQA / ARC-Easy** – combines the
  `TimoImhof/Splits_Subset_TriviaQa` subset for free-form answers and the
  `ai2_arc` `ARC-Easy` configuration for multiple choice questions. Common
  aliases (`train`, `validation`, `test`) return a blended set, while
  `triviaqa:<split>` or `arc_easy:<split>` target individual sources.

Use the `identifier` argument to select a specific example by index or by its
unique identifier:

```python
result = run_benchmark(openai_client, "triviaqa", split="arc_easy:test", identifier=0)
print(result.metadata["evaluation_details"])
```
