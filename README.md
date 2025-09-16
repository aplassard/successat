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
integration loads provider credentials from GitHub secrets and falls back to
`uv run pytest` automatically when a `.env` file is not present.

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
from successat.llm.clients import OpenAIClient, OpenRouterClient

# Instantiate using environment variables defined in .env
openai_client = OpenAIClient.from_env()
router_client = OpenRouterClient.from_env()

print(openai_client.chat("Explain reusable LLM benchmarking."))
print(router_client.chat("Explain reusable LLM benchmarking."))
```

Remember that invoking the clients requires valid API keys; the example is
intended for environments where the necessary credentials are available.
