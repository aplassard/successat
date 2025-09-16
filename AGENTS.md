# successat contributor notes

## Tooling
- Use [uv](https://docs.astral.sh/uv/) for dependency management and commands.
- Install or update dependencies with `uv sync`.
- Run tests with `uv run --env-file .env pytest` so the local `.env` values are
  loaded consistently.

## Environment variables
- The repository contains an `.env` file with placeholders for required API
  keys. Do not commit real credentials and do not modify this file in pull
  requests.
- Application code should access credentials through environment variables or
  the provided `from_env()` helpers in `successat.llm.clients`.

## Testing guidance
- Default unit tests can rely on mocked transports, but integration tests under
  `tests/integration/test_live_llm_clients.py` call the real OpenAI and
  OpenRouter APIs. They are marked with the `integtest` marker and are skipped
  automatically when credentials are unavailable.
- Run all checks with `uv run --env-file .env pytest`. To omit external calls in
  local development, use `uv run --env-file .env pytest -m "not integtest"`.
- GitHub Actions loads provider keys from repository secrets and automatically
  falls back to `uv run pytest` when a `.env` file is not present.
- Close instantiated `OpenAI` clients in tests to avoid resource warnings.

Following these practices keeps local development, CI, and production
configuration aligned.
