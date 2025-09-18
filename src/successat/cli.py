"""Command-line interface for executing successat benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Mapping, Sequence

from .benchmarks import Benchmark, BenchmarkResult, benchmark_registry, run_benchmark
from .llm import BaseLLMClient, OpenAIClient, OpenRouterClient


class CLIError(RuntimeError):
    """Raised when the CLI encounters a recoverable error."""

    def __init__(self, message: str, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code


CLIENT_FACTORIES: Mapping[str, type[BaseLLMClient]] = {
    "openai": OpenAIClient,
    "openrouter": OpenRouterClient,
}


def _build_parser() -> argparse.ArgumentParser:
    """Create and return the top-level argument parser."""

    parser = argparse.ArgumentParser(description="Run successat benchmarks from the CLI.")
    parser.add_argument(
        "--benchmark",
        "-b",
        help="Name of the benchmark to execute (e.g. gsm8k, mmlu, humaneval, humaneval+).",
    )
    parser.add_argument(
        "--client",
        choices=sorted(CLIENT_FACTORIES),
        default="openai",
        help="LLM client implementation to use.",
    )
    parser.add_argument(
        "--model",
        help="Override the default model configured for the selected client.",
    )
    parser.add_argument(
        "--app-name",
        help="Custom application name to include in client requests.",
    )
    parser.add_argument(
        "--api-key",
        help="API key for the chosen client. If omitted, credentials are loaded from the environment.",
    )
    parser.add_argument(
        "--split",
        help="Benchmark split to evaluate (defaults to the benchmark's preferred split).",
    )
    parser.add_argument(
        "--identifier",
        help=(
            "Identifier of the example to run. Provide an integer index or the example's explicit identifier."
        ),
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional parameters forwarded to the chat completion call (e.g. temperature=0.2).",
    )
    parser.add_argument(
        "--client-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional keyword arguments passed to the client constructor.",
    )
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List the available benchmark names and exit.",
    )
    parser.add_argument(
        "--benchmark-details",
        metavar="BENCHMARK",
        help="Show the available splits and example counts for the specified benchmark.",
    )
    return parser


def _parse_identifier(raw_identifier: str | None) -> int | str | None:
    """Convert the identifier argument into the appropriate type."""

    if raw_identifier is None:
        return None

    try:
        return int(raw_identifier)
    except ValueError:
        return raw_identifier


def _coerce_value(value: str) -> Any:
    """Attempt to coerce a string value into a richer Python type."""

    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue

    return value


def _parse_key_value_pairs(pairs: Sequence[str]) -> dict[str, Any]:
    """Convert KEY=VALUE strings into a dictionary."""

    parsed: dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            msg = f"Invalid parameter '{item}'. Expected the form KEY=VALUE."
            raise CLIError(msg)
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            msg = f"Invalid parameter '{item}'. Key may not be empty."
            raise CLIError(msg)
        parsed[key] = _coerce_value(raw_value.strip())
    return parsed


def _create_client(args: argparse.Namespace, client_kwargs: Mapping[str, Any]) -> BaseLLMClient:
    """Instantiate the requested client based on parsed arguments."""

    factory = CLIENT_FACTORIES[args.client]
    kwargs = dict(client_kwargs)
    if not kwargs:
        kwargs = {}

    try:
        if args.api_key:
            return factory(
                api_key=args.api_key,
                model=args.model,
                app_name=args.app_name,
                client_kwargs=kwargs or None,
            )
        return factory.from_env(
            model=args.model,
            app_name=args.app_name,
            client_kwargs=kwargs or None,
        )
    except EnvironmentError as exc:  # pragma: no cover - exercised in integration scenarios
        msg = str(exc)
        raise CLIError(msg, exit_code=1) from exc
    except ValueError as exc:
        msg = str(exc)
        raise CLIError(msg, exit_code=1) from exc


def _close_client(client: BaseLLMClient) -> None:
    """Close the underlying OpenAI client if available."""

    inner = getattr(client, "client", None)
    close = getattr(inner, "close", None)
    if callable(close):
        close()


class _IntrospectionClient:
    """Minimal client used when introspecting benchmark metadata."""

    model = "introspection"

    def chat_completion(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - defensive
        msg = "Introspection client cannot execute chat completions."
        raise RuntimeError(msg)


def _print_benchmark_details(benchmark_name: str) -> None:
    """Display the available splits and example counts for a benchmark."""

    try:
        benchmark_cls: type[Benchmark] = benchmark_registry.get(benchmark_name)
    except KeyError as exc:
        msg = str(exc)
        raise CLIError(msg, exit_code=1) from exc

    benchmark = benchmark_cls(_IntrospectionClient())

    try:
        split_names = list(benchmark.available_splits())
    except Exception as exc:  # pragma: no cover - benchmark misconfiguration
        msg = f"Failed to determine splits for benchmark '{benchmark.name}': {exc}"
        raise CLIError(msg, exit_code=1) from exc

    print(f"Benchmark: {benchmark.name}")
    print("Splits:")

    if not split_names:
        print("  (none)")
        return

    for split in split_names:
        try:
            examples = benchmark.examples_for_split(split)
        except Exception as exc:  # pragma: no cover - dataset access failures
            msg = f"Failed to load split '{split}' for benchmark '{benchmark.name}': {exc}"
            raise CLIError(msg, exit_code=1) from exc

        count = len(examples)
        noun = "example" if count == 1 else "examples"
        default_marker = " (default)" if split == benchmark.default_split else ""
        print(f"  - {split}{default_marker}: {count} {noun}")


def _format_result(result: BenchmarkResult) -> str:
    """Return a human-readable representation of the benchmark result."""

    metadata = json.dumps(result.metadata, indent=2, default=str) if result.metadata else "{}"
    prompt_preview = result.prompt.strip()
    formatted_prompt = prompt_preview if prompt_preview else "(empty)"
    response_text = result.response_text.strip()
    formatted_response = response_text if response_text else "(empty)"

    output_lines = [
        f"Benchmark: {result.benchmark}",
        f"Model: {result.model}",
        f"Split: {result.split}",
        f"Example ID: {result.example_id}",
        f"Correct: {result.correct}",
        "Prompt:",
        formatted_prompt,
        "Response:",
        formatted_response,
        "Metadata:",
        metadata,
    ]
    return "\n".join(output_lines)


def _run(args: argparse.Namespace) -> int:
    """Execute the CLI using the provided arguments."""

    if args.list_benchmarks and args.benchmark_details:
        raise CLIError("--list-benchmarks cannot be combined with --benchmark-details.")

    if args.list_benchmarks:
        for name in benchmark_registry.names():
            print(name)
        return 0

    if args.benchmark_details:
        _print_benchmark_details(args.benchmark_details)
        return 0

    if not args.benchmark:
        raise CLIError("A benchmark name is required. Provide --benchmark <name>.")

    identifier = _parse_identifier(args.identifier)
    chat_parameters = _parse_key_value_pairs(args.param)
    client_options = _parse_key_value_pairs(args.client_option)

    client = _create_client(args, client_options)
    try:
        result = run_benchmark(
            client,
            args.benchmark,
            identifier=identifier,
            split=args.split,
            **chat_parameters,
        )
    except Exception as exc:  # pragma: no cover - exercised in integration scenarios
        raise CLIError(str(exc), exit_code=1) from exc
    finally:
        _close_client(client)

    print(_format_result(result))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point invoked by the console script."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except CLIError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":  # pragma: no cover - manual execution
    sys.exit(main())
