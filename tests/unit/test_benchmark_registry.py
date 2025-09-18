"""Unit tests covering the benchmark registry utilities."""

from __future__ import annotations

import pytest

from successat.benchmarks import (
    GSM8KBenchmark,
    HumanEvalBenchmark,
    HumanEvalPlusBenchmark,
    MMLUBenchmark,
    TriviaQABenchmark,
    benchmark_registry,
    register_benchmarks,
    run_benchmark,
)
from successat.benchmarks.base import BenchmarkRegistry


def test_registry_contains_default_benchmarks() -> None:
    names = set(benchmark_registry.names())
    assert {"gsm8k", "mmlu", "humaneval", "humaneval+", "triviaqa"}.issubset(names)


def test_register_benchmarks_rejects_duplicates() -> None:
    registry = BenchmarkRegistry()
    register_benchmarks(registry, (GSM8KBenchmark,))

    with pytest.raises(ValueError):
        register_benchmarks(registry, (GSM8KBenchmark,))


def test_run_benchmark_unknown_name(fake_llm_client) -> None:
    client = fake_llm_client("placeholder response")

    with pytest.raises(KeyError):
        run_benchmark(client, "unknown-benchmark")

