"""Benchmark registry and convenience helpers."""

from __future__ import annotations

from typing import Iterable, Type

from .base import (
    Benchmark,
    BenchmarkExample,
    BenchmarkRegistry,
    BenchmarkResult,
    SupportsChatCompletion,
)
from .gsm8k import GSM8KBenchmark
from .humaneval import HumanEvalBenchmark, HumanEvalPlusBenchmark
from .livebench import (
    LiveBenchCodingBenchmark,
    LiveBenchMathBenchmark,
    LiveBenchReasoningBenchmark,
)
from .mmlu import MMLUBenchmark
from .triviaqa import TriviaQABenchmark

__all__ = [
    "Benchmark",
    "BenchmarkExample",
    "BenchmarkRegistry",
    "BenchmarkResult",
    "GSM8KBenchmark",
    "HumanEvalBenchmark",
    "HumanEvalPlusBenchmark",
    "MMLUBenchmark",
    "LiveBenchCodingBenchmark",
    "LiveBenchMathBenchmark",
    "LiveBenchReasoningBenchmark",
    "TriviaQABenchmark",
    "benchmark_registry",
    "register_benchmarks",
    "run_benchmark",
]


def register_benchmarks(registry: BenchmarkRegistry, benchmarks: Iterable[Type[Benchmark]]) -> None:
    """Register the provided benchmark classes with the registry."""

    for benchmark_cls in benchmarks:
        registry.register(benchmark_cls)


benchmark_registry = BenchmarkRegistry()
register_benchmarks(
    benchmark_registry,
    (
        GSM8KBenchmark,
        MMLUBenchmark,
        HumanEvalBenchmark,
        HumanEvalPlusBenchmark,
        LiveBenchCodingBenchmark,
        LiveBenchReasoningBenchmark,
        LiveBenchMathBenchmark,
        TriviaQABenchmark,
    ),
)


def run_benchmark(
    client: SupportsChatCompletion,
    benchmark_name: str,
    *,
    identifier: int | str | None = None,
    split: str | None = None,
    **kwargs: object,
) -> BenchmarkResult:
    """Execute the named benchmark using the provided client."""

    benchmark_cls = benchmark_registry.get(benchmark_name)
    runner = benchmark_cls(client)
    return runner.run(identifier=identifier, split=split, **kwargs)

