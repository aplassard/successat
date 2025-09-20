"""Unit tests covering benchmark timing enrichment."""

from __future__ import annotations

import pytest

import successat.benchmarks.base as base_module
from successat.benchmarks.base import Benchmark, BenchmarkExample


class _TimingBenchmark(Benchmark):
    """Minimal benchmark used to exercise timing behaviour."""

    name = "timing-demo"

    def __init__(self, client, *, metadata: dict[str, object] | None = None) -> None:  # type: ignore[override]
        super().__init__(client)
        self._metadata = metadata or {}

    def examples_for_split(self, split: str):  # type: ignore[override]
        return [
            BenchmarkExample(
                id="example-0",
                prompt="prompt",
                target="response",
                metadata=dict(self._metadata),
            )
        ]


def test_run_populates_timing_metadata(monkeypatch: pytest.MonkeyPatch, fake_llm_client) -> None:
    """Benchmarks should add timing metadata when it is missing."""

    times = iter([10.0, 10.5])

    def fake_perf_counter() -> float:
        return next(times)

    monkeypatch.setattr(base_module, "perf_counter", fake_perf_counter)

    client = fake_llm_client(["response"])
    benchmark = _TimingBenchmark(client)

    result = benchmark.run()

    assert result.time_to_first_token == pytest.approx(0.5)
    assert result.total_time == pytest.approx(0.5)
    assert result.metadata["time_to_first_token"] == pytest.approx(0.5)
    assert result.metadata["total_time"] == pytest.approx(0.5)


def test_run_preserves_existing_timing_metadata(monkeypatch: pytest.MonkeyPatch, fake_llm_client) -> None:
    """Pre-existing timing metadata should be preferred over measured durations."""

    times = iter([100.0, 101.0])

    def fake_perf_counter() -> float:
        return next(times)

    monkeypatch.setattr(base_module, "perf_counter", fake_perf_counter)

    metadata = {"time_to_first_token": "1.2", "total_time": 3.4}
    client = fake_llm_client(["response"])
    benchmark = _TimingBenchmark(client, metadata=metadata)

    result = benchmark.run()

    assert result.time_to_first_token == pytest.approx(1.2)
    assert result.total_time == pytest.approx(3.4)
    # Existing metadata values should remain untouched.
    assert result.metadata["time_to_first_token"] == "1.2"
    assert result.metadata["total_time"] == 3.4

