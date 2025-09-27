import asyncio
import threading
import time
from typing import List

from successat.benchmarks import (
    Benchmark,
    BenchmarkExample,
    BenchmarkRegistry,
    BenchmarkRunSpec,
    BenchmarkRunner,
)


class SleepyBenchmark(Benchmark):
    """Synthetic benchmark used to exercise the runner orchestration."""

    name = "sleepy"

    def __init__(self, client) -> None:
        super().__init__(client)
        self._examples: List[BenchmarkExample] = [
            BenchmarkExample(
                id=str(index),
                prompt=f"Solve for x when x = 42 (example {index})",
                target="42",
                metadata={"index": index},
            )
            for index in range(10)
        ]

    def examples_for_split(self, split: str):
        return self._examples

    def is_correct(self, example: BenchmarkExample, response_text: str, response: object):
        return response_text.strip() == "42", {"received": response_text.strip()}


class CountingLLMClient:
    """Client double that records concurrent invocations."""

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.model = "counting"
        self._lock = threading.Lock()
        self._inflight = 0
        self.max_inflight = 0

    def chat_completion(self, prompt: str, **kwargs):
        with self._lock:
            self._inflight += 1
            self.max_inflight = max(self.max_inflight, self._inflight)
        try:
            time.sleep(self.delay)
            return {"choices": [{"message": {"content": "42"}}]}
        finally:
            with self._lock:
                self._inflight -= 1


def _build_registry() -> BenchmarkRegistry:
    registry = BenchmarkRegistry()
    registry.register(SleepyBenchmark)
    return registry


def test_runner_honors_max_concurrency() -> None:
    registry = _build_registry()
    client = CountingLLMClient(delay=0.02)
    specs = [
        BenchmarkRunSpec("sleepy", client=client, identifier=index, split="test")
        for index in range(6)
    ]

    runner = BenchmarkRunner(registry=registry, max_concurrency=2)
    results = runner.run_all(specs)

    assert [result.example_id for result in results] == [str(i) for i in range(6)]
    assert client.max_inflight <= 2
    assert all(result.correct for result in results)


def test_runner_invokes_synchronous_callback(fake_llm_client) -> None:
    registry = _build_registry()
    client = fake_llm_client(["42", "42"])
    specs = [
        BenchmarkRunSpec("sleepy", client=client, identifier=index, split="test")
        for index in range(2)
    ]

    observed = []

    def _callback(spec: BenchmarkRunSpec, result):
        details = result.metadata.get("evaluation_details", {})
        observed.append((spec.identifier, details.get("received")))

    runner = BenchmarkRunner(registry=registry, max_concurrency=1)
    results = runner.run_all(specs, result_callback=_callback)

    assert len(results) == 2
    assert observed == [(0, "42"), (1, "42")]


def test_runner_invokes_async_callback(fake_llm_client) -> None:
    registry = _build_registry()
    client = fake_llm_client(["42"])
    spec = BenchmarkRunSpec("sleepy", client=client, identifier=0, split="test")

    received = []

    async def _callback(spec: BenchmarkRunSpec, result):
        await asyncio.sleep(0)
        received.append((spec.identifier, result.example_id, result.correct))

    runner = BenchmarkRunner(registry=registry)
    results = runner.run_all([spec], result_callback=_callback)

    assert len(results) == 1
    assert received == [(0, "0", True)]
