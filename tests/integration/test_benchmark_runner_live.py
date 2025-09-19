import os
from collections import deque
from typing import Deque, Dict

import pytest

from successat.benchmarks import BenchmarkRunSpec, BenchmarkRunner
from successat.llm.clients import OpenAIClient

pytestmark = pytest.mark.integtest


@pytest.fixture(scope="module")
def live_openai_client() -> OpenAIClient:
    try:
        client = OpenAIClient.from_env()
    except EnvironmentError as exc:
        pytest.skip(str(exc))

    try:
        yield client
    finally:
        client.client.close()


@pytest.fixture(scope="module")
def benchmark_model(live_openai_client: OpenAIClient) -> str:
    for env_var in ("OPENAI_BENCHMARK_MODEL", "OPENAI_INTEG_MODEL"):
        override = os.getenv(env_var)
        if override:
            return override
    return live_openai_client.model


def test_runner_executes_multiple_live_benchmarks(
    live_openai_client: OpenAIClient, benchmark_model: str
) -> None:
    runner = BenchmarkRunner(max_concurrency=2)
    updates: Deque[Dict[str, object]] = deque()

    def _callback(spec: BenchmarkRunSpec, result) -> None:
        updates.append(
            {
                "benchmark": spec.benchmark,
                "example": result.example_id,
                "correct": result.correct,
                "udni_payload": result.metadata.get("evaluation_details"),
            }
        )

    specs = [
        BenchmarkRunSpec(
            "gsm8k",
            client=live_openai_client,
            identifier=0,
            split="test",
            parameters={"model": benchmark_model},
        ),
        BenchmarkRunSpec(
            "mmlu",
            client=live_openai_client,
            identifier=0,
            split="validation",
            parameters={"model": benchmark_model},
        ),
    ]

    results = runner.run_all(specs, result_callback=_callback)

    assert {result.benchmark for result in results} == {"gsm8k", "mmlu"}
    assert len(results) == len(specs)
    assert len(updates) == len(specs)

    for update in updates:
        assert update["benchmark"] in {"gsm8k", "mmlu"}
        assert isinstance(update["example"], str)
        assert update["udni_payload"] is not None
