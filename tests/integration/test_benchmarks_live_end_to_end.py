"""Integration tests that execute the full benchmark pipeline against OpenAI."""

from __future__ import annotations

import os

import pytest

from successat.benchmarks import run_benchmark
from successat.llm.clients import OpenAIClient

pytestmark = pytest.mark.integtest


@pytest.fixture(scope="module")
def live_openai_client() -> OpenAIClient:
    """Instantiate the OpenAI client or skip if credentials are unavailable."""

    try:
        client = OpenAIClient.from_env()
    except EnvironmentError as exc:  # pragma: no cover - exercised when keys absent
        pytest.skip(str(exc))

    try:
        yield client
    finally:
        client.client.close()


@pytest.fixture(scope="module")
def benchmark_model(live_openai_client: OpenAIClient) -> str:
    """Resolve the model used for live benchmark execution."""

    for env_var in ("OPENAI_BENCHMARK_MODEL", "OPENAI_INTEG_MODEL"):
        override = os.getenv(env_var)
        if override:
            return override
    return live_openai_client.model


def _run(
    live_openai_client: OpenAIClient,
    benchmark: str,
    *,
    identifier: int | str | None = None,
    split: str | None = None,
    model: str,
    **kwargs,
):
    return run_benchmark(
        live_openai_client,
        benchmark,
        identifier=identifier,
        split=split,
        model=model,
        **kwargs,
    )


def test_gsm8k_benchmark_live_round_trip(
    live_openai_client: OpenAIClient, benchmark_model: str
) -> None:
    result = _run(
        live_openai_client,
        "gsm8k",
        identifier=0,
        split="test",
        model=benchmark_model,
    )

    assert result.benchmark == "gsm8k"
    assert "Final answer" in result.prompt
    assert result.metadata["question"]
    assert result.metadata["expected"]
    assert result.response_text.strip()
    assert "evaluation_details" in result.metadata


def test_mmlu_benchmark_live_round_trip(
    live_openai_client: OpenAIClient, benchmark_model: str
) -> None:
    result = _run(
        live_openai_client,
        "mmlu",
        identifier=0,
        split="validation",
        model=benchmark_model,
    )

    assert result.benchmark == "mmlu"
    assert result.metadata["subject"]
    assert result.metadata["evaluation_details"]["expected_letter"] in {"A", "B", "C", "D"}
    assert result.response_text.strip()


def test_humaneval_benchmark_live_round_trip(
    live_openai_client: OpenAIClient, benchmark_model: str
) -> None:
    result = _run(
        live_openai_client,
        "humaneval",
        identifier=0,
        split="test",
        model=benchmark_model,
    )

    entry_point = result.metadata["entry_point"]
    assert result.benchmark == "humaneval"
    assert entry_point
    assert f"def {entry_point}" in result.prompt
    assert result.metadata["canonical_solution"]
    assert result.response_text.strip()

    if not result.correct:
        details = result.metadata.get("evaluation_details")
        assert details is not None
        assert "error" in details


def test_triviaqa_short_answer_live_round_trip(
    live_openai_client: OpenAIClient, benchmark_model: str
) -> None:
    result = _run(
        live_openai_client,
        "triviaqa",
        identifier=0,
        split="triviaqa:split_1",
        model=benchmark_model,
    )

    evaluation_details = result.metadata["evaluation_details"]
    assert result.metadata["dataset"] == "triviaqa"
    assert result.metadata["question"]
    assert result.response_text.strip()
    assert "aliases" in evaluation_details


def test_arc_easy_live_round_trip(
    live_openai_client: OpenAIClient, benchmark_model: str
) -> None:
    result = _run(
        live_openai_client,
        "triviaqa",
        identifier=0,
        split="arc_easy:validation",
        model=benchmark_model,
    )

    evaluation_details = result.metadata["evaluation_details"]
    assert result.metadata["dataset"] == "arc_easy"
    assert result.metadata["question"]
    assert result.response_text.strip()
    assert "choices" in evaluation_details

