"""Unit tests for the TriviaQA benchmark configuration."""

from __future__ import annotations

from typing import Any, Iterable

import pytest

from successat.benchmarks.triviaqa import ARC_DATASET, TriviaQABenchmark


class _DummyClient:
    """Minimal client stub providing a model attribute."""

    model = "dummy-model"


def test_triviaqa_lists_arc_challenge_split() -> None:
    """Ensure arc-challenge variants are exposed via available_splits."""

    benchmark = TriviaQABenchmark(_DummyClient())
    splits = benchmark.available_splits()

    assert "arc_challenge:validation" in splits


def test_triviaqa_loads_arc_challenge_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arc-Challenge splits should load using the correct dataset configuration."""

    arc_rows: Iterable[dict[str, Any]] = [
        {
            "id": "challenge-1",
            "question": "Which option is correct?",
            "choices": {
                "label": ["A", "B"],
                "text": ["Option A", "Option B"],
            },
            "answerKey": "B",
        }
    ]

    def fake_load_dataset(name: str, config: str | None = None, *, split: str | None = None):
        if name == ARC_DATASET and config == "ARC-Challenge":
            assert split == "validation"
            return arc_rows
        raise AssertionError(f"Unexpected dataset request: {name!r}, {config!r}")

    monkeypatch.setattr("successat.benchmarks.triviaqa.load_dataset", fake_load_dataset)

    benchmark = TriviaQABenchmark(_DummyClient())
    examples = benchmark.examples_for_split("arc_challenge:validation")

    assert len(examples) == 1
    example = examples[0]
    assert example.metadata["dataset"] == "arc_challenge"
    assert example.id.startswith("arc-challenge-validation-")
    assert example.target == "B"

    correct, details = benchmark.is_correct(example, "The answer is B", None)
    assert correct is True
    assert details["predicted_letter"] == "B"
