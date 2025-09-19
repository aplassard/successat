"""Tests for the LiveBench coding benchmark implementation."""

from __future__ import annotations

import base64
import json
import pickle
import zlib
from datetime import datetime
from typing import Any, Iterable, List

import pytest

from successat.benchmarks.livebench import (
    LiveBenchCodingBenchmark,
    LiveBenchMathBenchmark,
    LiveBenchReasoningBenchmark,
)


class _DummyClient:
    model = "dummy"


def _encode_private_tests(tests: Iterable[dict[str, Any]]) -> str:
    json_blob = json.dumps(list(tests))
    compressed = zlib.compress(pickle.dumps(json_blob))
    return base64.b64encode(compressed).decode("ascii")


def _functional_row() -> dict[str, Any]:
    return {
        "question_id": "func-1",
        "category": "coding",
        "turns": [
            "Context about the task.",
            "Please implement the function described in the starter code.",
        ],
        "question_title": "add_numbers",
        "public_test_cases": json.dumps(
            [
                {"input": "[1, 2]", "output": "3", "testtype": "functional"},
            ]
        ),
        "private_test_cases": _encode_private_tests(
            [
                {"input": "[2, 3]", "output": "5", "testtype": "functional"},
            ]
        ),
        "original_json": {
            "starter_code": (
                "class Solution:\n"
                "    def add(self, a: int, b: int) -> int:\n"
                "        pass\n"
            ),
            "question_content": "Add two integers.",
            "metadata": json.dumps({"func_name": "add"}),
            "platform": "custom",
            "question_id": "func-1",
            "contest_id": "contest",
            "contest_date": datetime(2024, 7, 1),
            "starter_code": (
                "class Solution:\n"
                "    def add(self, a: int, b: int) -> int:\n"
                "        pass\n"
            ),
            "difficulty": "easy",
        },
        "release_date": datetime(2024, 6, 15),
        "citation": "LiveBench", 
        "task": "LCB_generation",
        "livebench_release_date": datetime(2024, 7, 1),
        "livebench_removal_date": None,
        "remainder": "",
        "solution": "",
        "partial_solution": "",
    }


def _stdin_row() -> dict[str, Any]:
    return {
        "question_id": "stdin-1",
        "category": "coding",
        "turns": ["Read an integer and output its double."],
        "question_title": "double_input",
        "public_test_cases": json.dumps(
            [
                {"input": "3\n", "output": "6\n", "testtype": "stdin"},
                {"input": "5\n", "output": "10\n", "testtype": "stdin"},
            ]
        ),
        "private_test_cases": "",
        "original_json": {
            "starter_code": "",
            "question_content": "Read an integer from stdin and print double.",
            "metadata": "{}",
            "platform": "custom",
            "question_id": "stdin-1",
            "contest_id": "contest",
            "contest_date": datetime(2024, 6, 1),
            "starter_code": "",
            "difficulty": "easy",
        },
        "release_date": datetime(2024, 5, 20),
        "citation": "LiveBench",
        "task": "LCB_generation",
        "livebench_release_date": datetime(2024, 6, 1),
        "livebench_removal_date": None,
        "remainder": "",
        "solution": "",
        "partial_solution": "",
    }


def _reasoning_row() -> dict[str, Any]:
    return {
        "question_id": "reason-1",
        "category": "reasoning",
        "ground_truth": "1, filmmaking, police-officer, journalist",
        "turns": [
            (
                "Solve the zebra puzzle and report your answers inside a <solution> tag "
                "as position, hobby, job, other job."
            ),
        ],
        "task": "zebra_puzzle",
        "level": 15,
        "livebench_release_date": datetime(2024, 11, 25),
        "livebench_removal_date": None,
    }


def _math_row() -> dict[str, Any]:
    return {
        "question_id": "math-1",
        "category": "math",
        "ground_truth": "1,6,7",
        "turns": [
            (
                "Fill in the missing expressions for the provided solution. "
                "Return your final answer as 'Answer: <ids>'."
            ),
        ],
        "task": "olympiad",
        "subtask": "usamo",
        "year": "2023",
        "hardness": 1.0,
        "expressions": "<expression 1> ...",
        "livebench_release_date": datetime(2024, 8, 31),
        "livebench_removal_date": None,
    }


def test_livebench_lists_history_splits(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_functional_row(), _stdin_row()]

    def fake_dataset(*args: Any, **kwargs: Any) -> List[dict[str, Any]]:
        return rows

    monkeypatch.setattr("successat.benchmarks.livebench.load_dataset", fake_dataset)

    benchmark = LiveBenchCodingBenchmark(_DummyClient())

    splits = benchmark.available_splits()

    assert splits == ["latest", "history", "all", "release-2024-06", "release-2024-07"]

    latest_examples = benchmark.examples_for_split("latest")
    history_examples = benchmark.examples_for_split("history")

    assert len(latest_examples) == 1
    assert len(history_examples) == 1
    assert latest_examples[0].metadata["livebench_release_date"] == "2024-07-01T00:00:00"
    assert history_examples[0].metadata["livebench_release_date"] == "2024-06-01T00:00:00"


def test_livebench_functional_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_functional_row()]

    def fake_dataset(*args: Any, **kwargs: Any) -> List[dict[str, Any]]:
        return rows

    monkeypatch.setattr("successat.benchmarks.livebench.load_dataset", fake_dataset)

    benchmark = LiveBenchCodingBenchmark(_DummyClient())
    example = benchmark.examples_for_split("latest")[0]

    correct_code = """
    class Solution:
        def add(self, a: int, b: int) -> int:
            return a + b
    """

    correct, details = benchmark.is_correct(example, correct_code, None)
    assert correct is True
    assert details["tests_run"] == 2
    assert details["test_mode"] == "functional"

    wrong_code = """
    class Solution:
        def add(self, a: int, b: int) -> int:
            return a - b
    """

    correct, details = benchmark.is_correct(example, wrong_code, None)
    assert correct is False
    assert details["error"] == "candidate output did not match expected result"
    assert details["test_index"] == 0


def test_livebench_stdin_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_stdin_row()]

    def fake_dataset(*args: Any, **kwargs: Any) -> List[dict[str, Any]]:
        return rows

    monkeypatch.setattr("successat.benchmarks.livebench.load_dataset", fake_dataset)

    benchmark = LiveBenchCodingBenchmark(_DummyClient())
    example = benchmark.examples_for_split("latest")[0]

    script = """```python
import sys
value = int(sys.stdin.read().strip())
print(value * 2)
```
"""

    correct, details = benchmark.is_correct(example, script, None)
    assert correct is True
    assert details["tests_run"] == 2
    assert details["test_mode"] == "stdin"

    failing_script = "print('not double')\n"
    correct, details = benchmark.is_correct(example, failing_script, None)
    assert correct is False
    assert details["error"] == "script output did not match expected value"
    assert details["test_index"] == 0


def test_livebench_prefers_last_code_block(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_functional_row()]

    def fake_dataset(*args: Any, **kwargs: Any) -> List[dict[str, Any]]:
        return rows

    monkeypatch.setattr("successat.benchmarks.livebench.load_dataset", fake_dataset)

    benchmark = LiveBenchCodingBenchmark(_DummyClient())
    example = benchmark.examples_for_split("latest")[0]

    response = """Here is how you might approach the problem.

```python
This is pseudocode and will not compile.
```

Now for the final implementation:

```python
class Solution:
    def add(self, a: int, b: int) -> int:
        return a + b
```

Hope that helps!
"""

    correct, details = benchmark.is_correct(example, response, None)

    assert correct is True
    assert details["tests_run"] == 2
    assert details["test_mode"] == "functional"


def test_livebench_reasoning_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_reasoning_row()]

    def fake_dataset(*args: Any, **kwargs: Any) -> List[dict[str, Any]]:
        return rows

    monkeypatch.setattr("successat.benchmarks.livebench.load_dataset", fake_dataset)

    benchmark = LiveBenchReasoningBenchmark(_DummyClient())
    example = benchmark.examples_for_split("latest")[0]

    response = (
        "Thought process...\n"
        "<solution>Position 1, The filmmaking, police officer, journalist.</solution>"
    )

    correct, details = benchmark.is_correct(example, response, None)

    assert correct is True
    assert details["predicted_normalised"] == [
        "1",
        "filmmaking",
        "police officer",
        "journalist",
    ]


def test_livebench_reasoning_requires_solution(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_reasoning_row()]

    def fake_dataset(*args: Any, **kwargs: Any) -> List[dict[str, Any]]:
        return rows

    monkeypatch.setattr("successat.benchmarks.livebench.load_dataset", fake_dataset)

    benchmark = LiveBenchReasoningBenchmark(_DummyClient())
    example = benchmark.examples_for_split("latest")[0]

    correct, details = benchmark.is_correct(example, "Answer: 1, filmmaking", None)

    assert correct is False
    assert details["error"] == "no solution block found"


def test_livebench_math_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_math_row()]

    def fake_dataset(*args: Any, **kwargs: Any) -> List[dict[str, Any]]:
        return rows

    monkeypatch.setattr("successat.benchmarks.livebench.load_dataset", fake_dataset)

    benchmark = LiveBenchMathBenchmark(_DummyClient())
    example = benchmark.examples_for_split("latest")[0]

    response = "Detailed reasoning...\nAnswer: 1, 6, 7"

    correct, details = benchmark.is_correct(example, response, None)

    assert correct is True
    assert details["predicted_numbers"] == ["1", "6", "7"]

    wrong = "Answer: 1, 6, 8"
    correct, details = benchmark.is_correct(example, wrong, None)

    assert correct is False
    assert details["error"] == "answer mismatch"
    assert details["mismatch_index"] == 2


def test_livebench_math_parses_answer_on_new_line(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_math_row()]

    def fake_dataset(*args: Any, **kwargs: Any) -> List[dict[str, Any]]:
        return rows

    monkeypatch.setattr("successat.benchmarks.livebench.load_dataset", fake_dataset)

    benchmark = LiveBenchMathBenchmark(_DummyClient())
    example = benchmark.examples_for_split("latest")[0]

    response = "Reasoning...\nAnswer:\n1, 6, 7"

    correct, details = benchmark.is_correct(example, response, None)

    assert correct is True
    assert details["predicted_numbers"] == ["1", "6", "7"]

