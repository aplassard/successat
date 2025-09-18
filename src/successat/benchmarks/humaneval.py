"""HumanEval benchmark implementation using the official dataset."""

from __future__ import annotations

import textwrap
from typing import Dict, List, Sequence

from datasets import load_dataset

from .base import Benchmark, BenchmarkExample


def _strip_code_fence(code: str) -> str:
    text = code.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.removeprefix("```")
        text = text.removesuffix("```")
        text = text.lstrip()
        if text.lower().startswith("python"):
            text = text[len("python") :].lstrip()
    return text


class HumanEvalBenchmark(Benchmark):
    """Evaluate code generation by running the official HumanEval unit tests."""

    name = "humaneval"
    description = "Python function synthesis with execution-based scoring."
    dataset_name = "openai_humaneval"
    default_split = "test"
    example_id_prefix = "humaneval"

    def __init__(self, client):
        super().__init__(client)
        self._examples: Dict[str, List[BenchmarkExample]] = {}

    def available_splits(self) -> Sequence[str]:
        return ["test"]

    def examples_for_split(self, split: str) -> Sequence[BenchmarkExample]:
        split_lower = split.lower()
        if split_lower != "test":
            msg = "HumanEval only exposes the 'test' split."
            raise ValueError(msg)

        if split_lower not in self._examples:
            dataset = load_dataset(self.dataset_name, split="test")
            examples = [self._convert_row(row, index) for index, row in enumerate(dataset)]
            self._examples[split_lower] = examples
        return self._examples[split_lower]

    def is_correct(self, example: BenchmarkExample, response_text: str, response: object):
        entry_point = str(example.target)
        candidate_code = _strip_code_fence(response_text)
        namespace: Dict[str, object] = {}

        try:
            exec(candidate_code, namespace)  # noqa: S102 - required to evaluate generated code
        except Exception as exc:  # pragma: no cover - error path validated in tests
            return False, {"error": f"candidate execution failed: {exc}"}

        candidate = namespace.get(entry_point)
        if not callable(candidate):
            return False, {"error": f"function '{entry_point}' not defined"}

        test_namespace: Dict[str, object] = {}
        test_code = example.metadata["test_code"]
        try:
            exec(test_code, test_namespace)  # noqa: S102 - dataset harness execution
        except Exception as exc:  # pragma: no cover - dataset should always execute
            return False, {"error": f"failed to load tests: {exc}"}

        check = test_namespace.get("check")
        if not callable(check):
            return False, {"error": "benchmark test harness missing 'check' function"}

        try:
            check(candidate)
        except AssertionError as exc:
            return False, {"error": f"assertion failed: {exc}"}
        except Exception as exc:  # pragma: no cover - dataset exercises edge cases
            return False, {"error": f"tests raised {exc.__class__.__name__}: {exc}"}

        return True, {}

    def _convert_row(self, row: dict, index: int) -> BenchmarkExample:
        task_id = row["task_id"]
        prompt = textwrap.dedent(row["prompt"]).strip()
        canonical_solution = row["canonical_solution"]
        entry_point = row["entry_point"]
        test_code = row["test"]

        metadata = {
            "task_id": task_id,
            "dataset_split": "test",
            "canonical_solution": canonical_solution,
            "entry_point": entry_point,
            "test_code": test_code,
        }

        system_prompt = (
            "You are a meticulous Python expert. Write only valid Python code that satisfies "
            "the specification. Do not include backticks or commentary."
        )

        return BenchmarkExample(
            id=f"{self.example_id_prefix}-{task_id}",
            prompt=prompt,
            target=entry_point,
            metadata=metadata,
            system_prompt=system_prompt,
        )


class HumanEvalPlusBenchmark(HumanEvalBenchmark):
    """Evaluate HumanEval+ problems using the EvalPlus augmented test suite."""

    name = "humaneval+"
    description = "Python synthesis with the HumanEval+ extended unit tests."
    dataset_name = "evalplus/humanevalplus"
    example_id_prefix = "humanevalplus"

