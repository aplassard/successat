"""LiveBench benchmark support."""

from __future__ import annotations

import re
from typing import Dict, Mapping, Sequence

from .base import Benchmark, BenchmarkExample

_CODE_FENCE_PATTERN = re.compile(r"```(?:[^`\n]*\r?\n)?(.*?)```", re.DOTALL)


def _strip_code_fence(code: str) -> str:
    """Return executable Python from a model response."""

    text = code.strip()

    blocks = _CODE_FENCE_PATTERN.findall(text)
    if blocks:
        return blocks[-1].strip()

    solution_index = text.find("class Solution")
    if solution_index != -1:
        return text[solution_index:].strip()

    return text


class LiveBenchBenchmark(Benchmark):
    """Evaluate LiveBench style problems."""

    name = "livebench"

    def examples_for_split(self, split: str) -> Sequence[BenchmarkExample]:  # pragma: no cover - dataset plumbing
        msg = "LiveBench dataset loading is not implemented in the test fixture."
        raise NotImplementedError(msg)

    def is_correct(
        self,
        example: BenchmarkExample,
        response_text: str,
        response: object,
    ) -> tuple[bool, Mapping[str, object]]:
        """Compile the candidate solution and execute the benchmark checks."""

        candidate_code = _strip_code_fence(response_text)
        namespace: Dict[str, object] = {}

        try:
            exec(candidate_code, namespace)  # noqa: S102 - required to evaluate generated code
        except Exception as exc:  # pragma: no cover - error path validated in tests
            return False, {"error": f"candidate execution failed: {exc}"}

        solution_cls = namespace.get("Solution")
        if not isinstance(solution_cls, type):
            return False, {"error": "class 'Solution' not defined"}

        test_code = example.metadata.get("test_code") if example.metadata else None
        if not test_code:
            return False, {"error": "benchmark test harness missing 'test_code'"}

        test_namespace: Dict[str, object] = {}
        try:
            exec(str(test_code), test_namespace)  # noqa: S102 - dataset harness execution
        except Exception as exc:  # pragma: no cover - dataset should always execute
            return False, {"error": f"failed to load tests: {exc}"}

        check = test_namespace.get("check")
        if not callable(check):
            return False, {"error": "benchmark test harness missing 'check' function"}

        try:
            check(solution_cls)
        except AssertionError as exc:
            return False, {"error": f"assertion failed: {exc}"}
        except Exception as exc:  # pragma: no cover - dataset exercises edge cases
            return False, {"error": f"tests raised {exc.__class__.__name__}: {exc}"}

        return True, {}


__all__ = ["LiveBenchBenchmark"]
