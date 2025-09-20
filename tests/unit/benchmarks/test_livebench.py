"""Tests for the LiveBench coding benchmark harness."""

from __future__ import annotations

import json
import textwrap

from successat.benchmarks.base import BenchmarkExample
from successat.benchmarks.livebench import (
    LiveBenchCodingBenchmark,
    _LiveBenchExampleData,
)


def _make_benchmark() -> LiveBenchCodingBenchmark:
    """Create a benchmark instance for direct evaluation."""

    return LiveBenchCodingBenchmark(client=None)  # type: ignore[arg-type]


def test_coerce_arguments_returns_all_inputs_when_stub_is_incomplete() -> None:
    """All parsed inputs should be forwarded when the stub omits parameters."""

    parsed_input = [[1, 2, 3], 5]
    args = LiveBenchCodingBenchmark._coerce_arguments(parsed_input, 1)

    assert args == ([1, 2, 3], 5)


def test_coerce_arguments_preserves_single_argument_lists() -> None:
    """Single-argument lists should remain wrapped as one argument."""

    parsed_input = [1, 2, 3]
    args = LiveBenchCodingBenchmark._coerce_arguments(parsed_input, 1)

    assert args == ([1, 2, 3],)


def test_functional_tests_prefer_candidate_signature() -> None:
    """A candidate-defined signature should override the starter stub."""

    benchmark = _make_benchmark()
    example_id = "livebench-coding-demo"
    example = BenchmarkExample(id=example_id, prompt="", target="pass")

    tests = [
        {
            "input": "[[1, 2, 3], 4]",
            "output": "10",
        }
    ]
    example_data = _LiveBenchExampleData(
        public_tests_json=json.dumps(tests),
        private_tests_blob=None,
        test_mode="functional",
        method_name="minOperations",
        parameter_count=1,
        starter_code=textwrap.dedent(
            """
            class Solution:
                def minOperations(self, nums):
                    pass
            """
        ),
    )
    benchmark._example_data[example_id] = example_data

    candidate_code = textwrap.dedent(
        """
        ```python
        class Solution:
            def minOperations(self, nums, k):
                return sum(nums) + k
        ```
        """
    )

    correct, details = benchmark.is_correct(example, candidate_code, response=None)

    assert correct is True
    assert details["tests_run"] == 1
    assert details["candidate_parameter_count"] == 2
    assert details["starter_parameter_count"] == 1


def test_functional_tests_surface_type_error_for_missing_parameter() -> None:
    """Candidates that omit required parameters should fail with TypeError."""

    benchmark = _make_benchmark()
    example_id = "livebench-coding-demo-missing"
    example = BenchmarkExample(id=example_id, prompt="", target="pass")

    tests = [
        {
            "input": "[[1, 2, 3], 4]",
            "output": "10",
        }
    ]
    example_data = _LiveBenchExampleData(
        public_tests_json=json.dumps(tests),
        private_tests_blob=None,
        test_mode="functional",
        method_name="minOperations",
        parameter_count=1,
        starter_code=textwrap.dedent(
            """
            class Solution:
                def minOperations(self, nums):
                    pass
            """
        ),
    )
    benchmark._example_data[example_id] = example_data

    candidate_code = textwrap.dedent(
        """
        ```python
        class Solution:
            def minOperations(self, nums):
                return sum(nums)
        ```
        """
    )

    correct, details = benchmark.is_correct(example, candidate_code, response=None)

    assert correct is False
    assert "TypeError" in details["error"]
    assert "positional argument" in details["error"]


def test_stdin_wrapper_supports_buffer_access() -> None:
    """Stdin-based tasks should support reading from both text and buffer APIs."""

    benchmark = _make_benchmark()
    example_id = "livebench-stdin-demo"
    example = BenchmarkExample(id=example_id, prompt="", target="pass")

    tests = [
        {
            "input": "alpha\nbeta\n",
            "output": "alpha\nbeta\nalpha\nbeta\n",
        }
    ]
    example_data = _LiveBenchExampleData(
        public_tests_json=json.dumps(tests),
        private_tests_blob=None,
        test_mode="stdin",
        method_name=None,
        parameter_count=0,
        starter_code="",
    )
    benchmark._example_data[example_id] = example_data

    candidate_code = textwrap.dedent(
        """
        ```python
        import sys

        data_bytes = sys.stdin.buffer.read()
        sys.stdin.seek(0)
        text_data = sys.stdin.read()
        print(data_bytes.decode("utf-8").strip())
        print(text_data.strip())
        ```
        """
    )

    correct, details = benchmark.is_correct(example, candidate_code, response=None)

    assert correct is True
    assert details["tests_run"] == 1
