"""Unit tests for the LiveBench benchmark helpers."""

from __future__ import annotations

from successat.benchmarks.base import BenchmarkExample
from successat.benchmarks.livebench import LiveBenchBenchmark


def test_is_correct_prefers_last_code_block_and_strips_explanations() -> None:
    """The benchmark should compile the final code block only."""

    benchmark = LiveBenchBenchmark(client=object())
    example = BenchmarkExample(
        id="livebench-0",
        prompt="",
        target=None,
        metadata={
            "test_code": (
                "def check(solution_cls):\n"
                "    instance = solution_cls()\n"
                "    assert instance.solve() == 42\n"
            )
        },
    )

    response_text = (
        "Here is a rough sketch:\n"
        "```python\n"
        "class Solution:\n"
        "    def solve(self):\n"
        "        return 0\n"
        "```\n"
        "After reconsidering, this version is better:\n"
        "```python\n"
        "class Solution:\n"
        "    def solve(self):\n"
        "        return 42\n"
        "```\n"
        "Thanks for reading!\n"
    )

    correct, details = benchmark.is_correct(example, response_text, response_text)

    assert correct
    assert details == {}
