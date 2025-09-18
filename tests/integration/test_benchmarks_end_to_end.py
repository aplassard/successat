"""End-to-end tests exercising the benchmark pipeline with fake LLMs."""

from __future__ import annotations

from successat.benchmarks import (
    GSM8KBenchmark,
    HumanEvalBenchmark,
    HumanEvalPlusBenchmark,
    MMLUBenchmark,
    TriviaQABenchmark,
    run_benchmark,
)


def test_gsm8k_benchmark_end_to_end(fake_llm_client) -> None:
    sample_client = fake_llm_client([])
    example = GSM8KBenchmark(sample_client).examples_for_split("test")[0]
    expected = example.target

    client = fake_llm_client(f"We compute the result step by step. Final answer: {expected}")

    result = run_benchmark(client, "gsm8k", identifier=0, split="test")

    assert result.correct is True
    assert result.metadata["expected"] == expected
    assert "Final answer" in client.calls[0]["prompt"]


def test_mmlu_benchmark_end_to_end(fake_llm_client) -> None:
    sample_client = fake_llm_client([])
    example = MMLUBenchmark(sample_client).examples_for_split("test")[0]
    expected = example.target

    client = fake_llm_client(f"The correct answer is {expected}.")

    result = run_benchmark(client, "mmlu", identifier=0, split="test")

    assert result.correct is True
    assert result.metadata["expected"] == expected
    assert result.metadata["evaluation_details"]["predicted_letter"] == expected


def test_humaneval_benchmark_end_to_end(fake_llm_client) -> None:
    sample_client = fake_llm_client([])
    example = HumanEvalBenchmark(sample_client).examples_for_split("test")[0]
    entry_point = example.metadata["entry_point"]
    canonical_solution = example.metadata["canonical_solution"]

    prompt_lines = example.prompt.splitlines()
    code_lines = []
    for idx, line in enumerate(prompt_lines):
        code_lines.append(line)
        if line.strip().startswith(f"def {entry_point}"):
            code_lines.extend(prompt_lines[idx + 1 :])
            break
    code_lines.extend(canonical_solution.splitlines())

    code = "\n".join(code_lines)
    response = f"```python\n{code}\n```"
    client = fake_llm_client(response)

    result = run_benchmark(client, "humaneval", identifier=example.id, split="test")

    assert result.correct is True
    assert example.target in result.response_text
    assert example.metadata["entry_point"] in client.calls[0]["prompt"]


def test_humaneval_plus_benchmark_end_to_end(fake_llm_client) -> None:
    sample_client = fake_llm_client([])
    example = HumanEvalPlusBenchmark(sample_client).examples_for_split("test")[0]
    entry_point = example.metadata["entry_point"]
    canonical_solution = example.metadata["canonical_solution"]

    prompt_lines = example.prompt.splitlines()
    code_lines = []
    for idx, line in enumerate(prompt_lines):
        code_lines.append(line)
        if line.strip().startswith(f"def {entry_point}"):
            code_lines.extend(prompt_lines[idx + 1 :])
            break
    code_lines.extend(canonical_solution.splitlines())

    code = "\n".join(code_lines)
    response = f"```python\n{code}\n```"
    client = fake_llm_client(response)

    result = run_benchmark(client, "humaneval+", identifier=example.id, split="test")

    assert result.benchmark == "humaneval+"
    assert result.correct is True
    assert example.target in result.response_text
    assert example.metadata["entry_point"] in client.calls[0]["prompt"]


def test_triviaqa_short_answer(fake_llm_client) -> None:
    sample_client = fake_llm_client([])
    example = TriviaQABenchmark(sample_client).examples_for_split("triviaqa:split_0")[0]
    answer = example.target

    client = fake_llm_client(f"The correct response is the {answer} forces.")

    result = run_benchmark(client, "triviaqa", identifier=example.id, split="triviaqa:split_0")

    assert result.correct is True
    assert answer in result.metadata["expected"]
    assert "matched_alias" in result.metadata["evaluation_details"]


def test_triviaqa_multiple_choice(fake_llm_client) -> None:
    sample_client = fake_llm_client([])
    example = TriviaQABenchmark(sample_client).examples_for_split("arc_easy:train")[0]
    expected_letter = example.target

    client = fake_llm_client(f"Plants release option {expected_letter}.")

    result = run_benchmark(client, "triviaqa", identifier=example.id, split="arc_easy:train")

    assert result.correct is True
    assert result.metadata["evaluation_details"]["predicted_letter"] == expected_letter


def test_incorrect_response_reports_details(fake_llm_client) -> None:
    client = fake_llm_client("The answer is 99")

    result = run_benchmark(client, "gsm8k", identifier=0, split="test")

    assert result.correct is False
    assert result.metadata["evaluation_details"]["predicted_number"] == "99"

