"""Implementation of the GSM8K benchmark backed by the public dataset."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Sequence

from datasets import load_dataset

from .base import Benchmark, BenchmarkExample


_ANSWER_DELIMITER = re.compile(r"####\s*(.+)")
_FINAL_ANSWER_PATTERN = re.compile(r"final answer\s*[:\-]?\s*(.*)", re.IGNORECASE | re.DOTALL)
_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


class GSM8KBenchmark(Benchmark):
    """Grade school math problems requiring multi-step reasoning."""

    name = "gsm8k"
    description = "Grade School Math 8K benchmark from OpenAI."
    dataset_name = "gsm8k"
    dataset_config = "main"

    def __init__(self, client):
        super().__init__(client)
        self._examples_by_split: Dict[str, List[BenchmarkExample]] = {}

    def available_splits(self) -> Sequence[str]:
        return ["train", "test"]

    # Public API ---------------------------------------------------------
    def examples_for_split(self, split: str) -> Sequence[BenchmarkExample]:
        dataset_split = self._resolve_split(split)
        if dataset_split not in self._examples_by_split:
            dataset = load_dataset(self.dataset_name, self.dataset_config, split=dataset_split)
            examples = [
                self._convert_row(row, index, dataset_split)
                for index, row in enumerate(dataset)
            ]
            self._examples_by_split[dataset_split] = examples
        return self._examples_by_split[dataset_split]

    def is_correct(self, example: BenchmarkExample, response_text: str, response: object):
        expected_answer = str(example.target)
        predicted_raw = self._extract_prediction(response_text)

        if predicted_raw is None:
            details = {"expected": expected_answer, "reason": "no answer detected"}
            return False, details

        expected_normalised = self._normalise_answer(expected_answer)
        predicted_normalised = self._normalise_answer(predicted_raw)

        details = {
            "expected": expected_answer,
            "expected_normalised": expected_normalised,
            "predicted_raw": predicted_raw,
            "predicted_normalised": predicted_normalised,
        }

        if predicted_normalised == expected_normalised:
            return True, details

        expected_number = self._extract_number(expected_answer)
        predicted_number = self._extract_number(predicted_raw)
        if expected_number is not None:
            details["expected_number"] = str(expected_number)
        if predicted_number is not None:
            details["predicted_number"] = str(predicted_number)

        if expected_number is not None and predicted_number is not None:
            return expected_number == predicted_number, details

        if expected_normalised and expected_normalised in predicted_normalised:
            return True, details

        return False, details

    # Internal helpers ---------------------------------------------------
    def _resolve_split(self, split: str) -> str:
        split_lower = split.lower()
        if split_lower in {"train", "test"}:
            return split_lower
        msg = "GSM8K provides only 'train' and 'test' splits."
        raise ValueError(msg)

    def _convert_row(self, row: dict, index: int, dataset_split: str) -> BenchmarkExample:
        question = row["question"].strip()
        raw_answer = row["answer"].strip()
        final_answer = self._extract_reference_answer(raw_answer)

        prompt = (
            "You are an expert math tutor. Solve the following grade school math "
            "problem and return only the final numeric answer.\n\n"
            f"{question}\n\nFinal answer:"
        )

        metadata = {
            "question": question,
            "dataset_split": dataset_split,
            "raw_answer": raw_answer,
        }

        return BenchmarkExample(
            id=f"gsm8k-{dataset_split}-{index}",
            prompt=prompt,
            target=final_answer,
            metadata=metadata,
        )

    @classmethod
    def _extract_reference_answer(cls, answer: str) -> str:
        match = _ANSWER_DELIMITER.search(answer)
        if match:
            return match.group(1).strip()
        return answer.strip()

    @classmethod
    def _extract_prediction(cls, response_text: str) -> str | None:
        stripped = response_text.strip()
        if not stripped:
            return None

        match = _FINAL_ANSWER_PATTERN.search(stripped)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate.splitlines()[0].strip()

        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if lines:
            return lines[-1]

        return stripped

    @staticmethod
    def _normalise_answer(answer: str) -> str:
        cleaned = answer.replace(",", "").replace("$", "").lower().strip()
        cleaned = re.sub(r"\b(dollars?|euros?|pounds)\b", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.rstrip(".")
        return cleaned.strip()

    @classmethod
    def _extract_number(cls, text: str) -> Decimal | None:
        matches = _NUMBER_PATTERN.findall(text.replace(",", ""))
        if not matches:
            return None
        last = matches[-1]
        try:
            return Decimal(last)
        except InvalidOperation:
            return None

