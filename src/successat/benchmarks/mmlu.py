"""Implementation of the MMLU benchmark using the public CAIS dataset."""

from __future__ import annotations

import re
from typing import Dict, List, Mapping, Sequence

from ..datasets import load_dataset

from .base import Benchmark, BenchmarkExample


CHOICE_LABELS = ("A", "B", "C", "D")


class MMLUBenchmark(Benchmark):
    """Multiple-choice reasoning benchmark spanning 57 academic subjects."""

    name = "mmlu"
    description = "Massive Multitask Language Understanding benchmark."
    dataset_name = "cais/mmlu"
    dataset_config = "all"
    default_split = "validation"

    _SPLIT_ALIASES: Mapping[str, str] = {
        "train": "auxiliary_train",
        "auxiliary_train": "auxiliary_train",
        "auxiliary-train": "auxiliary_train",
        "dev": "dev",
        "development": "dev",
        "validation": "validation",
        "val": "validation",
        "test": "test",
    }

    def __init__(self, client):
        super().__init__(client)
        self._examples_by_dataset_split: Dict[str, List[BenchmarkExample]] = {}
        self._alias_cache: Dict[str, List[BenchmarkExample]] = {}

    def available_splits(self) -> Sequence[str]:
        return ["auxiliary_train", "dev", "validation", "test"]

    def examples_for_split(self, split: str) -> Sequence[BenchmarkExample]:
        alias = split.lower()
        if alias in self._alias_cache:
            return self._alias_cache[alias]

        dataset_split = self._resolve_split(alias)
        if dataset_split not in self._examples_by_dataset_split:
            dataset = load_dataset(self.dataset_name, self.dataset_config, split=dataset_split)
            examples = [
                self._convert_row(row, index, dataset_split)
                for index, row in enumerate(dataset)
            ]
            self._examples_by_dataset_split[dataset_split] = examples

        examples = self._examples_by_dataset_split[dataset_split]
        self._alias_cache[alias] = examples
        return examples

    def is_correct(self, example: BenchmarkExample, response_text: str, response: object):
        choices: Mapping[str, str] = example.metadata["choices"]
        expected_letter = str(example.target).upper()
        predicted_letter, predicted_choice = self._parse_choice(response_text, choices)

        details = {
            "predicted_letter": predicted_letter,
            "predicted_choice": predicted_choice,
            "expected_letter": expected_letter,
            "subject": example.metadata.get("subject"),
        }

        if predicted_letter:
            return predicted_letter == expected_letter, details

        if predicted_choice:
            expected_choice = choices.get(expected_letter)
            details["expected_choice"] = expected_choice
            return (predicted_choice or "").lower() == (expected_choice or "").lower(), details

        details["reason"] = "no option detected"
        return False, details

    # Internal helpers ---------------------------------------------------
    def _resolve_split(self, alias: str) -> str:
        if alias in self._SPLIT_ALIASES:
            return self._SPLIT_ALIASES[alias]
        if alias in {"dev", "validation", "test", "auxiliary_train"}:
            return alias
        msg = (
            "MMLU split must be one of 'train', 'dev', 'validation', 'test', or the "
            "underlying dataset split names."
        )
        raise ValueError(msg)

    def _convert_row(self, row: Mapping[str, object], index: int, dataset_split: str) -> BenchmarkExample:
        question = str(row["question"]).strip()
        subject = str(row["subject"]).strip()
        choice_texts = list(row["choices"])
        answer_index = int(row["answer"])
        answer_letter = CHOICE_LABELS[answer_index]

        formatted_choices = "\n".join(
            f"{label}. {text}" for label, text in zip(CHOICE_LABELS, choice_texts)
        )

        prompt = (
            "Answer the multiple-choice question by selecting the correct option letter. "
            "Respond with the letter only.\n\n"
            f"Question: {question}\n"
            f"Options:\n{formatted_choices}\n\nAnswer:"
        )

        metadata = {
            "question": question,
            "subject": subject,
            "choices": dict(zip(CHOICE_LABELS, choice_texts)),
            "dataset_split": dataset_split,
            "answer_index": answer_index,
        }

        return BenchmarkExample(
            id=f"mmlu-{dataset_split}-{subject}-{index}",
            prompt=prompt,
            target=answer_letter,
            metadata=metadata,
        )

    @staticmethod
    def _parse_choice(text: str, choices: Mapping[str, str]) -> tuple[str | None, str | None]:
        match = re.search(r"\b([A-D])\b", text, flags=re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            return letter, choices.get(letter)

        normalised_text = text.strip().lower()
        for letter, option in choices.items():
            if option.lower() in normalised_text:
                return None, option

        return None, None

