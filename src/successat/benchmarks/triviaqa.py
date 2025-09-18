"""TriviaQA short answer with ARC-Easy and ARC-Challenge multiple choice."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, TypedDict

from ..datasets import load_dataset

from .base import Benchmark, BenchmarkExample


CHOICE_LABELS: Tuple[str, ...] = ("A", "B", "C", "D")
TRIVIAQA_DATASET = "TimoImhof/Splits_Subset_TriviaQa"
ARC_DATASET = "ai2_arc"


class _ArcVariant(TypedDict):
    """Metadata describing an ARC configuration."""

    config: str
    aliases: set[str]
    split_aliases: Mapping[str, str]


_ARC_SPLIT_ALIASES: Mapping[str, str] = {
    "train": "train",
    "validation": "validation",
    "val": "validation",
    "dev": "validation",
    "test": "test",
}

ARC_VARIANTS: Mapping[str, _ArcVariant] = {
    "arc_easy": {
        "config": "ARC-Easy",
        "aliases": {"arc", "arc_easy", "arc-easy"},
        "split_aliases": _ARC_SPLIT_ALIASES,
    },
    "arc_challenge": {
        "config": "ARC-Challenge",
        "aliases": {"arc_challenge", "arc-challenge"},
        "split_aliases": _ARC_SPLIT_ALIASES,
    },
}

_SHORT_ANSWER_NORMALISER = re.compile(r"[^a-z0-9]+")
_OPTION_PATTERN = re.compile(r"\b([A-D])\b", re.IGNORECASE)


class TriviaQABenchmark(Benchmark):
    """Blend of TriviaQA short-answer questions with ARC-Easy multiple choice."""

    name = "triviaqa"
    description = "TriviaQA subset plus ARC-Easy science questions."
    default_split = "validation"

    _TRIVIA_SPLIT_ALIASES: Mapping[str, str] = {
        "train": "split_0",
        "validation": "split_1",
        "val": "split_1",
        "dev": "split_1",
        "test": "split_2",
        "split_0": "split_0",
        "split_1": "split_1",
        "split_2": "split_2",
        "no_split": "no_split",
        "shortcut": "shortcut",
    }

    _ARC_SPLIT_ALIASES: Mapping[str, str] = {
        "train": "train",
        "validation": "validation",
        "val": "validation",
        "dev": "validation",
        "test": "test",
    }

    def __init__(self, client):
        super().__init__(client)
        self._trivia_examples: Dict[str, List[BenchmarkExample]] = {}
        self._arc_examples: Dict[Tuple[str, str], List[BenchmarkExample]] = {}
        self._combined_cache: Dict[str, List[BenchmarkExample]] = {}

    def available_splits(self) -> Sequence[str]:
        combined = ["train", "validation", "test"]
        trivia_splits = sorted({value for value in self._TRIVIA_SPLIT_ALIASES.values()})
        trivia_specs = [f"triviaqa:{split}" for split in trivia_splits]

        arc_specs: List[str] = []
        for dataset_key, variant in ARC_VARIANTS.items():
            variant_splits = sorted({value for value in variant["split_aliases"].values()})
            arc_specs.extend(f"{dataset_key}:{split}" for split in variant_splits)

        return [*combined, *trivia_specs, *arc_specs]

    def examples_for_split(self, split: str) -> Sequence[BenchmarkExample]:
        spec = self._resolve_split_spec(split)
        kind = spec["kind"]

        if kind == "triviaqa":
            return self._load_trivia_examples(spec["trivia"])

        if kind == "arc":
            arc_spec = spec["arc"]
            return self._load_arc_examples(arc_spec["dataset"], arc_spec["split"])

        split_key = split.lower()
        if split_key not in self._combined_cache:
            trivia_examples = self._load_trivia_examples(spec["trivia"])
            arc_spec = spec["arc"]
            arc_examples = self._load_arc_examples(arc_spec["dataset"], arc_spec["split"])
            self._combined_cache[split_key] = [*trivia_examples, *arc_examples]
        return self._combined_cache[split_key]

    def is_correct(self, example: BenchmarkExample, response_text: str, response: object):
        dataset = example.metadata.get("dataset")
        if dataset == "triviaqa":
            return self._score_short_answer(example, response_text)
        return self._score_multiple_choice(example, response_text)

    # Internal helpers ---------------------------------------------------
    def _resolve_split_spec(self, split: str) -> Mapping[str, str]:
        split_lower = split.lower()
        if ":" in split_lower:
            dataset, dataset_split = (part.strip() for part in split_lower.split(":", 1))
            if dataset in {"triviaqa", "trivia_qa"}:
                trivia_split = self._resolve_trivia_split(dataset_split)
                return {"kind": "triviaqa", "trivia": trivia_split}
            try:
                arc_dataset = self._resolve_arc_dataset(dataset)
            except ValueError as exc:
                msg = "Unknown dataset specifier for TriviaQA benchmark."
                raise ValueError(msg) from exc
            arc_split = self._resolve_arc_split(dataset_split, arc_dataset)
            return {"kind": "arc", "arc": {"dataset": arc_dataset, "split": arc_split}}
            msg = "Unknown dataset specifier for TriviaQA benchmark."
            raise ValueError(msg)

        trivia_split = self._resolve_trivia_split(split_lower, allow_missing=True)
        arc_split = self._resolve_arc_split(split_lower, "arc_easy", allow_missing=True)

        if trivia_split and arc_split:
            return {
                "kind": "combined",
                "trivia": trivia_split,
                "arc": {"dataset": "arc_easy", "split": arc_split},
            }
        if trivia_split:
            return {"kind": "triviaqa", "trivia": trivia_split}
        if arc_split:
            return {"kind": "arc", "arc": {"dataset": "arc_easy", "split": arc_split}}

        msg = (
            "Unsupported split. Use 'train', 'validation', 'test', "
            "'triviaqa:<split>', 'arc_easy:<split>' or 'arc_challenge:<split>'."
        )
        raise ValueError(msg)

    def _resolve_trivia_split(self, split: str, allow_missing: bool = False) -> str | None:
        if split in self._TRIVIA_SPLIT_ALIASES:
            return self._TRIVIA_SPLIT_ALIASES[split]
        if split.startswith("split_") and split[6:].isdigit():
            return split
        if split in {"no_split", "shortcut"}:
            return split
        if allow_missing:
            return None
        msg = "Unknown TriviaQA split."
        raise ValueError(msg)

    def _resolve_arc_dataset(self, dataset: str) -> str:
        for dataset_key, variant in ARC_VARIANTS.items():
            if dataset == dataset_key or dataset in variant["aliases"]:
                return dataset_key
        msg = "Unknown ARC dataset specifier."
        raise ValueError(msg)

    def _resolve_arc_split(
        self, split: str, dataset_key: str, allow_missing: bool = False
    ) -> str | None:
        variant = ARC_VARIANTS[dataset_key]
        alias_map = variant["split_aliases"]
        if split in alias_map:
            return alias_map[split]
        if split in {"train", "validation", "test"}:
            return split
        if allow_missing:
            return None
        config = variant["config"]
        msg = f"Unknown {config} split."
        raise ValueError(msg)

    def _load_trivia_examples(self, dataset_split: str) -> Sequence[BenchmarkExample]:
        if dataset_split not in self._trivia_examples:
            dataset = load_dataset(TRIVIAQA_DATASET, split=dataset_split)
            examples = [
                self._convert_trivia_example(row, index, dataset_split)
                for index, row in enumerate(dataset)
            ]
            self._trivia_examples[dataset_split] = examples
        return self._trivia_examples[dataset_split]

    def _load_arc_examples(self, dataset_key: str, dataset_split: str) -> Sequence[BenchmarkExample]:
        cache_key = (dataset_key, dataset_split)
        if cache_key not in self._arc_examples:
            variant = ARC_VARIANTS[dataset_key]
            dataset = load_dataset(ARC_DATASET, variant["config"], split=dataset_split)
            examples = [
                self._convert_arc_example(dataset_key, row, index, dataset_split)
                for index, row in enumerate(dataset)
            ]
            self._arc_examples[cache_key] = examples
        return self._arc_examples[cache_key]

    def _convert_trivia_example(self, row: Mapping[str, object], index: int, dataset_split: str) -> BenchmarkExample:
        question = str(row["question"]).strip()
        context = str(row.get("context", "")).strip()
        answer_info = row.get("answers") or {}
        aliases = [
            str(alias).strip()
            for alias in (answer_info.get("text") or [])
            if str(alias).strip()
        ]
        primary_answer = aliases[0] if aliases else ""

        prompt = (
            "Answer the following trivia question with a concise response.\n\n"
            f"Question: {question}\nAnswer:"
        )

        metadata = {
            "dataset": "triviaqa",
            "dataset_split": dataset_split,
            "question": question,
            "context": context,
            "aliases": aliases,
        }

        return BenchmarkExample(
            id=f"triviaqa-{dataset_split}-{row['id']}",
            prompt=prompt,
            target=primary_answer,
            metadata=metadata,
        )

    def _convert_arc_example(
        self, dataset_key: str, row: Mapping[str, object], index: int, dataset_split: str
    ) -> BenchmarkExample:
        question = str(row["question"]).strip()
        choices = row["choices"]
        labels: Iterable[str] = choices["label"]
        texts: Iterable[str] = choices["text"]
        choice_map = dict(zip(labels, texts))
        formatted_choices = "\n".join(
            f"{label}. {choice_map[label]}" for label in CHOICE_LABELS if label in choice_map
        )

        prompt = (
            "Answer the science question by choosing the correct option letter.\n\n"
            f"Question: {question}\nOptions:\n{formatted_choices}\n\nAnswer:"
        )

        metadata = {
            "dataset": dataset_key,
            "dataset_split": dataset_split,
            "question": question,
            "choices": {label: choice_map[label] for label in CHOICE_LABELS if label in choice_map},
        }

        return BenchmarkExample(
            id=f"{dataset_key.replace('_', '-')}-{dataset_split}-{row['id']}",
            prompt=prompt,
            target=str(row["answerKey"]).upper(),
            metadata=metadata,
        )

    def _score_short_answer(self, example: BenchmarkExample, response_text: str):
        aliases: Iterable[str] = example.metadata.get("aliases", [])
        prediction = self._extract_short_answer(response_text)
        normalised_prediction = self._normalise_short_answer(prediction)
        normalised_aliases = [self._normalise_short_answer(alias) for alias in aliases]

        details = {
            "prediction": prediction,
            "normalised_prediction": normalised_prediction,
            "aliases": list(aliases),
        }

        for alias in normalised_aliases:
            if alias and alias in normalised_prediction:
                details["matched_alias"] = alias
                return True, details

        details["reason"] = "no alias matched"
        return False, details

    def _score_multiple_choice(self, example: BenchmarkExample, response_text: str):
        choices: Mapping[str, str] = example.metadata.get("choices", {})
        expected = str(example.target).upper()
        match = _OPTION_PATTERN.search(response_text)

        details = {
            "choices": choices,
            "expected": expected,
        }

        if match:
            letter = match.group(1).upper()
            details["predicted_letter"] = letter
            details["predicted_choice"] = choices.get(letter)
            return letter == expected, details

        normalised = response_text.strip().lower()
        for letter, option in choices.items():
            if option.lower() in normalised:
                details["predicted_letter"] = letter
                details["predicted_choice"] = option
                return letter == expected, details

        details["reason"] = "no option detected"
        return False, details

    @staticmethod
    def _extract_short_answer(response_text: str) -> str:
        stripped = response_text.strip()
        if not stripped:
            return ""
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if not lines:
            return stripped
        return lines[-1]

    @staticmethod
    def _normalise_short_answer(answer: str) -> str:
        return _SHORT_ANSWER_NORMALISER.sub(" ", answer.lower()).strip()

