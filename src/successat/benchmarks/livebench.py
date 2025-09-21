"""Implementation of the LiveBench coding benchmark."""

from __future__ import annotations

import ast
import base64
import contextlib
import io
import json
import math
import pickle
import re
import sys
import zlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, List, Mapping, MutableMapping, Sequence

from ..datasets import load_dataset

from .base import Benchmark, BenchmarkExample


_CODE_FENCE_RE = re.compile(r"^```(?:python)?\s*|```$", re.IGNORECASE | re.MULTILINE)
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_SOLUTION_CLASS_RE = re.compile(r"^\s*class\s+Solution\b", re.MULTILINE)
_SIGNATURE_RE = re.compile(r"def\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)")
_SOLUTION_TAG_RE = re.compile(r"<solution>(.*?)</solution>", re.IGNORECASE | re.DOTALL)
_ANSWER_LINE_RE = re.compile(r"answer\s*:\s*(.*)", re.IGNORECASE)

_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}

_ORDINAL_WORDS = {
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "sixth": "6",
    "seventh": "7",
    "eighth": "8",
    "ninth": "9",
    "tenth": "10",
}

_BOOLEAN_NORMALISATIONS = {
    "y": "yes",
    "yes": "yes",
    "yeah": "yes",
    "yep": "yes",
    "true": "yes",
    "n": "no",
    "no": "no",
    "nope": "no",
    "false": "no",
}

_UNICODE_OPERATOR_REPLACEMENTS = {
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
    "−": "-",
    "–": "-",
    "—": "-",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "×": "*",
    "÷": "/",
    "·": "*",
    "∗": "*",
    "→": "->",
    "←": "<-",
}


@dataclass(frozen=True)
class _ConvertedRow:
    """Representation of a processed LiveBench dataset row."""

    example: BenchmarkExample
    release_key: str
    release_date: datetime | None
    extra_data: Any = None


def _prepare_prompt(row: Mapping[str, Any], fallback: str = "") -> tuple[str, Sequence[Mapping[str, str]] | None]:
    """Return the primary prompt and optional extra messages for a dataset row."""

    turns = [
        turn
        for turn in row.get("turns", [])
        if isinstance(turn, str) and turn.strip()
    ]
    prompt = turns[-1] if turns else fallback
    prompt = (prompt or "").strip()
    if not prompt:
        prompt = fallback.strip()

    extra_messages: Sequence[Mapping[str, str]] | None = None
    if len(turns) > 1:
        extra_messages = [
            {"role": "user", "content": text}
            for text in turns[:-1]
            if text.strip()
        ]

    return prompt, extra_messages if extra_messages else None


class _BaseLiveBenchBenchmark(Benchmark):
    """Common functionality shared by LiveBench benchmark variants."""

    dataset_name: str
    default_split = "latest"

    def __init__(self, client) -> None:
        super().__init__(client)
        self._examples_by_split: MutableMapping[str, List[BenchmarkExample]] = {}
        self._split_order: List[str] | None = None
        self._prepared = False

    # Benchmark API --------------------------------------------------
    def available_splits(self) -> Sequence[str]:
        self._ensure_prepared()
        if self._split_order is None:
            return [self.default_split]
        return list(self._split_order)

    def examples_for_split(self, split: str) -> Sequence[BenchmarkExample]:
        self._ensure_prepared()
        split_key = split.lower()
        try:
            return self._examples_by_split[split_key]
        except KeyError as exc:
            msg = (
                f"Split '{split}' is not available for {self.name}. "
                f"Available splits: {sorted(self._examples_by_split)}"
            )
            raise ValueError(msg) from exc

    # Internal helpers -----------------------------------------------
    def _ensure_prepared(self) -> None:
        if self._prepared:
            return

        dataset = load_dataset(self.dataset_name, split="test")
        self._prepare_examples(dataset)
        self._prepared = True

    def _prepare_examples(self, rows: Iterable[Mapping[str, Any]]) -> None:
        self._clear_extra_data()

        release_groups: MutableMapping[str, List[BenchmarkExample]] = defaultdict(list)
        release_dates: MutableMapping[str, datetime] = {}
        all_examples: List[BenchmarkExample] = []

        for index, row in enumerate(rows):
            converted = self._convert_row(row, index)
            if converted.extra_data is not None:
                self._store_example_data(converted.example.id, converted.extra_data)
            release_groups[converted.release_key].append(converted.example)
            if converted.release_date is not None:
                release_dates[converted.release_key] = converted.release_date
            all_examples.append(converted.example)

        if not all_examples:
            self._examples_by_split = {self.default_split: []}
            self._split_order = [self.default_split]
            return

        self._examples_by_split = {"all": list(all_examples)}

        for key, examples in release_groups.items():
            self._examples_by_split[f"release-{key}"] = list(examples)

        if release_groups:
            if release_dates:
                latest_key = max(
                    release_dates,
                    key=lambda item: (release_dates[item], item),
                )
            else:
                latest_key = next(iter(release_groups))
            self._examples_by_split[self.default_split] = list(
                release_groups[latest_key]
            )

            history_examples: List[BenchmarkExample] = []
            for key, examples in release_groups.items():
                if key != latest_key:
                    history_examples.extend(examples)
            if history_examples:
                self._examples_by_split["history"] = list(history_examples)

        for name, examples in list(self._examples_by_split.items()):
            self._examples_by_split[name] = list(examples)

        split_names: List[str] = []
        if self.default_split in self._examples_by_split:
            split_names.append(self.default_split)
        if "history" in self._examples_by_split:
            split_names.append("history")
        if "all" in self._examples_by_split:
            split_names.append("all")

        release_names = sorted(
            name for name in self._examples_by_split if name.startswith("release-")
        )
        split_names.extend(release_names)

        extras = [
            name
            for name in sorted(self._examples_by_split)
            if name not in split_names
        ]
        split_names.extend(extras)

        self._split_order = split_names
        self._examples_by_split = {
            name: self._examples_by_split[name] for name in self._split_order
        }

    def _convert_row(
        self,
        row: Mapping[str, Any],
        index: int,
    ) -> _ConvertedRow:  # pragma: no cover - defined by subclasses
        raise NotImplementedError

    def _clear_extra_data(self) -> None:
        """Reset any subclass-specific caches prior to loading data."""

    def _store_example_data(self, example_id: str, data: Any) -> None:
        """Persist extra data associated with ``example_id`` if required."""

    @staticmethod
    def _format_datetime(value: Any) -> str | None:
        if isinstance(value, datetime):
            return value.isoformat()
        return None

    @staticmethod
    def _format_release_key(release_dt: datetime | None) -> str:
        if release_dt is None:
            return "unknown"
        return release_dt.strftime("%Y-%m")

@dataclass
class _LiveBenchExampleData:
    """Additional information required to evaluate LiveBench examples."""

    public_tests_json: str
    private_tests_blob: str | None
    test_mode: str
    method_name: str | None
    parameter_count: int
    starter_code: str


class LiveBenchCodingBenchmark(_BaseLiveBenchBenchmark):
    """Evaluate LiveBench coding tasks using public and private test suites."""

    name = "livebench-coding"
    description = "Python coding problems from LiveBench with execution-based scoring."
    dataset_name = "livebench/coding"

    def __init__(self, client) -> None:
        super().__init__(client)
        self._example_data: dict[str, _LiveBenchExampleData] = {}

    def is_correct(self, example: BenchmarkExample, response_text: str, response: object):
        example_data = self._example_data.get(example.id)
        if example_data is None:  # pragma: no cover - defensive guard
            return False, {"error": f"no evaluation data for example {example.id}"}

        candidate_code = self._strip_code_fence(response_text)
        if not candidate_code.strip():
            return False, {"error": "model response did not contain Python code"}

        source_to_compile = candidate_code
        task = (example.metadata or {}).get("task")
        starter_code = example_data.starter_code
        if (
            task == "coding_completion"
            and starter_code.strip()
            and example_data.test_mode != "stdin"
            and not _SOLUTION_CLASS_RE.search(candidate_code)
        ):
            source_to_compile = self._merge_starter_and_completion(
                starter_code, candidate_code
            )

        try:
            compiled = compile(source_to_compile, "<candidate>", "exec")
        except SyntaxError as exc:
            return False, {"error": f"candidate code failed to compile: {exc}"}

        candidate_method_name, candidate_params = self._parse_signature(candidate_code)
        candidate_parameter_count: int | None = None
        if (
            example_data.method_name
            and candidate_method_name
            and candidate_method_name == example_data.method_name
        ):
            candidate_parameter_count = len(candidate_params)

        try:
            tests = self._load_tests(example_data)
        except ValueError as exc:
            return False, {"error": str(exc)}

        if not tests:
            return False, {"error": "benchmark example does not define any tests"}

        if example_data.test_mode == "stdin":
            return self._run_stdin_tests(compiled, tests)

        return self._run_functional_tests(
            compiled,
            tests,
            example_data,
            candidate_parameter_count,
        )

    # Internal helpers ---------------------------------------------------
    def _clear_extra_data(self) -> None:
        self._example_data.clear()

    def _store_example_data(self, example_id: str, data: Any) -> None:
        if isinstance(data, _LiveBenchExampleData):
            self._example_data[example_id] = data

    def _convert_row(
        self,
        row: Mapping[str, Any],
        index: int,
    ) -> _ConvertedRow:
        question_id = str(row["question_id"])
        fallback_prompt = row.get("original_json", {}).get("question_content", "")
        prompt, extra_messages = _prepare_prompt(row, fallback=fallback_prompt)

        release_dt: datetime | None = row.get("livebench_release_date")
        release_key = self._format_release_key(release_dt)

        public_tests_json = row.get("public_test_cases", "") or "[]"
        private_blob = row.get("private_test_cases") or None

        starter_code = row.get("original_json", {}).get("starter_code", "") or ""
        method_name, params = self._parse_signature(starter_code)

        prompt = self._inject_starter_code(prompt, starter_code)
        system_prompt = self._select_system_prompt(row.get("task"))

        tests_preview = self._preview_test_types(public_tests_json, private_blob)
        metadata = {
            "question_id": question_id,
            "question_title": row.get("question_title"),
            "category": row.get("category"),
            "task": row.get("task"),
            "release_date": self._format_datetime(row.get("release_date")),
            "livebench_release_date": self._format_datetime(release_dt),
            "livebench_removal_date": self._format_datetime(row.get("livebench_removal_date")),
            "public_test_count": tests_preview["public"],
            "private_test_count": tests_preview["private"],
            "test_types": tests_preview["types"],
        }

        example = BenchmarkExample(
            id=f"livebench-coding-{question_id}",
            prompt=prompt,
            target="pass",
            metadata=metadata,
            system_prompt=system_prompt,
            extra_messages=extra_messages,
        )

        example_data = _LiveBenchExampleData(
            public_tests_json=public_tests_json,
            private_tests_blob=private_blob,
            test_mode=tests_preview["mode"],
            method_name=method_name,
            parameter_count=len(params),
            starter_code=starter_code,
        )

        return _ConvertedRow(
            example=example,
            release_key=release_key,
            release_date=release_dt,
            extra_data=example_data,
        )

    @staticmethod
    def _inject_starter_code(prompt: str, starter_code: str) -> str:
        if not prompt or not starter_code.strip():
            return prompt

        replacement_block = "```python\n" + starter_code.strip("\n") + "\n```"
        updated, count = _CODE_BLOCK_RE.subn(replacement_block, prompt, count=1)
        if count == 0:
            return prompt
        return updated

    @staticmethod
    def _select_system_prompt(task: str | None) -> str | None:
        if not task:
            return (
                "You are an expert Python developer. Follow the user's last instructions exactly. "
                "Respond with Python code only, inside a single markdown code block, with no explanations."
            )

        task_lower = task.lower()
        if task_lower == "coding_completion":
            return (
                "You are an expert Python developer. Provide only the missing portion requested by the user. "
                "Return Python code alone inside a markdown code block; do not repeat the starter code or add explanations."
            )

        return (
            "You are an expert Python developer. Produce a complete solution that follows the user's request. "
            "Reply with Python code only inside a single markdown code block, without any commentary."
        )

    @staticmethod
    def _merge_starter_and_completion(starter: str, completion: str) -> str:
        indentation = LiveBenchCodingBenchmark._infer_completion_indent(starter)
        completion = completion.lstrip("\n")

        starter_body = starter.rstrip("\r\n")
        starter_lines = starter_body.split("\n")
        if starter_lines and starter_lines[-1].strip() == "":
            starter_lines.pop()
        starter_prefix = "\n".join(starter_lines)
        if starter_prefix:
            starter_prefix += "\n"

        if indentation:
            adjusted_lines: list[str] = []
            for line in completion.splitlines():
                if not line.strip():
                    adjusted_lines.append(line)
                    continue
                first_char = line[0]
                if first_char.isspace():
                    adjusted_lines.append(line)
                else:
                    adjusted_lines.append(indentation + line)
            completion = "\n".join(adjusted_lines)

        return starter_prefix + completion

    @staticmethod
    def _infer_completion_indent(starter: str) -> str:
        for line in reversed(starter.splitlines()):
            if not line:
                continue
            whitespace = line[: len(line) - len(line.lstrip(" 	"))]
            stripped = line.strip()
            if not stripped:
                if whitespace:
                    return whitespace
                continue
            if stripped.endswith(":"):
                return whitespace + "    "
            return whitespace
        return ""

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        fenced_blocks = _CODE_BLOCK_RE.findall(text)
        if fenced_blocks:
            return LiveBenchCodingBenchmark._normalise_code(fenced_blocks[-1].strip())

        match = _SOLUTION_CLASS_RE.search(text)
        if match:
            snippet = text[match.start() :].strip()
            return LiveBenchCodingBenchmark._normalise_code(snippet)

        stripped = _CODE_FENCE_RE.sub("", text).strip()
        return LiveBenchCodingBenchmark._normalise_code(stripped)

    @staticmethod
    def _normalise_code(text: str) -> str:
        if not text:
            return text
        for original, replacement in _UNICODE_OPERATOR_REPLACEMENTS.items():
            if original in text:
                text = text.replace(original, replacement)
        return text

    @staticmethod
    def _parse_signature(code: str) -> tuple[str | None, tuple[str, ...]]:
        match = _SIGNATURE_RE.search(code)
        if not match:
            return None, ()

        params_raw = match.group("params")
        params: List[str] = []
        for raw in params_raw.split(","):
            token = raw.strip()
            if not token or token.startswith("*"):
                continue
            name = token.split(":", 1)[0].split("=", 1)[0].strip()
            if name == "self":
                continue
            params.append(name)
        return match.group("name"), tuple(params)

    @staticmethod
    def _preview_test_types(public_json: str, private_blob: str | None) -> Mapping[str, Any]:
        try:
            public_tests = json.loads(public_json)
        except json.JSONDecodeError:
            public_tests = []

        private_tests: list[Mapping[str, Any]] = []
        if private_blob:
            try:
                private_json = pickle.loads(zlib.decompress(base64.b64decode(private_blob)))
                private_tests = json.loads(private_json)
            except Exception:  # pragma: no cover - dataset should be well formed
                private_tests = []

        test_types = {test.get("testtype", "functional") for test in public_tests + private_tests}
        mode = "stdin" if test_types == {"stdin"} else "functional"

        return {
            "public": len(public_tests),
            "private": len(private_tests),
            "types": sorted(test_types),
            "mode": mode,
        }

    def _load_tests(self, example_data: _LiveBenchExampleData) -> list[Mapping[str, Any]]:
        try:
            tests: list[Mapping[str, Any]] = json.loads(example_data.public_tests_json)
        except json.JSONDecodeError as exc:
            msg = f"failed to parse public test cases: {exc}"
            raise ValueError(msg) from exc

        if example_data.private_tests_blob:
            try:
                blob = base64.b64decode(example_data.private_tests_blob)
                private_json = pickle.loads(zlib.decompress(blob))
                private_tests = json.loads(private_json)
                tests.extend(private_tests)
            except Exception as exc:  # pragma: no cover - defensive guard
                msg = f"failed to parse private test cases: {exc}"
                raise ValueError(msg) from exc

        return tests

    def _run_functional_tests(
        self,
        compiled: Any,
        tests: Sequence[Mapping[str, Any]],
        example_data: _LiveBenchExampleData,
        candidate_parameter_count: int | None = None,
    ) -> tuple[bool, Mapping[str, Any]]:
        if example_data.method_name is None:
            return False, {"error": "functional task is missing a method signature"}

        namespace: dict[str, Any] = {"__name__": "__main__"}
        try:
            exec(compiled, namespace)
        except Exception as exc:
            return False, {"error": f"candidate code raised {exc.__class__.__name__}: {exc}"}

        solution_cls = namespace.get("Solution")
        if solution_cls is None:
            return False, {"error": "candidate code did not define a Solution class"}

        method_name = example_data.method_name
        details: dict[str, Any] = {
            "tests_run": 0,
            "test_mode": "functional",
            "starter_parameter_count": example_data.parameter_count,
        }
        if candidate_parameter_count is not None:
            details["candidate_parameter_count"] = candidate_parameter_count

        for index, test in enumerate(tests):
            parsed_input = self._parse_value(test.get("input", ""))
            parameter_count = (
                candidate_parameter_count
                if candidate_parameter_count is not None
                else example_data.parameter_count
            )
            args = self._coerce_arguments(parsed_input, parameter_count)
            expected = self._parse_value(test.get("output", ""))

            try:
                instance = solution_cls()
                method = getattr(instance, method_name)
                result = method(*args)
            except Exception as exc:
                return False, {
                    "error": f"test {index} raised {exc.__class__.__name__}: {exc}",
                    "test_index": index,
                    "input": test.get("input"),
                }

            if not self._values_match(expected, result):
                return False, {
                    "error": "candidate output did not match expected result",
                    "test_index": index,
                    "input": test.get("input"),
                    "expected": expected,
                    "actual": result,
                }

            details["tests_run"] = index + 1

        return True, details

    def _run_stdin_tests(
        self,
        compiled: Any,
        tests: Sequence[Mapping[str, Any]],
    ) -> tuple[bool, Mapping[str, Any]]:
        details: dict[str, Any] = {"tests_run": 0, "test_mode": "stdin"}

        for index, test in enumerate(tests):
            input_payload = test.get("input", "")
            expected_output = test.get("output", "")

            stdin_buffer = io.BytesIO(input_payload.encode("utf-8"))
            stdin = io.TextIOWrapper(stdin_buffer, encoding="utf-8", newline="")
            stdout = io.StringIO()
            exec_globals: dict[str, Any] = {"__name__": "__main__"}

            try:
                with contextlib.redirect_stdout(stdout):
                    original_stdin = sys.stdin
                    sys.stdin = stdin
                    try:
                        exec(compiled, exec_globals)
                    finally:
                        sys.stdin = original_stdin
                        try:
                            stdin.detach()
                        except Exception:
                            pass
            except Exception as exc:
                return False, {
                    "error": f"test {index} raised {exc.__class__.__name__}: {exc}",
                    "test_index": index,
                    "input": input_payload,
                }

            actual_output = stdout.getvalue()
            if not self._text_outputs_match(expected_output, actual_output):
                return False, {
                    "error": "script output did not match expected value",
                    "test_index": index,
                    "input": input_payload,
                    "expected": expected_output,
                    "actual": actual_output,
                }

            details["tests_run"] = index + 1

        return True, details

    @staticmethod
    def _parse_value(raw: str) -> Any:
        if raw == "":
            return ""
        single_value = LiveBenchCodingBenchmark._parse_scalar_value(raw)
        if single_value is not raw:
            return single_value

        if "\n" in raw:
            multi_value = LiveBenchCodingBenchmark._parse_multi_value_string(raw)
            if multi_value:
                if len(multi_value) == 1:
                    return multi_value[0]
                return multi_value

        return raw

    @staticmethod
    def _parse_scalar_value(raw: str) -> Any:
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(raw)
            except (json.JSONDecodeError, ValueError, SyntaxError):
                continue
        return raw

    @staticmethod
    def _parse_multi_value_string(raw: str) -> list[Any]:
        values: list[Any] = []
        for segment in raw.splitlines():
            stripped = segment.strip()
            if not stripped:
                continue
            values.append(LiveBenchCodingBenchmark._parse_scalar_value(stripped))
        return values

    @staticmethod
    def _coerce_arguments(
        parsed_input: Any, parameter_count: int | None
    ) -> tuple[Any, ...]:
        if isinstance(parsed_input, (list, tuple)):
            if parameter_count == 0:
                return tuple(parsed_input)
            if parameter_count == 1:
                if len(parsed_input) == 1:
                    return (parsed_input[0],)
                if all(
                    not isinstance(item, (list, tuple, dict)) for item in parsed_input
                ):
                    return (parsed_input,)
                return tuple(parsed_input)
            return tuple(parsed_input)

        if parameter_count == 0:
            return ()

        return (parsed_input,)

    @classmethod
    def _values_match(cls, expected: Any, actual: Any) -> bool:
        if isinstance(expected, float) or isinstance(actual, float):
            try:
                return math.isclose(float(expected), float(actual), rel_tol=1e-6, abs_tol=1e-6)
            except (TypeError, ValueError):
                return False

        if isinstance(expected, list) and isinstance(actual, list):
            if len(expected) != len(actual):
                return False
            return all(cls._values_match(e, a) for e, a in zip(expected, actual))

        if isinstance(expected, tuple) and isinstance(actual, (list, tuple)):
            if len(expected) != len(actual):
                return False
            return all(cls._values_match(e, a) for e, a in zip(expected, actual))

        if isinstance(expected, dict) and isinstance(actual, dict):
            if expected.keys() != actual.keys():
                return False
            return all(cls._values_match(expected[key], actual[key]) for key in expected)

        return expected == actual

    @staticmethod
    def _text_outputs_match(expected: str, actual: str) -> bool:
        return expected.strip() == actual.strip()


class LiveBenchReasoningBenchmark(_BaseLiveBenchBenchmark):
    """Evaluate LiveBench reasoning tasks with structured answers."""

    name = "livebench-reasoning"
    description = "Logic and reasoning questions from LiveBench with string-based scoring."
    dataset_name = "livebench/reasoning"

    def _convert_row(
        self,
        row: Mapping[str, Any],
        index: int,
    ) -> _ConvertedRow:
        question_id = str(row["question_id"])
        prompt, extra_messages = _prepare_prompt(row)

        release_dt: datetime | None = row.get("livebench_release_date")
        release_key = self._format_release_key(release_dt)

        ground_truth = str(row.get("ground_truth", "")).strip()
        metadata = {
            "question_id": question_id,
            "category": row.get("category"),
            "task": row.get("task"),
            "level": row.get("level"),
            "livebench_release_date": self._format_datetime(release_dt),
            "livebench_removal_date": self._format_datetime(row.get("livebench_removal_date")),
        }

        example = BenchmarkExample(
            id=f"livebench-reasoning-{question_id}",
            prompt=prompt,
            target=ground_truth,
            metadata=metadata,
            system_prompt=None,
            extra_messages=extra_messages,
        )

        return _ConvertedRow(
            example=example,
            release_key=release_key,
            release_date=release_dt,
            extra_data=None,
        )

    def is_correct(
        self,
        example: BenchmarkExample,
        response_text: str,
        response: object,
    ) -> tuple[bool, Mapping[str, Any]]:
        expected_answers = self._parse_expected_answers(example.target)
        solution_block = self._extract_solution_block(response_text)

        details: dict[str, Any] = {
            "expected_answers": expected_answers,
        }

        if solution_block is None:
            details["error"] = "no solution block found"
            return False, details

        predicted_answers = self._split_solution_answers(solution_block)
        details["solution_block"] = solution_block
        details["predicted_answers"] = predicted_answers

        if not predicted_answers:
            details["error"] = "no answers found in solution block"
            return False, details

        expected_normalised = [self._normalise_answer(value) for value in expected_answers]
        predicted_normalised = [self._normalise_answer(value) for value in predicted_answers]

        details["expected_normalised"] = expected_normalised
        details["predicted_normalised"] = predicted_normalised

        if len(predicted_normalised) != len(expected_normalised):
            details["error"] = "incorrect number of answers"
            details["expected_count"] = len(expected_normalised)
            details["predicted_count"] = len(predicted_normalised)
            return False, details

        for index, (expected_value, predicted_value) in enumerate(
            zip(expected_normalised, predicted_normalised)
        ):
            if expected_value != predicted_value:
                details["error"] = "answer mismatch"
                details["mismatch_index"] = index
                details["expected_value"] = expected_value
                details["predicted_value"] = predicted_value
                return False, details

        return True, details

    @staticmethod
    def _parse_expected_answers(target: Any) -> list[str]:
        if target is None:
            return []
        if isinstance(target, str):
            parts = [segment.strip() for segment in target.split(",")]
        else:
            parts = [str(target).strip()]
        return [part for part in parts if part]

    @staticmethod
    def _extract_solution_block(response_text: str) -> str | None:
        matches = _SOLUTION_TAG_RE.findall(response_text)
        if matches:
            return matches[-1].strip()
        return None

    @staticmethod
    def _split_solution_answers(solution_block: str) -> list[str]:
        cleaned = solution_block.strip()
        if not cleaned:
            return []

        cleaned = re.sub(r"</?solution>", "", cleaned, flags=re.IGNORECASE)

        if "," in cleaned:
            parts = cleaned.split(",")
        else:
            lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
            if len(lines) > 1:
                parts = lines
            else:
                parts = re.split(r"\s+", cleaned)

        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _normalise_answer(value: str) -> str:
        cleaned = value.strip().lower()
        cleaned = cleaned.strip(".,;:!?")
        cleaned = cleaned.replace("-", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"^[\"'`]+|[\"'`]+$", "", cleaned)
        cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned)
        cleaned = re.sub(r"^(position|person)\s+", "", cleaned)
        cleaned = re.sub(r"^at\s+", "", cleaned)

        match = re.fullmatch(r"(-?\d+)(?:st|nd|rd|th)?", cleaned)
        if match:
            cleaned = match.group(1)

        if cleaned in _BOOLEAN_NORMALISATIONS:
            cleaned = _BOOLEAN_NORMALISATIONS[cleaned]
        if cleaned in _ORDINAL_WORDS:
            cleaned = _ORDINAL_WORDS[cleaned]
        if cleaned in _NUMBER_WORDS:
            cleaned = _NUMBER_WORDS[cleaned]

        cleaned = re.sub(r"^(position|person)\s+", "", cleaned)
        cleaned = cleaned.strip()
        return cleaned


class LiveBenchMathBenchmark(_BaseLiveBenchBenchmark):
    """Evaluate LiveBench math reasoning tasks using expression identifiers."""

    name = "livebench-math"
    description = "Mathematical reasoning tasks from LiveBench scored via expression IDs."
    dataset_name = "livebench/math"

    def _convert_row(
        self,
        row: Mapping[str, Any],
        index: int,
    ) -> _ConvertedRow:
        question_id = str(row["question_id"])
        prompt, extra_messages = _prepare_prompt(row)

        release_dt: datetime | None = row.get("livebench_release_date")
        release_key = self._format_release_key(release_dt)

        ground_truth = str(row.get("ground_truth", "")).strip()
        metadata = {
            "question_id": question_id,
            "category": row.get("category"),
            "task": row.get("task"),
            "subtask": row.get("subtask"),
            "year": row.get("year"),
            "hardness": row.get("hardness"),
            "expressions": row.get("expressions"),
            "livebench_release_date": self._format_datetime(release_dt),
            "livebench_removal_date": self._format_datetime(row.get("livebench_removal_date")),
        }

        example = BenchmarkExample(
            id=f"livebench-math-{question_id}",
            prompt=prompt,
            target=ground_truth,
            metadata=metadata,
            system_prompt=None,
            extra_messages=extra_messages,
        )

        return _ConvertedRow(
            example=example,
            release_key=release_key,
            release_date=release_dt,
            extra_data=None,
        )

    def is_correct(
        self,
        example: BenchmarkExample,
        response_text: str,
        response: object,
    ) -> tuple[bool, Mapping[str, Any]]:
        expected_numbers = self._extract_numbers(example.target)
        answer_line = self._extract_answer_line(response_text)

        details: dict[str, Any] = {
            "expected_numbers": expected_numbers,
        }

        if answer_line is None:
            details["error"] = "answer line not found"
            return False, details

        predicted_numbers = self._extract_numbers(answer_line)
        details["answer_line"] = answer_line
        details["predicted_numbers"] = predicted_numbers

        if not predicted_numbers:
            details["error"] = "no numeric answers found"
            return False, details

        if len(predicted_numbers) != len(expected_numbers):
            details["error"] = "incorrect number of answers"
            details["expected_count"] = len(expected_numbers)
            details["predicted_count"] = len(predicted_numbers)
            return False, details

        for index, (expected_value, predicted_value) in enumerate(
            zip(expected_numbers, predicted_numbers)
        ):
            if expected_value != predicted_value:
                details["error"] = "answer mismatch"
                details["mismatch_index"] = index
                details["expected_value"] = expected_value
                details["predicted_value"] = predicted_value
                return False, details

        return True, details

    @staticmethod
    def _extract_numbers(value: Any) -> list[str]:
        if value is None:
            return []
        text = str(value)
        return re.findall(r"-?\d+", text)

    @staticmethod
    def _extract_answer_line(response_text: str) -> str | None:
        lines = response_text.splitlines()
        answer_value: str | None = None

        for index, line in enumerate(lines):
            match = _ANSWER_LINE_RE.search(line)
            if match:
                candidate = match.group(1).strip()
                if candidate:
                    answer_value = candidate
                elif index + 1 < len(lines):
                    next_line = lines[index + 1].strip()
                    if next_line:
                        answer_value = next_line
                continue

            if line.strip().lower() == "answer" and index + 1 < len(lines):
                next_line = lines[index + 1].strip()
                if next_line:
                    answer_value = next_line

        if answer_value is not None and answer_value.strip():
            return answer_value.strip()

        matches = _ANSWER_LINE_RE.findall(response_text)
        if matches:
            candidate = matches[-1].strip()
            if candidate:
                return candidate

        return None
