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


@dataclass
class _LiveBenchExampleData:
    """Additional information required to evaluate LiveBench examples."""

    public_tests_json: str
    private_tests_blob: str | None
    test_mode: str
    method_name: str | None
    parameter_count: int
    starter_code: str


class LiveBenchCodingBenchmark(Benchmark):
    """Evaluate LiveBench coding tasks using public and private test suites."""

    name = "livebench-coding"
    description = "Python coding problems from LiveBench with execution-based scoring."
    dataset_name = "livebench/coding"
    default_split = "latest"

    def __init__(self, client) -> None:
        super().__init__(client)
        self._examples_by_split: MutableMapping[str, List[BenchmarkExample]] = {}
        self._split_order: List[str] | None = None
        self._example_data: dict[str, _LiveBenchExampleData] = {}
        self._prepared = False

    # Benchmark API ------------------------------------------------------
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

    def is_correct(self, example: BenchmarkExample, response_text: str, response: object):
        example_data = self._example_data.get(example.id)
        if example_data is None:  # pragma: no cover - defensive guard
            return False, {"error": f"no evaluation data for example {example.id}"}

        candidate_code = self._strip_code_fence(response_text)
        if not candidate_code.strip():
            return False, {"error": "model response did not contain Python code"}

        try:
            compiled = compile(candidate_code, "<candidate>", "exec")
        except SyntaxError as exc:
            return False, {"error": f"candidate code failed to compile: {exc}"}

        try:
            tests = self._load_tests(example_data)
        except ValueError as exc:
            return False, {"error": str(exc)}

        if not tests:
            return False, {"error": "benchmark example does not define any tests"}

        if example_data.test_mode == "stdin":
            return self._run_stdin_tests(compiled, tests)

        return self._run_functional_tests(compiled, tests, example_data)

    # Internal helpers ---------------------------------------------------
    def _ensure_prepared(self) -> None:
        if self._prepared:
            return

        dataset = load_dataset(self.dataset_name, split="test")
        self._prepare_examples(dataset)
        self._prepared = True

    def _prepare_examples(self, rows: Iterable[Mapping[str, Any]]) -> None:
        release_groups: MutableMapping[str, List[BenchmarkExample]] = defaultdict(list)
        release_dates: MutableMapping[str, datetime] = {}
        all_examples: List[BenchmarkExample] = []

        for index, row in enumerate(rows):
            example, example_data, release_key, release_date = self._convert_row(row, index)
            self._example_data[example.id] = example_data
            release_groups[release_key].append(example)
            if release_date is not None:
                release_dates[release_key] = release_date
            all_examples.append(example)

        if not all_examples:
            self._examples_by_split = {self.default_split: []}
            self._split_order = [self.default_split]
            return

        self._examples_by_split = {"all": all_examples}

        for key, examples in release_groups.items():
            self._examples_by_split[f"release-{key}"] = examples

        latest_key = None
        if release_dates:
            latest_key = max(release_dates, key=lambda item: (release_dates[item], item))
            self._examples_by_split[self.default_split] = list(release_groups[latest_key])
        else:
            latest_key = next(iter(release_groups))
            self._examples_by_split[self.default_split] = list(release_groups[latest_key])

        history_examples: List[BenchmarkExample] = []
        for key, examples in release_groups.items():
            if key != latest_key:
                history_examples.extend(examples)
        if history_examples:
            self._examples_by_split["history"] = history_examples

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
    ) -> tuple[BenchmarkExample, _LiveBenchExampleData, str, datetime | None]:
        question_id = str(row["question_id"])
        turns = [turn for turn in row.get("turns", []) if isinstance(turn, str) and turn.strip()]
        prompt = turns[-1] if turns else row.get("original_json", {}).get("question_content", "")
        prompt = prompt.strip()
        extra_messages: List[Mapping[str, str]] | None = None
        if len(turns) > 1:
            extra_messages = [{"role": "user", "content": text} for text in turns[:-1]]

        release_dt: datetime | None = row.get("livebench_release_date")
        release_key = self._format_release_key(release_dt)

        public_tests_json = row.get("public_test_cases", "") or "[]"
        private_blob = row.get("private_test_cases") or None

        starter_code = row.get("original_json", {}).get("starter_code", "") or ""
        method_name, params = self._parse_signature(starter_code)

        tests_preview = self._preview_test_types(public_tests_json, private_blob)
        target = "pass"
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
            target=target,
            metadata=metadata,
            system_prompt=None,
            extra_messages=extra_messages,
        )

        test_mode = tests_preview["mode"]
        example_data = _LiveBenchExampleData(
            public_tests_json=public_tests_json,
            private_tests_blob=private_blob,
            test_mode=test_mode,
            method_name=method_name,
            parameter_count=len(params),
            starter_code=starter_code,
        )

        return example, example_data, release_key, release_dt

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

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        fenced_blocks = _CODE_BLOCK_RE.findall(text)
        if fenced_blocks:
            return fenced_blocks[-1].strip()

        match = _SOLUTION_CLASS_RE.search(text)
        if match:
            return text[match.start() :].strip()

        return _CODE_FENCE_RE.sub("", text).strip()

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
        details: dict[str, Any] = {"tests_run": 0, "test_mode": "functional"}

        for index, test in enumerate(tests):
            parsed_input = self._parse_value(test.get("input", ""))
            args = self._coerce_arguments(parsed_input, example_data.parameter_count)
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

            stdin = io.StringIO(input_payload)
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
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                return raw

    @staticmethod
    def _coerce_arguments(parsed_input: Any, parameter_count: int) -> tuple[Any, ...]:
        if parameter_count == 0:
            return ()
        if parameter_count == 1:
            return (parsed_input,)
        if isinstance(parsed_input, (list, tuple)) and len(parsed_input) == parameter_count:
            return tuple(parsed_input)
        if isinstance(parsed_input, (list, tuple)):
            return tuple(parsed_input)
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

