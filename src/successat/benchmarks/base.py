"""Base abstractions for reusable LLM benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Protocol, Sequence


@dataclass(frozen=True)
class BenchmarkExample:
    """A single example within a benchmark split."""

    id: str
    prompt: str
    target: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)
    system_prompt: str | None = None
    extra_messages: Sequence[Mapping[str, str]] | None = None


@dataclass(slots=True)
class BenchmarkResult:
    """Result of running a benchmark example."""

    benchmark: str
    model: str
    split: str
    example_id: str
    prompt: str
    response: Any
    response_text: str
    correct: bool
    metadata: Dict[str, Any]
    time_to_first_token: float | None = None
    total_time: float | None = None


class SupportsChatCompletion(Protocol):
    """Protocol describing the chat interface required by benchmarks."""

    model: str

    def chat_completion(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        extra_messages: Sequence[Mapping[str, str]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return the raw response from an LLM chat completion."""


class BenchmarkRegistry:
    """Registry for benchmark implementations."""

    def __init__(self) -> None:
        self._benchmarks: Dict[str, type["Benchmark"]] = {}

    def register(self, benchmark_cls: type["Benchmark"]) -> None:
        """Register a benchmark implementation."""

        name = benchmark_cls.name.lower()
        if name in self._benchmarks:
            msg = f"Benchmark '{benchmark_cls.name}' is already registered."
            raise ValueError(msg)
        self._benchmarks[name] = benchmark_cls

    def get(self, name: str) -> type["Benchmark"]:
        """Retrieve a registered benchmark by name."""

        try:
            return self._benchmarks[name.lower()]
        except KeyError as exc:  # pragma: no cover - sanity guard
            msg = f"Benchmark '{name}' is not registered. Available: {sorted(self._benchmarks)}"
            raise KeyError(msg) from exc

    def names(self) -> List[str]:
        """Return the list of registered benchmark names."""

        return sorted(self._benchmarks)


class Benchmark:
    """Base class for individual benchmark implementations."""

    name: str
    description: str = ""
    default_split: str = "test"

    def __init__(self, client: SupportsChatCompletion) -> None:
        self.client = client

    def available_splits(self) -> Sequence[str]:
        """Return the collection of supported split names for the benchmark."""

        return [self.default_split]

    # Public API ---------------------------------------------------------
    def run(
        self,
        *,
        identifier: int | str | None = None,
        split: str | None = None,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Run the benchmark example and return a detailed result."""

        chosen_split = split or self.default_split
        example = self._select_example(chosen_split, identifier)
        prompt = self.build_prompt(example)
        start_time = perf_counter()
        response = self._call_model(example, prompt, **kwargs)
        total_elapsed = perf_counter() - start_time
        response_text = self.extract_text(response)
        correct, details = self.is_correct(example, response_text, response)

        metadata: Dict[str, Any] = dict(example.metadata)
        metadata.setdefault("expected", example.target)
        if details:
            metadata["evaluation_details"] = details

        time_to_first_token = _extract_timing_value(response, "time_to_first_token")
        total_time = _extract_timing_value(response, "total_time")

        metadata_time_to_first = _coerce_float(metadata.get("time_to_first_token"))
        metadata_total_time = _coerce_float(metadata.get("total_time"))

        if time_to_first_token is None:
            time_to_first_token = metadata_time_to_first
        if total_time is None:
            total_time = metadata_total_time

        if time_to_first_token is None:
            time_to_first_token = total_elapsed
        if total_time is None:
            total_time = total_elapsed

        if _should_set_timing(metadata, "time_to_first_token"):
            metadata["time_to_first_token"] = time_to_first_token
        if _should_set_timing(metadata, "total_time"):
            metadata["total_time"] = total_time

        return BenchmarkResult(
            benchmark=self.name,
            model=self._model_name(),
            split=chosen_split,
            example_id=example.id,
            prompt=prompt,
            response=response,
            response_text=response_text,
            correct=correct,
            metadata=metadata,
            time_to_first_token=time_to_first_token,
            total_time=total_time,
        )

    # Hooks for subclasses -----------------------------------------------
    def examples_for_split(self, split: str) -> Sequence[BenchmarkExample]:
        """Return examples for the requested split."""

        raise NotImplementedError

    def build_prompt(self, example: BenchmarkExample) -> str:
        """Build the prompt used for the given example."""

        return example.prompt

    def chat_parameters(self, example: BenchmarkExample) -> Mapping[str, Any]:
        """Additional provider-specific parameters for the chat call."""

        return {}

    def extract_text(self, response: Any) -> str:
        """Extract the textual answer from the LLM response."""

        if isinstance(response, str):
            return response

        choices: Any
        if isinstance(response, Mapping):
            choices = response.get("choices")
        else:
            choices = getattr(response, "choices", None)

        if not choices:
            return str(response)

        first_choice = choices[0]
        message: Any
        if isinstance(first_choice, Mapping):
            message = first_choice.get("message")
        else:
            message = getattr(first_choice, "message", None)

        if message is None:
            return str(response)

        if isinstance(message, Mapping):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)

        if isinstance(content, str):
            return content

        if isinstance(content, Iterable):
            parts: List[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
            if parts:
                return "".join(parts)

        if content is None:
            return ""

        return str(content)

    def is_correct(
        self,
        example: BenchmarkExample,
        response_text: str,
        response: Any,
    ) -> tuple[bool, Mapping[str, Any]]:
        """Determine if the response is correct."""

        expected = example.target
        is_match = str(expected).strip() == response_text.strip()
        return is_match, {}

    # Internal helpers ---------------------------------------------------
    def _select_example(
        self,
        split: str,
        identifier: int | str | None,
    ) -> BenchmarkExample:
        examples = list(self.examples_for_split(split))
        if not examples:
            msg = f"Benchmark '{self.name}' has no examples for split '{split}'."
            raise ValueError(msg)

        if identifier is None:
            return examples[0]

        if isinstance(identifier, int):
            try:
                return examples[identifier]
            except IndexError as exc:
                msg = f"Example index {identifier} out of range for split '{split}'."
                raise ValueError(msg) from exc

        for example in examples:
            if example.id == identifier:
                return example

        msg = f"Example '{identifier}' not found in split '{split}'."
        raise ValueError(msg)

    def _call_model(
        self,
        example: BenchmarkExample,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        params: MutableMapping[str, Any] = dict(self.chat_parameters(example))
        params.update(kwargs)
        return self.client.chat_completion(
            prompt,
            system_prompt=example.system_prompt,
            extra_messages=example.extra_messages,
            **params,
        )

    def _model_name(self) -> str:
        return getattr(self.client, "model", "unknown")


def _coerce_float(value: Any) -> float | None:
    """Attempt to convert ``value`` into a floating point number."""

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None

    return None


def _should_set_timing(metadata: MutableMapping[str, Any], key: str) -> bool:
    """Return ``True`` if the timing metadata for ``key`` should be set."""

    if key not in metadata:
        return True

    return metadata[key] is None


def _extract_timing_value(response: Any, key: str) -> float | None:
    """Inspect ``response`` for a timing attribute named ``key``."""

    to_inspect: list[Any] = [response]
    seen: set[int] = set()

    while to_inspect:
        current = to_inspect.pop()
        if current is None:
            continue

        if isinstance(current, (str, bytes, bytearray)):
            continue

        identifier = id(current)
        if identifier in seen:
            continue
        seen.add(identifier)

        if isinstance(current, Mapping):
            if key in current:
                coerced = _coerce_float(current[key])
                if coerced is not None:
                    return coerced

            for nested_key in ("usage", "timings", "meta", "metadata"):
                nested = current.get(nested_key)
                if nested is not None:
                    to_inspect.append(nested)

        value = getattr(current, key, None)
        if value is not None:
            coerced = _coerce_float(value)
            if coerced is not None:
                return coerced

        for attr in ("usage", "timings", "meta", "metadata", "model_extra"):
            nested = getattr(current, attr, None)
            if nested is not None:
                to_inspect.append(nested)

    return None

