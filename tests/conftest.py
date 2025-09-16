"""Shared pytest fixtures for benchmark tests."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, List, Sequence

import pytest


@dataclass
class FakeLLMResponse:
    """Minimal stand-in for an OpenAI chat response."""

    content: str

    def __post_init__(self) -> None:
        message = SimpleNamespace(content=self.content)
        self.choices = [SimpleNamespace(message=message)]


class FakeLLMClient:
    """Test double that records prompts and returns canned responses."""

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses: List[str] = list(responses)
        self.model = "fake-test-model"
        self.calls: List[dict[str, Any]] = []

    def chat_completion(self, prompt: str, **kwargs: Any) -> FakeLLMResponse:
        if not self._responses:
            msg = "No more fake responses configured."
            raise AssertionError(msg)

        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        return FakeLLMResponse(self._responses.pop(0))


@pytest.fixture
def fake_llm_client() -> Callable[[Sequence[str] | str], FakeLLMClient]:
    """Factory fixture returning a `FakeLLMClient` instance."""

    def _factory(responses: Sequence[str] | str) -> FakeLLMClient:
        if isinstance(responses, str):
            responses = [responses]
        return FakeLLMClient(responses)

    return _factory

