"""Integration tests that exercise the real provider APIs when credentials exist."""

from __future__ import annotations

import os

import pytest

from successat.llm.clients import BaseLLMClient, OpenAIClient, OpenRouterClient

pytestmark = pytest.mark.integtest


def _client_from_env(client_cls: type[BaseLLMClient]) -> BaseLLMClient:
    """Create a client instance, skipping the test if credentials are absent."""

    try:
        return client_cls.from_env()
    except EnvironmentError as exc:  # pragma: no cover - exercised when keys absent
        pytest.skip(str(exc))


def _resolve_model(env_var: str, default: str) -> str:
    override = os.getenv(env_var)
    return override or default


def test_openai_client_live_chat_completion() -> None:
    client = _client_from_env(OpenAIClient)

    try:
        response = client.chat(
            "Reply with a short acknowledgement of success.",
            model=_resolve_model("OPENAI_INTEG_MODEL", client.model),
        )
    finally:
        client.client.close()

    assert isinstance(response, str)
    assert response.strip()


def test_openrouter_client_live_chat_completion() -> None:
    client = _client_from_env(OpenRouterClient)

    try:
        response = client.chat(
            "Respond with a concise confirmation that OpenRouter is reachable.",
            model=_resolve_model("OPENROUTER_INTEG_MODEL", client.model),
        )
    finally:
        client.client.close()

    assert isinstance(response, str)
    assert response.strip()
