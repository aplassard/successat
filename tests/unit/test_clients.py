"""Unit tests for the LLM client implementations."""

import pytest

from successat.llm.clients import OpenAIClient, OpenRouterClient


def test_openai_client_requires_api_key() -> None:
    with pytest.raises(ValueError):
        OpenAIClient(api_key="")


def test_openai_client_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = OpenAIClient.from_env()

    try:
        assert client.api_key == "test-key"
        assert client.app_name == "successat"
        assert client.model == "gpt-5-nano"
        assert client.client.timeout == 60.0
        assert client.client.max_retries == 3
    finally:
        client.client.close()


def test_openai_client_from_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(EnvironmentError):
        OpenAIClient.from_env()


def test_openrouter_client_default_configuration() -> None:
    client = OpenRouterClient(api_key="router-key")

    try:
        headers = client.client.default_headers
        assert headers["HTTP-Referer"] == "https://github.com/successat/successat"
        assert headers["X-Title"] == "successat"
        assert headers["User-Agent"] == "successat"
        assert str(client.client._client.base_url) == "https://openrouter.ai/api/v1/"
        assert client.client.timeout == 60.0
        assert client.client.max_retries == 3
    finally:
        client.client.close()


def test_openai_client_allows_overriding_timeout_and_retries() -> None:
    client = OpenAIClient(api_key="key", client_kwargs={"timeout": 10.0, "max_retries": 5})

    try:
        assert client.client.timeout == 10.0
        assert client.client.max_retries == 5
    finally:
        client.client.close()


def test_openrouter_client_allows_overriding_timeout_and_retries() -> None:
    client = OpenRouterClient(
        api_key="router-key",
        client_kwargs={"timeout": 15.0, "max_retries": 4},
    )

    try:
        assert client.client.timeout == 15.0
        assert client.client.max_retries == 4
    finally:
        client.client.close()


def test_openrouter_client_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")

    client = OpenRouterClient.from_env()

    try:
        assert client.api_key == "router-key"
        assert client.model == "gpt-5-nano"
        assert client.app_name == "successat"
    finally:
        client.client.close()
