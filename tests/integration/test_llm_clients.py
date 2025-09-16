"""Integration tests for the concrete LLM clients."""

from __future__ import annotations

import json
from typing import Any, Dict

import httpx

from successat.llm.clients import OpenAIClient, OpenRouterClient


def _mock_chat_response(message: str, capture: Dict[str, Any]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        capture["payload"] = payload
        capture["headers"] = dict(request.headers)
        return httpx.Response(200, json={"choices": [{"message": {"content": message}}]})

    return httpx.MockTransport(handler)


def test_openai_client_chat_completion_round_trip() -> None:
    captured: Dict[str, Any] = {}
    transport = _mock_chat_response("Model reply", captured)
    http_client = httpx.Client(transport=transport, base_url="https://api.openai.com/v1")
    client = OpenAIClient(api_key="test-key", client_kwargs={"http_client": http_client})

    try:
        content = client.chat(
            "Hello there",
            system_prompt="system message",
            extra_messages=[{"role": "assistant", "content": "context"}],
        )
    finally:
        client.client.close()

    assert content == "Model reply"
    payload = captured["payload"]
    assert payload["model"] == "gpt-5-nano"
    assert payload["messages"][0] == {"role": "system", "content": "system message"}
    assert payload["messages"][1] == {"role": "user", "content": "Hello there"}
    assert payload["messages"][2] == {"role": "assistant", "content": "context"}
    headers = captured["headers"]
    assert headers.get("user-agent") == "successat"


def test_openrouter_client_includes_custom_headers_and_model_override() -> None:
    captured: Dict[str, Any] = {}
    transport = _mock_chat_response("Router reply", captured)
    http_client = httpx.Client(transport=transport, base_url="https://openrouter.ai/api/v1")
    client = OpenRouterClient(
        api_key="router-key",
        client_kwargs={"http_client": http_client},
    )

    try:
        content = client.chat("Prompt", model="custom-model")
    finally:
        client.client.close()

    assert content == "Router reply"
    payload = captured["payload"]
    assert payload["model"] == "custom-model"
    headers = captured["headers"]
    assert headers.get("http-referer") == "https://github.com/successat/successat"
    assert headers.get("x-title") == "successat"
    assert headers.get("user-agent") == "successat"
