"""Client implementations for interacting with LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Iterable, Mapping, Sequence

from openai import OpenAI


class BaseLLMClient(ABC):
    """Abstract base class for LLM client implementations."""

    env_var_name: ClassVar[str]
    default_model: ClassVar[str] = "gpt-5-nano"
    default_app_name: ClassVar[str] = "successat"

    def __init__(
        self,
        *,
        api_key: str,
        model: str | None = None,
        app_name: str | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not api_key:
            msg = "An API key is required to create an LLM client."
            raise ValueError(msg)

        self.api_key = api_key
        self.model = model or self.default_model
        self.app_name = app_name or self.default_app_name
        self._client_kwargs: Dict[str, Any] = dict(client_kwargs or {})
        self._client = self._create_client()

    @property
    def client(self) -> OpenAI:
        """Return the underlying OpenAI client instance."""

        return self._client

    def chat(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        extra_messages: Sequence[Mapping[str, str]] | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute a chat completion request and return the model response."""

        messages: list[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        if extra_messages:
            messages.extend({"role": msg["role"], "content": msg["content"]} for msg in extra_messages)

        result = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            **kwargs,
        )
        first_choice = result.choices[0]
        message_content = getattr(first_choice.message, "content", "")

        if isinstance(message_content, str):
            return message_content

        if isinstance(message_content, Iterable):
            parts = []
            for item in message_content:  # type: ignore[not-an-iterable]
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
            if parts:
                return "".join(parts)

        return ""

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        app_name: str | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> "BaseLLMClient":
        """Instantiate the client using credentials sourced from the environment."""

        import os

        api_key = os.getenv(cls.env_var_name)
        if not api_key:
            msg = f"Environment variable {cls.env_var_name} is required."
            raise EnvironmentError(msg)

        return cls(
            api_key=api_key,
            model=model,
            app_name=app_name,
            client_kwargs=client_kwargs,
        )

    @abstractmethod
    def _create_client(self) -> OpenAI:
        """Create and return the configured OpenAI client."""


class OpenAIClient(BaseLLMClient):
    """LLM client implementation backed by OpenAI's API."""

    env_var_name: ClassVar[str] = "OPENAI_API_KEY"

    def _create_client(self) -> OpenAI:  # pragma: no cover - exercised in integration tests
        headers = self._build_default_headers()
        client_kwargs = self._pop_client_kwargs()
        client_kwargs.setdefault("default_headers", headers)

        return OpenAI(api_key=self.api_key, **client_kwargs)

    def _build_default_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"User-Agent": self.app_name}
        existing: Mapping[str, str] | None = None
        if "default_headers" in self._client_kwargs:
            raw_headers = self._client_kwargs["default_headers"]
            if isinstance(raw_headers, Mapping):
                existing = raw_headers
        if existing:
            headers = {**existing, **headers}
        return headers

    def _pop_client_kwargs(self) -> Dict[str, Any]:
        kwargs = dict(self._client_kwargs)
        kwargs.pop("default_headers", None)
        return kwargs


class OpenRouterClient(BaseLLMClient):
    """LLM client implementation for the OpenRouter API."""

    env_var_name: ClassVar[str] = "OPENROUTER_API_KEY"
    base_url: ClassVar[str] = "https://openrouter.ai/api/v1"
    default_referer: ClassVar[str] = "https://github.com/successat/successat"

    def __init__(
        self,
        *,
        api_key: str,
        model: str | None = None,
        app_name: str | None = None,
        referer: str | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.referer = referer or self.default_referer
        super().__init__(
            api_key=api_key,
            model=model,
            app_name=app_name,
            client_kwargs=client_kwargs,
        )

    def _create_client(self) -> OpenAI:  # pragma: no cover - exercised in integration tests
        headers = self._build_default_headers()
        client_kwargs = dict(self._client_kwargs)
        client_kwargs.pop("default_headers", None)
        base_url = client_kwargs.pop("base_url", self.base_url)

        return OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            default_headers=headers,
            **client_kwargs,
        )

    def _build_default_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "HTTP-Referer": self.referer,
            "X-Title": self.app_name,
            "User-Agent": self.app_name,
        }
        if "default_headers" in self._client_kwargs and isinstance(
            self._client_kwargs["default_headers"], Mapping
        ):
            headers = {**self._client_kwargs["default_headers"], **headers}
        return headers
