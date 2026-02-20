"""Provider adapters for LLM APIs."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

from .errors import ConfigError, ProviderError

DEFAULT_MODELS: dict[str, str] = {"anthropic": "claude-sonnet-4-20250514", "openai": "gpt-4o-mini"}
MODEL_PREFIXES: dict[str, str] = {"claude": "anthropic", "gpt": "openai", "o1": "openai"}


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        timeout: int = 60,
    ) -> str:
        """Send messages to the LLM and return response text."""


class AnthropicProvider(BaseProvider):
    """Provider adapter for Anthropic's Claude API."""

    def __init__(self, model: str | None = None) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ConfigError("ANTHROPIC_API_KEY not set. Get key at https://console.anthropic.com/")
        try:
            import anthropic
        except ImportError as e:
            raise ConfigError("anthropic package not installed. Run: pip install anthropic") from e
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model or DEFAULT_MODELS["anthropic"]

    def complete(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        timeout: int = 60,
    ) -> str:
        try:
            kwargs: dict[str, object] = {
                "model": self._model,
                "max_tokens": 4096,
                "messages": messages,
            }
            if system_prompt:
                kwargs["system"] = system_prompt
            response = self._client.messages.create(**kwargs, timeout=timeout)  # type: ignore[call-overload]
            return str(response.content[0].text)
        except Exception as e:
            raise ProviderError(f"Anthropic API error: {e}") from e


class OpenAIProvider(BaseProvider):
    """Provider adapter for OpenAI's API."""

    def __init__(self, model: str | None = None) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ConfigError("OPENAI_API_KEY not set. Get key at https://platform.openai.com/api-keys")
        try:
            import openai
        except ImportError as e:
            raise ConfigError("openai package not installed. Run: pip install openai") from e
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model or DEFAULT_MODELS["openai"]

    def complete(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        timeout: int = 60,
    ) -> str:
        try:
            all_messages: list[dict[str, str]] = []
            if system_prompt:
                all_messages.append({"role": "system", "content": system_prompt})
            all_messages.extend(messages)
            response = self._client.chat.completions.create(
                model=self._model, messages=all_messages, timeout=timeout  # type: ignore[arg-type]
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise ProviderError(f"OpenAI API error: {e}") from e


def get_provider(model: str | None = None, provider: str | None = None) -> BaseProvider:
    """Get provider based on config. Priority: explicit > model prefix > env vars."""
    if provider:
        p = provider.lower()
        if p == "anthropic":
            return AnthropicProvider(model)
        if p == "openai":
            return OpenAIProvider(model)
        raise ConfigError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")
    if model:
        for prefix, prov in MODEL_PREFIXES.items():
            if model.lower().startswith(prefix):
                return AnthropicProvider(model) if prov == "anthropic" else OpenAIProvider(model)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicProvider(model)
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIProvider(model)
    raise ConfigError("No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")
