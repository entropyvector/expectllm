"""Tests for expectllm providers."""
import pytest
import sys
from unittest.mock import patch, MagicMock
from expectllm import ConfigError, ProviderError
from expectllm.providers import (
    get_provider,
    AnthropicProvider,
    OpenAIProvider,
    DEFAULT_MODELS,
    MODEL_PREFIXES,
)


def create_mock_anthropic_module():
    """Create a mock anthropic module."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Mock response")]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    mock_module = MagicMock()
    mock_module.Anthropic.return_value = mock_client
    return mock_module, mock_client


def create_mock_openai_module():
    """Create a mock openai module."""
    mock_message = MagicMock()
    mock_message.content = "Mock response"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_module = MagicMock()
    mock_module.OpenAI.return_value = mock_client
    return mock_module, mock_client


class TestGetProvider:
    """Tests for get_provider function."""

    def test_get_provider_detects_anthropic_from_env(self, mock_anthropic_env):
        """get_provider detects Anthropic from ANTHROPIC_API_KEY."""
        mock_module, _ = create_mock_anthropic_module()
        with patch.dict(sys.modules, {"anthropic": mock_module}):
            provider = get_provider()
        assert isinstance(provider, AnthropicProvider)

    def test_get_provider_detects_openai_from_env(self, mock_openai_env):
        """get_provider detects OpenAI from OPENAI_API_KEY."""
        mock_module, _ = create_mock_openai_module()
        with patch.dict(sys.modules, {"openai": mock_module}):
            provider = get_provider()
        assert isinstance(provider, OpenAIProvider)

    def test_get_provider_prefers_anthropic(self, monkeypatch):
        """get_provider prefers Anthropic when both keys present."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        mock_module, _ = create_mock_anthropic_module()
        with patch.dict(sys.modules, {"anthropic": mock_module}):
            provider = get_provider()
        assert isinstance(provider, AnthropicProvider)

    def test_get_provider_explicit_override(self, mock_openai_env):
        """get_provider respects explicit provider parameter."""
        mock_module, _ = create_mock_openai_module()
        with patch.dict(sys.modules, {"openai": mock_module}):
            provider = get_provider(provider="openai")
        assert isinstance(provider, OpenAIProvider)

    def test_get_provider_detects_from_model_prefix(self, mock_openai_env):
        """get_provider detects provider from model name prefix."""
        mock_module, _ = create_mock_openai_module()
        with patch.dict(sys.modules, {"openai": mock_module}):
            provider = get_provider(model="gpt-4")
        assert isinstance(provider, OpenAIProvider)

    def test_get_provider_claude_prefix(self, mock_anthropic_env):
        """get_provider detects Anthropic from claude- prefix."""
        mock_module, _ = create_mock_anthropic_module()
        with patch.dict(sys.modules, {"anthropic": mock_module}):
            provider = get_provider(model="claude-3-opus")
        assert isinstance(provider, AnthropicProvider)

    def test_get_provider_raises_without_api_key(self, no_api_keys):
        """get_provider raises ConfigError when no API key found."""
        with pytest.raises(ConfigError) as exc_info:
            get_provider()
        assert "No API key found" in str(exc_info.value)

    def test_get_provider_raises_for_unknown_provider(self, mock_anthropic_env):
        """get_provider raises ConfigError for unknown provider."""
        with pytest.raises(ConfigError) as exc_info:
            get_provider(provider="unknown")
        assert "Unknown provider" in str(exc_info.value)


class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    def test_anthropic_provider_requires_api_key(self, no_api_keys):
        """AnthropicProvider raises ConfigError without API key."""
        with pytest.raises(ConfigError) as exc_info:
            AnthropicProvider()
        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_anthropic_provider_uses_default_model(self, mock_anthropic_env):
        """AnthropicProvider uses default model when not specified."""
        mock_module, _ = create_mock_anthropic_module()
        with patch.dict(sys.modules, {"anthropic": mock_module}):
            provider = AnthropicProvider()
        assert provider._model == DEFAULT_MODELS["anthropic"]

    def test_anthropic_provider_uses_custom_model(self, mock_anthropic_env):
        """AnthropicProvider uses custom model when specified."""
        mock_module, _ = create_mock_anthropic_module()
        with patch.dict(sys.modules, {"anthropic": mock_module}):
            provider = AnthropicProvider(model="claude-3-opus")
        assert provider._model == "claude-3-opus"

    def test_anthropic_provider_complete_returns_string(self, mock_anthropic_env):
        """AnthropicProvider.complete returns string."""
        mock_module, _ = create_mock_anthropic_module()
        with patch.dict(sys.modules, {"anthropic": mock_module}):
            provider = AnthropicProvider()
            result = provider.complete([{"role": "user", "content": "Hello"}])
        assert isinstance(result, str)
        assert result == "Mock response"

    def test_anthropic_provider_wraps_errors(self, mock_anthropic_env):
        """AnthropicProvider wraps API errors in ProviderError."""
        mock_module, mock_client = create_mock_anthropic_module()
        mock_client.messages.create.side_effect = Exception("API error")
        with patch.dict(sys.modules, {"anthropic": mock_module}):
            provider = AnthropicProvider()
            with pytest.raises(ProviderError) as exc_info:
                provider.complete([{"role": "user", "content": "Hello"}])
        assert "Anthropic API error" in str(exc_info.value)


class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    def test_openai_provider_requires_api_key(self, no_api_keys):
        """OpenAIProvider raises ConfigError without API key."""
        with pytest.raises(ConfigError) as exc_info:
            OpenAIProvider()
        assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_openai_provider_uses_default_model(self, mock_openai_env):
        """OpenAIProvider uses default model when not specified."""
        mock_module, _ = create_mock_openai_module()
        with patch.dict(sys.modules, {"openai": mock_module}):
            provider = OpenAIProvider()
        assert provider._model == DEFAULT_MODELS["openai"]

    def test_openai_provider_complete_returns_string(self, mock_openai_env):
        """OpenAIProvider.complete returns string."""
        mock_module, _ = create_mock_openai_module()
        with patch.dict(sys.modules, {"openai": mock_module}):
            provider = OpenAIProvider()
            result = provider.complete([{"role": "user", "content": "Hello"}])
        assert isinstance(result, str)
        assert result == "Mock response"

    def test_openai_provider_includes_system_prompt(self, mock_openai_env):
        """OpenAIProvider includes system prompt as first message."""
        mock_module, mock_client = create_mock_openai_module()
        with patch.dict(sys.modules, {"openai": mock_module}):
            provider = OpenAIProvider()
            provider.complete(
                [{"role": "user", "content": "Hello"}],
                system_prompt="You are helpful"
            )
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful"

    def test_openai_provider_wraps_errors(self, mock_openai_env):
        """OpenAIProvider wraps API errors in ProviderError."""
        mock_module, mock_client = create_mock_openai_module()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        with patch.dict(sys.modules, {"openai": mock_module}):
            provider = OpenAIProvider()
            with pytest.raises(ProviderError) as exc_info:
                provider.complete([{"role": "user", "content": "Hello"}])
        assert "OpenAI API error" in str(exc_info.value)


class TestModelPrefixes:
    """Tests for model prefix detection."""

    def test_claude_prefix_maps_to_anthropic(self):
        """claude prefix maps to anthropic."""
        assert MODEL_PREFIXES["claude"] == "anthropic"

    def test_gpt_prefix_maps_to_openai(self):
        """gpt prefix maps to openai."""
        assert MODEL_PREFIXES["gpt"] == "openai"

    def test_o1_prefix_maps_to_openai(self):
        """o1 prefix maps to openai."""
        assert MODEL_PREFIXES["o1"] == "openai"


class TestProviderAdvanced:
    """Advanced provider tests (OAI-002 to OAI-008, ANT-002 to ANT-005)."""

    def test_openai_provider_custom_model(self, mock_openai_env):
        """OAI-004: OpenAI provider with custom model."""
        mock_module, _ = create_mock_openai_module()
        with patch.dict(sys.modules, {"openai": mock_module}):
            provider = OpenAIProvider(model="gpt-4-turbo")
        assert provider._model == "gpt-4-turbo"

    def test_anthropic_provider_passes_system_prompt(self, mock_anthropic_env):
        """ANT: Anthropic provider passes system prompt."""
        mock_module, mock_client = create_mock_anthropic_module()
        with patch.dict(sys.modules, {"anthropic": mock_module}):
            provider = AnthropicProvider()
            provider.complete(
                [{"role": "user", "content": "Hello"}],
                system_prompt="Be concise"
            )
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs.get("system") == "Be concise"

    def test_openai_provider_no_system_prompt(self, mock_openai_env):
        """OpenAI provider without system prompt."""
        mock_module, mock_client = create_mock_openai_module()
        with patch.dict(sys.modules, {"openai": mock_module}):
            provider = OpenAIProvider()
            provider.complete([{"role": "user", "content": "Hello"}])
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        # Should not have system message
        assert messages[0]["role"] == "user"

    def test_anthropic_provider_timeout_passed(self, mock_anthropic_env):
        """Anthropic provider passes timeout to API."""
        mock_module, mock_client = create_mock_anthropic_module()
        with patch.dict(sys.modules, {"anthropic": mock_module}):
            provider = AnthropicProvider()
            provider.complete(
                [{"role": "user", "content": "Hello"}],
                timeout=120
            )
        # Verify timeout was passed
        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs.get("timeout") == 120

    def test_openai_provider_timeout_passed(self, mock_openai_env):
        """OpenAI provider passes timeout to API."""
        mock_module, mock_client = create_mock_openai_module()
        with patch.dict(sys.modules, {"openai": mock_module}):
            provider = OpenAIProvider()
            provider.complete(
                [{"role": "user", "content": "Hello"}],
                timeout=90
            )
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs.get("timeout") == 90

    def test_openai_provider_handles_empty_response(self, mock_openai_env):
        """OAI-008: Handle empty/None response content."""
        mock_module, mock_client = create_mock_openai_module()
        # Simulate empty content
        mock_client.chat.completions.create.return_value.choices[0].message.content = None
        with patch.dict(sys.modules, {"openai": mock_module}):
            provider = OpenAIProvider()
            result = provider.complete([{"role": "user", "content": "Hello"}])
        # Should return empty string, not None
        assert result == ""

    def test_anthropic_provider_missing_package(self, mock_anthropic_env):
        """ANT: ConfigError when anthropic package not installed."""
        with patch.dict(sys.modules, {"anthropic": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                with pytest.raises(ConfigError) as exc_info:
                    AnthropicProvider()
        assert "anthropic package not installed" in str(exc_info.value)

    def test_openai_provider_missing_package(self, mock_openai_env):
        """OAI: ConfigError when openai package not installed."""
        with patch.dict(sys.modules, {"openai": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                with pytest.raises(ConfigError) as exc_info:
                    OpenAIProvider()
        assert "openai package not installed" in str(exc_info.value)


class TestProviderErrorHandling:
    """Error handling tests for providers."""

    def test_anthropic_authentication_error(self, mock_anthropic_env):
        """ANT-002: Anthropic wraps auth errors properly."""
        mock_module, mock_client = create_mock_anthropic_module()
        mock_client.messages.create.side_effect = Exception("Authentication failed: Invalid API key")
        with patch.dict(sys.modules, {"anthropic": mock_module}):
            provider = AnthropicProvider()
            with pytest.raises(ProviderError) as exc_info:
                provider.complete([{"role": "user", "content": "Hi"}])
        assert "Anthropic API error" in str(exc_info.value)
        assert "Authentication" in str(exc_info.value)

    def test_openai_authentication_error(self, mock_openai_env):
        """OAI-002: OpenAI wraps auth errors properly."""
        mock_module, mock_client = create_mock_openai_module()
        mock_client.chat.completions.create.side_effect = Exception("401: Unauthorized")
        with patch.dict(sys.modules, {"openai": mock_module}):
            provider = OpenAIProvider()
            with pytest.raises(ProviderError) as exc_info:
                provider.complete([{"role": "user", "content": "Hi"}])
        assert "OpenAI API error" in str(exc_info.value)

    def test_anthropic_rate_limit_error(self, mock_anthropic_env):
        """ANT-005: Anthropic rate limit wrapped in ProviderError."""
        mock_module, mock_client = create_mock_anthropic_module()
        mock_client.messages.create.side_effect = Exception("Rate limit exceeded (429)")
        with patch.dict(sys.modules, {"anthropic": mock_module}):
            provider = AnthropicProvider()
            with pytest.raises(ProviderError) as exc_info:
                provider.complete([{"role": "user", "content": "Hi"}])
        assert "Anthropic API error" in str(exc_info.value)

    def test_openai_rate_limit_error(self, mock_openai_env):
        """OAI-003: OpenAI rate limit wrapped in ProviderError."""
        mock_module, mock_client = create_mock_openai_module()
        mock_client.chat.completions.create.side_effect = Exception("Rate limit exceeded")
        with patch.dict(sys.modules, {"openai": mock_module}):
            provider = OpenAIProvider()
            with pytest.raises(ProviderError) as exc_info:
                provider.complete([{"role": "user", "content": "Hi"}])
        assert "OpenAI API error" in str(exc_info.value)


class TestBaseProviderAbstract:
    """Tests for BaseProvider abstract class."""

    def test_base_provider_is_abstract(self):
        """BaseProvider cannot be instantiated directly."""
        from expectllm.providers import BaseProvider
        with pytest.raises(TypeError) as exc_info:
            BaseProvider()
        assert "abstract" in str(exc_info.value).lower() or "instantiate" in str(exc_info.value).lower()

    def test_base_provider_requires_complete_method(self):
        """BaseProvider requires complete() to be implemented."""
        from expectllm.providers import BaseProvider

        class IncompleteProvider(BaseProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()


class TestGetProviderAdvanced:
    """Advanced tests for get_provider function."""

    def test_get_provider_with_both_model_and_provider(self, mock_openai_env):
        """get_provider with both model and explicit provider."""
        mock_module, _ = create_mock_openai_module()
        with patch.dict(sys.modules, {"openai": mock_module}):
            # Explicit provider overrides model prefix detection
            provider = get_provider(model="claude-3-opus", provider="openai")
        assert isinstance(provider, OpenAIProvider)

    def test_get_provider_model_prefix_case_insensitive(self, mock_anthropic_env):
        """Model prefix detection is case-insensitive."""
        mock_module, _ = create_mock_anthropic_module()
        with patch.dict(sys.modules, {"anthropic": mock_module}):
            # Uppercase prefix should still work
            provider = get_provider(model="CLAUDE-3-opus")
        assert isinstance(provider, AnthropicProvider)

    def test_get_provider_gpt_prefix_case_insensitive(self, mock_openai_env):
        """GPT model prefix detection is case-insensitive."""
        mock_module, _ = create_mock_openai_module()
        with patch.dict(sys.modules, {"openai": mock_module}):
            provider = get_provider(model="GPT-4-turbo")
        assert isinstance(provider, OpenAIProvider)


class TestAnthropicProviderAdvanced:
    """Advanced tests for AnthropicProvider."""

    def test_anthropic_provider_handles_empty_response(self, mock_anthropic_env):
        """AnthropicProvider handles empty response content."""
        mock_module, mock_client = create_mock_anthropic_module()
        # Simulate empty content
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="")]
        mock_client.messages.create.return_value = mock_response
        with patch.dict(sys.modules, {"anthropic": mock_module}):
            provider = AnthropicProvider()
            result = provider.complete([{"role": "user", "content": "Hello"}])
        assert result == ""

    def test_anthropic_provider_without_system_prompt(self, mock_anthropic_env):
        """AnthropicProvider works without system prompt."""
        mock_module, mock_client = create_mock_anthropic_module()
        with patch.dict(sys.modules, {"anthropic": mock_module}):
            provider = AnthropicProvider()
            provider.complete([{"role": "user", "content": "Hello"}])
        call_kwargs = mock_client.messages.create.call_args.kwargs
        # system should not be in kwargs when not provided
        assert "system" not in call_kwargs or call_kwargs.get("system") is None

    def test_anthropic_provider_error_preserves_cause(self, mock_anthropic_env):
        """AnthropicProvider preserves original exception as cause."""
        mock_module, mock_client = create_mock_anthropic_module()
        original_error = ValueError("Original error")
        mock_client.messages.create.side_effect = original_error
        with patch.dict(sys.modules, {"anthropic": mock_module}):
            provider = AnthropicProvider()
            with pytest.raises(ProviderError) as exc_info:
                provider.complete([{"role": "user", "content": "Hi"}])
        assert exc_info.value.__cause__ is original_error


class TestOpenAIProviderAdvanced:
    """Advanced tests for OpenAIProvider."""

    def test_openai_provider_error_preserves_cause(self, mock_openai_env):
        """OpenAIProvider preserves original exception as cause."""
        mock_module, mock_client = create_mock_openai_module()
        original_error = ValueError("Original error")
        mock_client.chat.completions.create.side_effect = original_error
        with patch.dict(sys.modules, {"openai": mock_module}):
            provider = OpenAIProvider()
            with pytest.raises(ProviderError) as exc_info:
                provider.complete([{"role": "user", "content": "Hi"}])
        assert exc_info.value.__cause__ is original_error

    def test_openai_provider_messages_order_correct(self, mock_openai_env):
        """OpenAI provider sends messages in correct order."""
        mock_module, mock_client = create_mock_openai_module()
        with patch.dict(sys.modules, {"openai": mock_module}):
            provider = OpenAIProvider()
            provider.complete(
                [
                    {"role": "user", "content": "First"},
                    {"role": "assistant", "content": "Response"},
                    {"role": "user", "content": "Second"}
                ],
                system_prompt="System"
            )
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "First"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "Second"


class TestModelPrefixesAdvanced:
    """Advanced tests for model prefix handling."""

    def test_all_model_prefixes_defined(self):
        """All expected model prefixes are defined."""
        assert "claude" in MODEL_PREFIXES
        assert "gpt" in MODEL_PREFIXES
        assert "o1" in MODEL_PREFIXES

    def test_default_models_defined(self):
        """Default models are defined for all providers."""
        assert "anthropic" in DEFAULT_MODELS
        assert "openai" in DEFAULT_MODELS
        assert DEFAULT_MODELS["anthropic"].startswith("claude")
        assert "gpt" in DEFAULT_MODELS["openai"] or "o1" in DEFAULT_MODELS["openai"]
