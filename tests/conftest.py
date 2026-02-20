"""Pytest fixtures and mocks for expectllm tests."""
import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, List, Optional


class MockProvider:
    """Mock provider for testing without real API calls."""

    def __init__(self, responses: Optional[List[str]] = None) -> None:
        self.responses = responses or ["Test response"]
        self.call_count = 0
        self.last_messages: List[Dict[str, str]] = []
        self.last_system_prompt: Optional[str] = None

    def complete(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        timeout: int = 60,
    ) -> str:
        self.last_messages = messages
        self.last_system_prompt = system_prompt
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response


@pytest.fixture
def mock_provider():
    """Create a mock provider instance."""
    return MockProvider()


@pytest.fixture
def mock_anthropic_env(monkeypatch):
    """Set up environment for Anthropic provider."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def mock_openai_env(monkeypatch):
    """Set up environment for OpenAI provider."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def no_api_keys(monkeypatch):
    """Remove all API keys from environment."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def conversation_with_mock(mock_anthropic_env):
    """Create a Conversation with mocked provider."""
    from expectllm import Conversation

    mock = MockProvider()
    with patch("expectllm.conversation.get_provider", return_value=mock):
        conv = Conversation()
        conv._mock = mock  # Expose mock for test assertions
        yield conv


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for provider tests."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Mock response")]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    return mock_client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for provider tests."""
    mock_message = MagicMock()
    mock_message.content = "Mock response"

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    return mock_client
