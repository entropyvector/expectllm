"""llm-expect: Expect scripts for LLM conversations.

Example:
    >>> from llmexpect import Conversation
    >>> c = Conversation()
    >>> c.send("What is 2+2?")
    >>> c.expect(r"(\\d+)")
    True
    >>> c.match.group(1)
    '4'
"""

from .conversation import Conversation
from .errors import ConfigError, ExpectError, LLMExpectError, ProviderError
from .providers import get_provider

__version__ = "0.1.0"
__all__ = [
    "Conversation",
    "ExpectError",
    "ProviderError",
    "ConfigError",
    "LLMExpectError",
    "get_provider",
]
