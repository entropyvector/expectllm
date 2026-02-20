"""Custom exceptions for llm-expect."""


class LLMExpectError(Exception):
    """Base exception for llm-expect."""


class ExpectError(LLMExpectError):
    """Pattern not found in response."""

    def __init__(self, pattern: str, response: str, suggestion: str = "") -> None:
        self.pattern = pattern
        self.response = response[:500] if len(response) > 500 else response
        self.suggestion = suggestion or "Try a more flexible pattern or use retry."
        message = (
            f"Pattern '{pattern}' not found in response.\n\n"
            f"Response was:\n{self.response}"
            f"{'...' if len(response) > 500 else ''}\n\n"
            f"Suggestion: {self.suggestion}"
        )
        super().__init__(message)


class ProviderError(LLMExpectError):
    """Provider/API related error."""


class ConfigError(LLMExpectError):
    """Configuration/setup error."""
