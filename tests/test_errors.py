"""Tests for llm-expect error classes."""
import pytest
from llmexpect import ExpectError, ProviderError, ConfigError, LLMExpectError


class TestExpectError:
    """Tests for ExpectError exception."""

    def test_expect_error_includes_pattern_in_message(self):
        """ExpectError includes pattern in message."""
        err = ExpectError(r"\d+", "no numbers here")
        assert r"\d+" in str(err)

    def test_expect_error_includes_response_snippet(self):
        """ExpectError includes response snippet."""
        err = ExpectError("pattern", "this is the response")
        assert "this is the response" in str(err)

    def test_expect_error_truncates_long_response(self):
        """ExpectError truncates response to 500 chars."""
        long_response = "x" * 1000
        err = ExpectError("pattern", long_response)
        assert len(err.response) == 500
        assert "..." in str(err)

    def test_expect_error_includes_suggestion(self):
        """ExpectError includes suggestion."""
        err = ExpectError("pattern", "response", "Try this instead")
        assert "Try this instead" in str(err)

    def test_expect_error_default_suggestion(self):
        """ExpectError has default suggestion."""
        err = ExpectError("pattern", "response")
        assert "retry" in str(err).lower()

    def test_expect_error_inherits_from_base(self):
        """ExpectError inherits from LLMExpectError."""
        err = ExpectError("pattern", "response")
        assert isinstance(err, LLMExpectError)
        assert isinstance(err, Exception)


class TestProviderError:
    """Tests for ProviderError exception."""

    def test_provider_error_message(self):
        """ProviderError stores message correctly."""
        err = ProviderError("API rate limit exceeded")
        assert "API rate limit exceeded" in str(err)

    def test_provider_error_inherits_from_base(self):
        """ProviderError inherits from LLMExpectError."""
        err = ProviderError("error")
        assert isinstance(err, LLMExpectError)


class TestConfigError:
    """Tests for ConfigError exception."""

    def test_config_error_message(self):
        """ConfigError stores message correctly."""
        err = ConfigError("Missing API key")
        assert "Missing API key" in str(err)

    def test_config_error_inherits_from_base(self):
        """ConfigError inherits from LLMExpectError."""
        err = ConfigError("error")
        assert isinstance(err, LLMExpectError)


class TestErrorHierarchy:
    """Tests for error hierarchy."""

    def test_all_errors_catchable_by_base(self):
        """All errors can be caught by LLMExpectError."""
        errors = [
            ExpectError("p", "r"),
            ProviderError("error"),
            ConfigError("error"),
        ]
        for err in errors:
            try:
                raise err
            except LLMExpectError:
                pass  # Expected
            except Exception:
                pytest.fail(f"{type(err).__name__} not caught by LLMExpectError")


class TestExpectErrorAdvanced:
    """Advanced ExpectError tests (ERR-001 to ERR-006)."""

    def test_expect_error_pattern_attribute(self):
        """ERR-001: Pattern accessible as attribute."""
        err = ExpectError(r"^\d+$", "no numbers")
        assert err.pattern == r"^\d+$"

    def test_expect_error_response_attribute(self):
        """ERR-002: Response accessible as attribute."""
        err = ExpectError("pat", "my response text")
        assert err.response == "my response text"

    def test_expect_error_truncation_at_500(self):
        """ERR-003: Truncation exactly at 500 chars."""
        response = "a" * 600
        err = ExpectError("pat", response)
        assert len(err.response) == 500
        # Original response was longer
        assert len(response) == 600

    def test_expect_error_suggestion_attribute(self):
        """ERR-004: Custom suggestion stored correctly."""
        err = ExpectError("pat", "resp", "Use case-insensitive flag")
        assert err.suggestion == "Use case-insensitive flag"

    def test_expect_error_is_catchable(self):
        """ERR-006: ExpectError is catchable."""
        caught = False
        try:
            raise ExpectError("test", "response")
        except ExpectError:
            caught = True
        assert caught is True

    def test_expect_error_re_raiseable(self):
        """ExpectError can be caught and re-raised."""
        with pytest.raises(ExpectError):
            try:
                raise ExpectError("test", "response")
            except ExpectError as e:
                assert "test" in str(e)
                raise

    def test_expect_error_with_empty_response(self):
        """ExpectError with empty response."""
        err = ExpectError("pattern", "")
        assert err.response == ""
        assert "pattern" in str(err)

    def test_expect_error_with_unicode(self):
        """ExpectError handles unicode in pattern and response."""
        err = ExpectError("café", "I love café au lait")
        assert "café" in str(err)

    def test_expect_error_with_newlines(self):
        """ExpectError handles multiline response."""
        err = ExpectError("pattern", "line1\nline2\nline3")
        assert "line1" in str(err)
        assert "line2" in str(err)


class TestErrorMessages:
    """Tests for error message quality."""

    def test_provider_error_preserves_original(self):
        """ProviderError preserves original error message."""
        original = "Connection timeout after 30s"
        err = ProviderError(original)
        assert original in str(err)

    def test_config_error_actionable(self):
        """ConfigError messages are actionable."""
        err = ConfigError("ANTHROPIC_API_KEY not set. Get key at https://console.anthropic.com/")
        msg = str(err)
        assert "ANTHROPIC_API_KEY" in msg
        assert "https://" in msg  # Contains URL for help

    def test_expect_error_format(self):
        """ExpectError has expected format."""
        err = ExpectError("my_pattern", "my_response", "my_suggestion")
        msg = str(err)
        # Should have pattern, response, and suggestion sections
        assert "my_pattern" in msg
        assert "my_response" in msg
        assert "my_suggestion" in msg

    def test_error_chaining_preserved(self):
        """Error chaining preserved for debugging."""
        original = ValueError("original error")
        try:
            try:
                raise original
            except ValueError as e:
                raise ProviderError(f"Wrapped: {e}") from e
        except ProviderError as wrapped:
            assert wrapped.__cause__ is original


class TestEdgeCaseErrors:
    """Edge case tests for errors."""

    def test_expect_error_special_chars_in_pattern(self):
        """ExpectError handles special chars in pattern."""
        pattern = r"[\w]+\s*[<>]\s*\d+"
        err = ExpectError(pattern, "no match")
        assert pattern in str(err)

    def test_expect_error_very_short_response(self):
        """ExpectError handles very short response."""
        err = ExpectError("long_pattern_here", "x")
        assert "x" in str(err)

    def test_config_error_empty_message(self):
        """ConfigError with empty message."""
        err = ConfigError("")
        # Should not crash
        str(err)

    def test_provider_error_none_like_message(self):
        """ProviderError with unusual message."""
        err = ProviderError("None")
        assert "None" in str(err)


class TestExpectErrorBoundary:
    """Boundary tests for ExpectError truncation."""

    def test_truncation_exactly_at_500(self):
        """Response of exactly 500 chars is not truncated."""
        response = "x" * 500
        err = ExpectError("pattern", response)
        assert err.response == response
        assert len(err.response) == 500
        # No ellipsis in message since exactly 500
        assert "..." not in str(err) or str(err).count("...") == 0

    def test_truncation_at_501(self):
        """Response of 501 chars is truncated to 500."""
        response = "y" * 501
        err = ExpectError("pattern", response)
        assert len(err.response) == 500
        assert err.response == "y" * 500
        # Should have ellipsis
        assert "..." in str(err)

    def test_truncation_at_499(self):
        """Response of 499 chars is not truncated."""
        response = "z" * 499
        err = ExpectError("pattern", response)
        assert err.response == response
        assert len(err.response) == 499


class TestExpectErrorFromRegex:
    """Tests for ExpectError from regex compilation errors."""

    def test_expect_error_preserves_regex_cause(self):
        """When ExpectError wraps regex error, cause is preserved."""
        import re
        from llmexpect import Conversation
        from unittest.mock import patch, MagicMock

        # Create a mock provider
        mock_provider = MagicMock()
        mock_provider.complete.return_value = "test response"

        with patch("llmexpect.conversation.get_provider", return_value=mock_provider):
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
                c = Conversation()
                c.send("Hello")

                with pytest.raises(ExpectError) as exc_info:
                    c.expect(r"[invalid")  # Invalid regex

                # Check that __cause__ is a re.error
                assert exc_info.value.__cause__ is not None
                assert isinstance(exc_info.value.__cause__, re.error)


class TestErrorInheritance:
    """Tests for error inheritance and polymorphism."""

    def test_expect_error_is_instance_of_exception(self):
        """ExpectError is instance of Exception."""
        err = ExpectError("p", "r")
        assert isinstance(err, Exception)

    def test_provider_error_is_instance_of_exception(self):
        """ProviderError is instance of Exception."""
        err = ProviderError("error")
        assert isinstance(err, Exception)

    def test_config_error_is_instance_of_exception(self):
        """ConfigError is instance of Exception."""
        err = ConfigError("error")
        assert isinstance(err, Exception)

    def test_llmexpect_error_is_base_for_all(self):
        """All custom errors inherit from LLMExpectError."""
        assert issubclass(ExpectError, LLMExpectError)
        assert issubclass(ProviderError, LLMExpectError)
        assert issubclass(ConfigError, LLMExpectError)


class TestErrorRepr:
    """Tests for error representation."""

    def test_expect_error_str_contains_all_parts(self):
        """ExpectError str contains pattern, response, and suggestion."""
        err = ExpectError("my_pattern", "my_response", "my_suggestion")
        s = str(err)
        assert "my_pattern" in s
        assert "my_response" in s
        assert "my_suggestion" in s

    def test_provider_error_str_matches_message(self):
        """ProviderError str is the message."""
        err = ProviderError("API failed")
        assert "API failed" in str(err)

    def test_config_error_str_matches_message(self):
        """ConfigError str is the message."""
        err = ConfigError("Missing key")
        assert "Missing key" in str(err)
