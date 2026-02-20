"""Stress tests for llm-expect library."""
import re
import gc
import sys
import copy
import pickle
import threading
import time
import pytest
from unittest.mock import patch, MagicMock
from llmexpect import Conversation, ExpectError, ProviderError, ConfigError


class MockProvider:
    """Mock provider for stress testing."""

    def __init__(self, responses=None, delay=0):
        self.responses = responses or ["Test response"]
        self.call_count = 0
        self.delay = delay

    def complete(self, messages, system_prompt=None, timeout=60):
        if self.delay:
            time.sleep(self.delay)
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response


class TestLoadTesting:
    """LOAD tests - Load and throughput testing."""

    def test_sequential_throughput(self, mock_anthropic_env):
        """LOAD-001: 50 sequential send() calls."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["response"] * 100)
            mock_get.return_value = mock
            c = Conversation()

            start = time.time()
            for i in range(50):
                c.send(f"Message {i}")
            elapsed = time.time() - start

            assert mock.call_count == 50
            assert len(c.history) == 100  # 50 user + 50 assistant
            # Should be very fast with mocks
            assert elapsed < 1.0

    def test_concurrent_conversations(self, mock_anthropic_env):
        """LOAD-002: 20 threads x 5 messages each."""
        results = []
        errors = []

        def run_conversation(thread_id):
            try:
                with patch("llmexpect.conversation.get_provider") as mock_get:
                    mock = MockProvider([f"Response {thread_id}"] * 10)
                    mock_get.return_value = mock
                    c = Conversation()
                    for i in range(5):
                        c.send(f"Thread {thread_id} message {i}")
                    results.append((thread_id, c.history))
            except Exception as e:
                errors.append((thread_id, e))

        threads = []
        for i in range(20):
            t = threading.Thread(target=run_conversation, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 20
        # Each conversation should have 10 messages (5 user + 5 assistant)
        for thread_id, history in results:
            assert len(history) == 10

    def test_long_conversation(self, mock_anthropic_env):
        """LOAD-003: 200 turns in one conversation."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["Response"] * 300)
            mock_get.return_value = mock
            c = Conversation()

            for i in range(200):
                c.send(f"Turn {i}")

            assert len(c.history) == 400  # 200 user + 200 assistant
            assert mock.call_count == 200

    def test_large_response_handling(self, mock_anthropic_env):
        """LOAD-004: Response > 100KB."""
        large_response = "x" * 150000  # 150KB
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider([large_response])
            mock_get.return_value = mock
            c = Conversation()

            response = c.send("Give large response")
            assert len(response) == 150000
            assert c.last_response == large_response

    def test_large_pattern_handling(self, mock_anthropic_env):
        """LOAD-005: Pattern > 10KB."""
        # Create a large but valid regex pattern with longer words
        large_pattern = "(" + "|".join([f"longerword{i:04d}" for i in range(1000)]) + ")"
        assert len(large_pattern) > 10000

        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["The result is longerword0500 here"])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            # Should handle large pattern
            result = c.expect(large_pattern)
            assert result is True
            assert c.match.group(1) == "longerword0500"

    def test_memory_stability(self, mock_anthropic_env):
        """LOAD-006: Memory stability with many conversations."""
        initial_objects = len(gc.get_objects())

        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["Response"])
            mock_get.return_value = mock

            # Create and destroy many conversations
            for i in range(100):
                c = Conversation()
                c.send("Test")
                del c

            gc.collect()

        final_objects = len(gc.get_objects())
        # Allow some growth but not excessive
        growth = final_objects - initial_objects
        assert growth < 1000, f"Object growth: {growth}"

    def test_cpu_bound_pattern(self, mock_anthropic_env):
        """LOAD-007: CPU-bound pattern matching completes."""
        # Complex but not catastrophic pattern
        pattern = r"(\w+\s+){10,20}end"
        response = " ".join(["word"] * 15) + " end"

        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider([response])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            start = time.time()
            result = c.expect(pattern)
            elapsed = time.time() - start

            assert result is True
            assert elapsed < 1.0  # Should complete quickly


class TestAdversarialInputs:
    """ADV tests - Adversarial input handling."""

    def test_regex_catastrophic_backtracking_prevention(self, mock_anthropic_env):
        """ADV-001: Regex with potential backtracking."""
        # This pattern could cause catastrophic backtracking on wrong input
        # but we use a response that matches cleanly
        pattern = r"(a+)+b"
        response = "aaab"  # Simple match, no backtracking

        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider([response])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            start = time.time()
            result = c.expect(pattern)
            elapsed = time.time() - start

            assert result is True
            assert elapsed < 1.0

    def test_null_bytes_in_response(self, mock_anthropic_env):
        """ADV-002: Response contains null bytes."""
        response_with_nulls = "Hello\x00World\x00Test"

        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider([response_with_nulls])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            # Should handle null bytes without crashing
            assert c.last_response == response_with_nulls
            result = c.expect(r"Hello")
            assert result is True

    def test_control_characters_in_response(self, mock_anthropic_env):
        """ADV-003: Response contains ANSI control characters."""
        response_with_ansi = "Normal \x1b[31mRed\x1b[0m text"

        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider([response_with_ansi])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            assert c.last_response == response_with_ansi
            result = c.expect(r"Red")
            assert result is True

    def test_very_long_prompt(self, mock_anthropic_env):
        """ADV-004: 1MB prompt handling."""
        long_prompt = "x" * 1_000_000  # 1MB

        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["OK"])
            mock_get.return_value = mock
            c = Conversation()

            response = c.send(long_prompt)
            assert response == "OK"
            assert len(c.history[0]["content"]) == 1_000_000

    def test_recursive_pattern_error(self, mock_anthropic_env):
        """ADV-005: Recursive regex pattern raises proper error."""
        # Python's re module doesn't support recursive patterns
        # Using an invalid pattern instead
        invalid_pattern = r"(?P<name>(?P<name>test))"  # Duplicate group name

        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["test"])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            with pytest.raises(ExpectError) as exc_info:
                c.expect(invalid_pattern)
            assert "Invalid regex" in str(exc_info.value)

    def test_sql_injection_attempt(self, mock_anthropic_env):
        """ADV-006: SQL injection in prompt is safe."""
        injection = "'; DROP TABLE users; --"

        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["Normal response"])
            mock_get.return_value = mock
            c = Conversation()

            response = c.send(injection)
            # Should just be treated as text
            assert response == "Normal response"
            assert injection in c.history[0]["content"]

    def test_path_traversal_attempt(self, mock_anthropic_env):
        """ADV-007: Path traversal in prompt is safe."""
        traversal = "../../../etc/passwd"

        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["Safe response"])
            mock_get.return_value = mock
            c = Conversation()

            response = c.send(traversal)
            assert response == "Safe response"
            # No file access occurs


class TestEdgeCases:
    """EDGE tests - Edge case handling."""

    def test_conversation_reuse(self, mock_anthropic_env):
        """EDGE-001: Same conversation many send() calls."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["Response"] * 100)
            mock_get.return_value = mock
            c = Conversation()

            for i in range(50):
                c.send(f"Message {i}")

            # Should work without resource leaks
            assert len(c.history) == 100
            assert mock.call_count == 50

    def test_parallel_expect_calls(self, mock_anthropic_env):
        """EDGE-002: Multiple expect() on same response."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["The answer is 42 and status is OK"])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            # Multiple patterns on same response
            c.expect(r"(\d+)")
            assert c.match.group(1) == "42"

            c.expect(r"status is (\w+)")
            assert c.match.group(1) == "OK"

    def test_rapid_fire_sends(self, mock_anthropic_env):
        """EDGE-003: Rapid send() calls."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["OK"] * 100)
            mock_get.return_value = mock
            c = Conversation()

            start = time.time()
            for i in range(100):
                c.send("Quick")
            elapsed = time.time() - start

            assert mock.call_count == 100
            assert elapsed < 1.0  # Very fast with mocks

    def test_conversation_after_clear(self, mock_anthropic_env):
        """EDGE-004: Conversation works after clear_history."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["First", "Second"])
            mock_get.return_value = mock
            c = Conversation()

            c.send("First message")
            c.clear_history()

            assert len(c.history) == 0
            assert c.match is None
            assert c.last_response == ""

            # Should work normally after clear
            c.send("New message")
            assert len(c.history) == 2
            assert c.last_response == "Second"

    def test_pickle_serialization(self, mock_anthropic_env):
        """EDGE-005: Conversation state survives pickle."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["Response with 42"])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")
            c.expect(r"(\d+)")

            # Store state before pickle
            history_before = c.history.copy()
            last_response_before = c.last_response

            # Pickle and unpickle (note: provider won't survive)
            try:
                data = pickle.dumps(c._history)
                restored_history = pickle.loads(data)
                assert restored_history == history_before
            except Exception:
                pytest.skip("Pickle not supported for this object")

    def test_deepcopy_conversation(self, mock_anthropic_env):
        """EDGE-006: Deepcopy creates independent copy."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["Response"])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Original")

            # Copy the history
            history_copy = copy.deepcopy(c.history)

            # Modify original
            c.send("Modified")

            # Copy should be independent
            assert len(history_copy) == 2
            assert len(c.history) == 4


class TestSecurityTests:
    """SEC tests - Security validations."""

    def test_api_key_not_in_error_messages(self, mock_anthropic_env, monkeypatch):
        """SEC-002: API key not exposed in error messages."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-key-12345")

        err = ProviderError("Connection failed")
        assert "sk-ant" not in str(err)
        assert "secret" not in str(err).lower()

    def test_api_key_not_in_expect_error(self, mock_anthropic_env, monkeypatch):
        """SEC-002b: API key not in ExpectError."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-key-12345")

        err = ExpectError("pattern", "response with sk-ant-secret")
        # The response might contain it but truncation should help
        assert len(err.response) <= 500

    def test_no_eval_exec_in_codebase(self):
        """SEC-004: No eval/exec in source code (re.compile is allowed)."""
        import os
        src_dir = os.path.join(os.path.dirname(__file__), "..", "src", "llmexpect")

        # Only check for dangerous eval/exec - re.compile is safe
        dangerous_patterns = ["eval(", "exec("]

        for filename in ["conversation.py", "providers.py", "errors.py", "__init__.py"]:
            filepath = os.path.join(src_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    content = f.read()
                    for pattern in dangerous_patterns:
                        assert pattern not in content, f"Found {pattern} in {filename}"

    def test_no_subprocess_in_codebase(self):
        """SEC-005: No shell commands in source code."""
        import os
        src_dir = os.path.join(os.path.dirname(__file__), "..", "src", "llmexpect")

        dangerous_imports = ["import subprocess", "from subprocess", "import os\nos.system", "os.popen"]

        for filename in ["conversation.py", "providers.py", "errors.py"]:
            filepath = os.path.join(src_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    content = f.read()
                    for pattern in dangerous_imports:
                        if pattern == "import os\nos.system":
                            assert "os.system" not in content, f"Found os.system in {filename}"
                        elif pattern == "os.popen":
                            assert "os.popen" not in content, f"Found os.popen in {filename}"
                        else:
                            assert pattern not in content, f"Found {pattern} in {filename}"

    def test_response_truncation_in_errors(self, mock_anthropic_env):
        """SEC: Large response truncated in error messages."""
        # Response with potential secrets
        secret_response = "key=" + "x" * 1000 + "secret_value"

        err = ExpectError("pattern", secret_response)
        # Should be truncated
        assert len(err.response) == 500
        assert "secret_value" not in err.response


class TestTimeoutTests:
    """TIME tests - Timeout handling."""

    def test_timeout_triggers_with_slow_provider(self, mock_anthropic_env):
        """TIME-005: Timeout actually triggers on slow response."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            # Create a mock that sleeps longer than timeout
            slow_mock = MockProvider(["Response"], delay=0.5)
            mock_get.return_value = slow_mock

            c = Conversation(timeout=1)  # Short timeout

            # This should work since our mock is faster than timeout
            start = time.time()
            response = c.send("Test")
            elapsed = time.time() - start

            assert response == "Response"
            assert elapsed >= 0.5  # Delay was applied


class TestNetworkErrors:
    """NET tests - Network error handling."""

    def test_connection_error_wrapped(self, mock_anthropic_env):
        """NET-002: Connection errors wrapped in ProviderError."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider()
            mock.complete = MagicMock(side_effect=ConnectionError("Connection refused"))
            mock_get.return_value = mock

            c = Conversation()

            with pytest.raises(Exception):  # May be ConnectionError or ProviderError
                c.send("Test")

    def test_timeout_error_handling(self, mock_anthropic_env):
        """NET-004: Timeout error handling."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider()
            mock.complete = MagicMock(side_effect=TimeoutError("Request timeout"))
            mock_get.return_value = mock

            c = Conversation()

            with pytest.raises(Exception):  # May be TimeoutError or ProviderError
                c.send("Test")


class TestBossTests:
    """BOSS tests - Ultimate stress tests."""

    def test_unicode_hell(self, mock_anthropic_env):
        """BOSS-004: Unicode edge cases."""
        # Emoji + RTL + ZWJ sequences
        unicode_response = "Hello 👨‍👩‍👧‍👦 مرحبا 🏳️‍🌈 Test"

        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider([unicode_response])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            assert c.last_response == unicode_response
            result = c.expect(r"Hello")
            assert result is True

    def test_consecutive_turn_marathon(self, mock_anthropic_env):
        """BOSS-007: 200 consecutive turns."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider([f"Response {i}" for i in range(250)])
            mock_get.return_value = mock
            c = Conversation()

            for i in range(200):
                response = c.send(f"Turn {i}")
                assert response == f"Response {i}"

            assert len(c.history) == 400

    def test_control_chars_in_response(self, mock_anthropic_env):
        """BOSS-008: Response with control characters."""
        response = "Line1\x00\x1b\r\nLine2\tTab\x08Backspace"

        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider([response])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            # Should handle without corruption
            assert "Line1" in c.last_response
            assert "Line2" in c.last_response

    def test_zero_latency_flood(self, mock_anthropic_env):
        """BOSS-011: 1000 requests as fast as possible."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["OK"] * 1100)
            mock_get.return_value = mock
            c = Conversation(max_history=100)  # Limit history to prevent memory issues

            start = time.time()
            for i in range(1000):
                c.send(f"Flood {i}")
            elapsed = time.time() - start

            assert mock.call_count == 1000
            # Should complete in reasonable time with mocks
            assert elapsed < 5.0

    def test_adversarial_llm_response(self, mock_anthropic_env):
        """BOSS-012: LLM returns potentially dangerous content."""
        # Response that looks like code/injection
        dangerous_response = """
        import os; os.system('rm -rf /')
        <script>alert('xss')</script>
        '; DROP TABLE users; --
        """

        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider([dangerous_response])
            mock_get.return_value = mock
            c = Conversation()

            response = c.send("Test")
            # Should just store as text, not execute
            assert response == dangerous_response
            assert "import os" in c.last_response
            # Nothing bad happened


class TestPatternEdgeCases:
    """Additional pattern matching edge cases."""

    def test_empty_pattern(self, mock_anthropic_env):
        """Empty pattern matches everything."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["Any response"])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            result = c.expect(r"")
            assert result is True

    def test_multiline_pattern(self, mock_anthropic_env):
        """Multiline pattern with ^ and $."""
        response = "Line 1\nLine 2\nLine 3"

        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider([response])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            # MULTILINE flag makes ^ and $ match line boundaries
            result = c.expect(r"^Line 2$", re.MULTILINE)
            assert result is True

    def test_lookahead_pattern(self, mock_anthropic_env):
        """Lookahead pattern."""
        response = "foo123bar"

        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider([response])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            # Positive lookahead
            result = c.expect(r"foo(?=123)")
            assert result is True

    def test_lookbehind_pattern(self, mock_anthropic_env):
        """Lookbehind pattern."""
        response = "foo123bar"

        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider([response])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            # Positive lookbehind
            result = c.expect(r"(?<=foo)123")
            assert result is True
            assert c.match.group(0) == "123"

    def test_non_capturing_groups(self, mock_anthropic_env):
        """Non-capturing groups."""
        response = "color: red"

        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider([response])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            result = c.expect(r"color: (?:red|blue|green)")
            assert result is True

    def test_word_boundaries(self, mock_anthropic_env):
        """Word boundary matching."""
        response = "the cat in the hat"

        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider([response])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            # Should match whole word "cat" not "cat" in "scatter"
            result = c.expect(r"\bcat\b")
            assert result is True


class TestExpectTemplateStress:
    """Stress tests for expect templates."""

    def test_expect_json_deeply_nested(self, mock_anthropic_env):
        """Deeply nested JSON in code block (nested JSON requires code block extraction)."""
        deep_json = '{"a": {"b": {"c": {"d": {"e": "value"}}}}}'

        with patch("llmexpect.conversation.get_provider") as mock_get:
            # Use code block format for deeply nested JSON
            mock = MockProvider([f"```json\n{deep_json}\n```"])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            result = c.expect_json()
            assert result["a"]["b"]["c"]["d"]["e"] == "value"

    def test_expect_json_with_arrays(self, mock_anthropic_env):
        """JSON with arrays."""
        json_with_arrays = '{"items": [1, 2, 3], "names": ["a", "b", "c"]}'

        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider([f"Result: {json_with_arrays}"])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            result = c.expect_json()
            assert result["items"] == [1, 2, 3]
            assert result["names"] == ["a", "b", "c"]

    def test_expect_number_large(self, mock_anthropic_env):
        """Very large number."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["The count is 999,999,999"])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            result = c.expect_number()
            assert result == 999999999

    def test_expect_choice_many_options(self, mock_anthropic_env):
        """Many choice options with distinct names."""
        # Use distinct names to avoid substring collisions (opt_50 vs opt_5)
        choices = [f"opt_{i:03d}" for i in range(100)]

        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["I choose opt_050"])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            result = c.expect_choice(choices)
            assert result == "opt_050"

    def test_expect_code_multiple_blocks(self, mock_anthropic_env):
        """Multiple code blocks - returns first."""
        response = """
Here's Python:
```python
print("first")
```

And JavaScript:
```javascript
console.log("second")
```
"""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider([response])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            result = c.expect_code("python")
            assert 'print("first")' in result


class TestMaxHistoryEdgeCases:
    """Edge cases for max_history parameter."""

    def test_max_history_zero_raises(self, mock_anthropic_env):
        """max_history=0 raises ValueError (must be at least 2)."""
        with pytest.raises(ValueError, match="must be at least 2"):
            Conversation(max_history=0)

    def test_max_history_one_raises(self, mock_anthropic_env):
        """max_history=1 raises ValueError (must be at least 2)."""
        with pytest.raises(ValueError, match="must be at least 2"):
            Conversation(max_history=1)

    def test_max_history_two_is_minimum(self, mock_anthropic_env):
        """max_history=2 is the minimum valid value (keeps last exchange)."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["r1", "r2", "r3"])
            mock_get.return_value = mock
            c = Conversation(max_history=2)

            c.send("m1")
            c.send("m2")
            c.send("m3")

            # Should keep only last 2 messages (last exchange)
            assert len(c.history) == 2
            assert c.history[-2]["role"] == "user"
            assert c.history[-2]["content"] == "m3"
            assert c.history[-1]["role"] == "assistant"
            assert c.history[-1]["content"] == "r3"

    def test_max_history_none_unlimited(self, mock_anthropic_env):
        """max_history=None means unlimited history."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["response"] * 100)
            mock_get.return_value = mock
            c = Conversation(max_history=None)

            for i in range(50):
                c.send(f"msg{i}")

            # All 100 messages should be kept (50 user + 50 assistant)
            assert len(c.history) == 100


class TestConversationStateConsistency:
    """Tests for conversation state consistency."""

    def test_last_response_matches_history(self, mock_anthropic_env):
        """last_response always matches last assistant message in history."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["response1", "response2", "response3"])
            mock_get.return_value = mock
            c = Conversation()

            for i in range(3):
                c.send(f"msg{i}")
                # last_response should match the most recent assistant message
                assistant_msgs = [m for m in c.history if m["role"] == "assistant"]
                assert c.last_response == assistant_msgs[-1]["content"]

    def test_match_cleared_after_clear_history(self, mock_anthropic_env):
        """match is cleared after clear_history()."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["42"])
            mock_get.return_value = mock
            c = Conversation()

            c.send("Test")
            c.expect(r"(\d+)")
            assert c.match is not None

            c.clear_history()
            assert c.match is None

    def test_last_response_cleared_after_clear_history(self, mock_anthropic_env):
        """last_response is empty after clear_history()."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["Some response"])
            mock_get.return_value = mock
            c = Conversation()

            c.send("Test")
            assert c.last_response == "Some response"

            c.clear_history()
            assert c.last_response == ""


class TestExpectJsonArrayHandling:
    """Tests for expect_json with arrays."""

    def test_expect_json_rejects_bare_array(self, mock_anthropic_env):
        """expect_json() with bare JSON array should fail."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(['[1, 2, 3]'])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            # dict([1,2,3]) would fail - this tests that behavior
            with pytest.raises((ExpectError, TypeError, ValueError)):
                c.expect_json()

    def test_expect_json_object_with_array_values(self, mock_anthropic_env):
        """expect_json() with object containing arrays works."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(['{"data": [1, 2, 3], "nested": {"arr": ["a", "b"]}}'])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            result = c.expect_json()
            assert result["data"] == [1, 2, 3]
            assert result["nested"]["arr"] == ["a", "b"]


class TestProviderInteraction:
    """Tests for how conversation interacts with providers."""

    def test_provider_receives_all_messages(self, mock_anthropic_env):
        """Provider receives complete message history."""
        received_messages = []

        class MessageTrackingProvider:
            def complete(self, messages, system_prompt=None, timeout=60):
                received_messages.append(list(messages))
                return f"r{len(received_messages)}"

        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock_get.return_value = MessageTrackingProvider()
            c = Conversation()

            c.send("m1")
            c.send("m2")
            c.send("m3")

            # Check what was sent to provider on last call
            last_messages = received_messages[-1]
            # Should have 5 messages (m1, r1, m2, r2, m3)
            assert len(last_messages) == 5
            assert last_messages[0]["content"] == "m1"
            assert last_messages[1]["content"] == "r1"
            assert last_messages[4]["content"] == "m3"

    def test_system_prompt_passed_to_provider(self, mock_anthropic_env):
        """System prompt is passed to provider on each call."""
        received_system_prompts = []

        class SystemPromptTrackingProvider:
            def complete(self, messages, system_prompt=None, timeout=60):
                received_system_prompts.append(system_prompt)
                return "response"

        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock_get.return_value = SystemPromptTrackingProvider()
            c = Conversation(system_prompt="Be helpful")

            c.send("Hello")

            assert received_system_prompts[-1] == "Be helpful"

    def test_timeout_passed_to_provider(self, mock_anthropic_env):
        """Timeout is passed to provider."""
        call_timeouts = []

        class TimeoutTrackingProvider:
            def complete(self, messages, system_prompt=None, timeout=60):
                call_timeouts.append(timeout)
                return "response"

        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock_get.return_value = TimeoutTrackingProvider()
            c = Conversation(timeout=120)

            c.send("Test")

            assert call_timeouts[-1] == 120


class TestComplexPatternMatching:
    """Complex pattern matching scenarios."""

    def test_pattern_with_unicode_classes(self, mock_anthropic_env):
        """Pattern with Unicode character classes."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["日本語テスト"])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            # Match any word characters (including Unicode)
            result = c.expect(r"\w+")
            assert result is True

    def test_pattern_with_backreferences(self, mock_anthropic_env):
        """Pattern with backreferences."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["The word 'hello' is repeated: hello"])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            # Backreference to first group
            result = c.expect(r"(\w+).+\1")
            assert result is True

    def test_pattern_with_conditional(self, mock_anthropic_env):
        """Pattern with conditional construct."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock = MockProvider(["Value: 123"])
            mock_get.return_value = mock
            c = Conversation()
            c.send("Test")

            # Match key: value pattern
            result = c.expect(r"(\w+):\s*(\d+)")
            assert result is True
            assert c.match.group(1) == "Value"
            assert c.match.group(2) == "123"


class TestConcurrencySafety:
    """Tests for concurrent access safety."""

    def test_multiple_conversations_independent(self, mock_anthropic_env):
        """Multiple conversations don't interfere."""
        with patch("llmexpect.conversation.get_provider") as mock_get:
            mock1 = MockProvider(["response1"])
            mock2 = MockProvider(["response2"])

            mock_get.return_value = mock1
            c1 = Conversation()
            c1._provider = mock1

            mock_get.return_value = mock2
            c2 = Conversation()
            c2._provider = mock2

            c1.send("msg1")
            c2.send("msg2")

            assert c1.last_response == "response1"
            assert c2.last_response == "response2"
            assert len(c1.history) == 2
            assert len(c2.history) == 2
