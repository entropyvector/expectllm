"""Core Conversation class for llm-expect."""
from __future__ import annotations

import json
import re
from re import Match
from typing import Any

from .errors import ExpectError
from .providers import BaseProvider, get_provider


class Conversation:
    """A conversation with an LLM using expect-style pattern matching."""

    def __init__(
        self,
        model: str | None = None,
        system_prompt: str | None = None,
        timeout: int = 60,
        provider: str | None = None,
        max_history: int | None = None,
    ) -> None:
        """Initialize conversation. Args auto-detected from env if not specified."""
        if timeout <= 0:
            raise ValueError("timeout must be a positive integer")
        if max_history is not None and max_history < 2:
            raise ValueError("max_history must be at least 2 (user + assistant)")
        self._provider: BaseProvider = get_provider(model, provider)
        self._model = model
        self._system_prompt = system_prompt
        self._timeout = timeout
        self._history: list[dict[str, str]] = []
        self._match: Match[str] | None = None
        self._last_response: str = ""
        self._max_history = max_history

    def send(self, message: str, expect: str | None = None, flags: int = 0) -> str:
        """Send message to LLM. If expect provided, validates response matches pattern."""
        full_message = message
        if expect:
            instruction = _pattern_to_instruction(expect)
            if instruction:
                full_message = f"{message}\n\n{instruction}"
        self._history.append({"role": "user", "content": full_message})
        response = self._provider.complete(
            messages=self._history, system_prompt=self._system_prompt, timeout=self._timeout
        )
        self._history.append({"role": "assistant", "content": response})
        self._last_response = response
        if self._max_history is not None and len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        if expect:
            self.expect(expect, flags)
        return response

    def expect(self, pattern: str, flags: int = 0) -> bool:
        """Check if last response matches regex pattern. Raises ExpectError if not."""
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            raise ExpectError(pattern, self._last_response, f"Invalid regex: {e}") from e
        match = compiled.search(self._last_response)
        if match:
            self._match = match
            return True
        raise ExpectError(pattern, self._last_response)

    def send_expect(
        self, message: str, pattern: str, flags: int = 0, timeout: int | None = None
    ) -> Match[str]:
        """Send message and expect pattern. Returns match object."""
        original_timeout = self._timeout
        if timeout is not None:
            self._timeout = timeout
        try:
            self.send(message)
            self.expect(pattern, flags)
        finally:
            self._timeout = original_timeout
        assert self._match is not None
        return self._match

    def expect_json(self) -> dict[str, Any]:
        """Extract JSON object from last response."""
        response = self._last_response
        # Limit response length to prevent ReDoS attacks on regex patterns
        if len(response) > 100000:
            raise ExpectError("JSON object", response[:500], "Response too long.")
        code_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", response)
        if code_match:
            try:
                return dict(json.loads(code_match.group(1)))
            except json.JSONDecodeError:
                pass
        json_match = re.search(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", response)
        if json_match:
            try:
                return dict(json.loads(json_match.group(1)))
            except json.JSONDecodeError:
                pass
        raise ExpectError("JSON object", response, "Return JSON in a ```json code block.")

    def expect_number(self) -> int:
        """Extract first number from last response.

        Matches: integers (42), negative (-5), with commas (1,000)
        """
        # Pattern requires at least one digit, with optional commas between digits
        match = re.search(r"-?\d{1,3}(?:,\d{3})*|-?\d+", self._last_response)
        if match:
            self._match = match
            return int(match.group(0).replace(",", ""))
        raise ExpectError("number", self._last_response, "Include a number in response.")

    def expect_choice(self, choices: list[str], case_sensitive: bool = False) -> str:
        """Match one of the provided choices in last response.

        Raises ValueError if choices is empty or all empty strings.
        """
        # Filter out empty strings - they would match anything
        valid_choices = [c for c in choices if c]
        if not valid_choices:
            raise ValueError("choices must contain at least one non-empty string")

        response = self._last_response
        search_response = response if case_sensitive else response.lower()
        for choice in valid_choices:
            search_choice = choice if case_sensitive else choice.lower()
            if search_choice in search_response:
                flags = 0 if case_sensitive else re.IGNORECASE
                match = re.search(re.escape(choice), response, flags)
                if match:
                    self._match = match
                return choice
        raise ExpectError(f"one of: {', '.join(valid_choices)}", response)

    def expect_yesno(self) -> bool:
        """Match yes/no response. Returns True for yes/true/y, False for no/false/n."""
        match = re.search(r"\b(yes|no|true|false|y|n)\b", self._last_response, re.IGNORECASE)
        if match:
            self._match = match
            return match.group(1).lower() in ("yes", "true", "y")
        raise ExpectError("yes/no", self._last_response, "Reply with YES or NO.")

    def expect_code(self, language: str | None = None) -> str:
        """Extract code block from last response."""
        if language:
            pattern = rf"```{re.escape(language)}\n([\s\S]*?)\n```"
        else:
            pattern = r"```(?:\w+)?\n([\s\S]*?)\n```"
        match = re.search(pattern, self._last_response, re.IGNORECASE)
        if match:
            self._match = match
            return match.group(1)
        lang_hint = f" {language}" if language else ""
        raise ExpectError(f"code block{lang_hint}", self._last_response)

    @property
    def match(self) -> Match[str] | None:
        """Last successful match object."""
        return self._match

    @property
    def history(self) -> list[dict[str, str]]:
        """Full conversation history (returns copy)."""
        return list(self._history)

    @property
    def last_response(self) -> str:
        """Most recent response."""
        return self._last_response

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history, self._match, self._last_response = [], None, ""


def _pattern_to_instruction(pattern: str) -> str:
    """Convert regex pattern to human-readable format instruction."""
    if re.search(r"\^?\(?(YES|NO)\|?(YES|NO)?\)?\$?", pattern, re.IGNORECASE):
        return "Reply with exactly 'YES' or 'NO'."
    if pattern in (r"^(\d+)$", r"(\d+)", r"^\d+$"):
        return "Include a number in your response."
    choice_match = re.search(r"\(([^)]+\|[^)]+)\)", pattern)
    if choice_match:
        parts = choice_match.group(1).split("|")
        choices = [c.strip() for c in parts if c.strip() and not c.startswith("?")]
        if choices:
            return f"Include one of: {', '.join(choices)}."
    if "json" in pattern.lower() or r"\{" in pattern:
        return "Return valid JSON."
    if "```" in pattern:
        lang_match = re.search(r"```(\w+)", pattern)
        if lang_match:
            return f"Put your code in a ```{lang_match.group(1)} code block."
        return "Put response in a code block."
    kv_match = re.search(r"([A-Z_]+):\s*\(", pattern)
    if kv_match:
        return f"Include '{kv_match.group(1)}: ' followed by your answer."
    return ""
