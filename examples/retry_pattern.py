#!/usr/bin/env python3
"""Retry pattern for robust extraction."""

from expectllm import Conversation, ExpectError


def extract_with_retry(prompt: str, max_retries: int = 3) -> dict:
    """Extract JSON with automatic retry on failure."""
    c = Conversation()
    c.send(prompt)

    for attempt in range(max_retries):
        try:
            return c.expect_json()
        except ExpectError:
            if attempt < max_retries - 1:
                c.send("Please format your response as valid JSON.")
            else:
                raise ExpectError(f"Failed to extract JSON after {max_retries} attempts")

    return {}  # unreachable


if __name__ == "__main__":
    result = extract_with_retry(
        "List 3 programming languages with their year of creation as JSON array"
    )
    print(f"Result: {result}")
