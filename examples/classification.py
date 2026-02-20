#!/usr/bin/env python3
"""Classification with expect_choice."""

from expectllm import Conversation

messages = [
    "The login button doesn't work on mobile",
    "Can you add dark mode?",
    "How do I reset my password?",
]

c = Conversation()

for msg in messages:
    c.send(f"Classify this support ticket: '{msg}'")
    category = c.expect_choice(["bug", "feature", "question"])
    print(f"'{msg[:30]}...' -> {category}")
    c.clear_history()  # Reset for next classification
