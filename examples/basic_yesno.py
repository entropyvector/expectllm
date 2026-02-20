#!/usr/bin/env python3
"""Basic yes/no decision example."""

from expectllm import Conversation

c = Conversation()
c.send("Is Python dynamically typed? Reply YES or NO")

if c.expect_yesno():
    print("The LLM answered YES")
else:
    print("The LLM answered NO")
