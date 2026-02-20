#!/usr/bin/env python3
"""Extract structured JSON from text."""

from expectllm import Conversation

text = "Meeting with John at 3pm tomorrow in the conference room"

c = Conversation()
c.send(f"Parse this into JSON with fields (person, time, date, location):\n\n{text}")

data = c.expect_json()
print(f"Extracted: {data}")
