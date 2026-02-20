#!/usr/bin/env python3
"""Extract structured data using regex patterns."""

from expectllm import Conversation

text = "Contact John Smith at john@example.com or call 555-1234"

c = Conversation()
c.send(f"""Extract contact info from:
{text}

Format:
NAME: <name>
EMAIL: <email>
PHONE: <phone>""")

c.expect(r"NAME: (.+)\nEMAIL: (.+)\nPHONE: (.+)")

print(f"Name: {c.match.group(1)}")
print(f"Email: {c.match.group(2)}")
print(f"Phone: {c.match.group(3)}")
