#!/usr/bin/env python3
"""Multi-turn code review example."""

import re

from llmexpect import Conversation

code = '''
def process_user(data):
    query = f"SELECT * FROM users WHERE id = {data['id']}"
    return db.execute(query)
'''

c = Conversation()
c.send(f"Review this code for security issues:\n```python\n{code}\n```")
c.expect(r"(sql injection|security|vulnerability)", re.IGNORECASE)

print("Issues found:")
print(c.last_response)

# Follow-up
c.send("How would you fix it?")
fixed_code = c.expect_code("python")
print("\nFixed code:")
print(fixed_code)
