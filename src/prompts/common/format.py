"""Format prompt templates."""

FORMAT_PROMPT = """
Response format:
- Plain text only, no markdown
- Concise and direct
- Conversational, like chatting with a friend
"""

JSON_FORMAT_PROMPT = """
Return JSON format:
{schema}

Return JSON only, nothing else.
"""

LIST_FORMAT_PROMPT = """
Reply in the following format (one per line):
{items}

Output the list directly, nothing else.
"""
