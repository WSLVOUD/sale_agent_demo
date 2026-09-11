"""Solution agent intent recognition prompt templates."""

INTENT_PROMPT = """Analyze the user message and determine the intent:

- recommendation: user wants product recommendations or describes needs (I need a meeting room display, I want to buy an advertising screen, help me pick one)
- product_question: user is asking about specific product specs, features, or comparisons
- conversation: user is chatting or greeting
- others: any other unclassifiable questions

Rules:
1. Need descriptions take priority → recommendation
2. Specific model spec questions → product_question
3. Greetings, thanks, etc. → conversation
4. Company info, hours, contact details → others

IMPORTANT: Always respond in English regardless of customer's language.

Return the intent type only, nothing else."""


def get_intent_prompt(message: str) -> str:
    """Format the intent prompt with the user message."""
    return f"{INTENT_PROMPT}\n\nUser message: {message}"
