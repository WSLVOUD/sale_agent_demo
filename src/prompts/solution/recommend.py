"""Solution agent recommendation prompt templates."""

RECOMMEND_PROMPT = """Customer needs: {requirement}

Product info:
{product}

{additional_context}

Generate one natural, conversational recommendation line that includes:
1. Complete product model name
2. 1-2 core selling points / reasons to recommend
3. If necessary, 1-2 key specs (e.g. pixel pitch, dimensions)

Requirements:
- ALWAYS respond in English, regardless of customer's language
- Conversational, like recommending something to a friend — do not recite the manual
- Greeting is only needed on the first recommendation; subsequent recommendations start directly with the product
- Strictly no more than 60 characters, shorter is better
- No markdown, no lists, no bold
- Never reveal internal info: do not say "based on records", "I queried", "from the database", etc.
- Only recommend products you can confirm are a match
- Do not mention budget or price ranges
- 【Mandatory rule】Always output the complete model name — never write only the suffix
- Do not repeat needs the customer already stated

Output the recommendation directly (2-3 sentences max):"""


RECOMMEND_FOLLOW_UP = """Based on the recommended products, generate 1 short follow-up sentence.

Already recommended: {recommendations}

Requirements:
1. 1 sentence, conversational, like chatting with a friend
2. No emoji, no bold, no lists
3. No more than 15 words
4. Never ask about budget or price
5. Generate only from the content of already-recommended products
6. ALWAYS respond in English

Output directly:"""


def get_recommend_prompt(
    product: str,
    requirement: str,
    additional_context: str = "",
) -> str:
    """Format the recommendation prompt with product and requirement."""
    return RECOMMEND_PROMPT.format(
        requirement=requirement,
        product=product,
        additional_context=additional_context,
    )


def get_follow_up_prompt(recommendations: str) -> str:
    """Format the follow-up prompt."""
    return RECOMMEND_FOLLOW_UP.format(recommendations=recommendations)
