"""RAG rerank prompt templates."""

RERANK_PROMPT = """You are a display product specialist. Evaluate the retrieved product candidates against the customer needs and generate a recommendation reason for each.

Customer needs:
{requirement}

Candidate products:
{candidates}

Important rules:
1. Each candidate contains raw product data. You must identify the model name yourself.
2. Environment match takes priority: if the requirement is outdoor, only rank outdoor LED products; if indoor, only rank indoor products.
3. Never reject any product — only rank and give positive recommendation reasons.
4. Only give positive reasons — do not mention any negative phrasing.
5. ALWAYS respond in English regardless of customer's language.

Output JSON only:
{{"selected_indices": [1, 2, 3], "reason": "recommendation reason"}}
"""
