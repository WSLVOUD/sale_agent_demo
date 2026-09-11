"""RAG query rewrite prompt templates."""

QUERY_REWRITE_PROMPT = """You are an information retrieval expert. Rewrite the user's question into a query better suited for retrieval.

Original question: {question}

Requirements:
1. Extract key concepts and terms
2. Expand possible synonyms
3. Keep the query concise
4. Remove colloquial expressions
5. ALWAYS respond in English regardless of input language

Return only the rewritten query, nothing else."""
