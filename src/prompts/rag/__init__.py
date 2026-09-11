"""RAG prompts module."""
from .query_rewrite import QUERY_REWRITE_PROMPT
from .rerank import RERANK_PROMPT

__all__ = ["QUERY_REWRITE_PROMPT", "RERANK_PROMPT"]
