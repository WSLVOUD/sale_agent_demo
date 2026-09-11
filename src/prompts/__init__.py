"""Prompts module for prompt templates.

All prompts are stored as Python variables for easy import and modification.
"""
from .common.system import SYSTEM_PROMPT
from .common.format import FORMAT_PROMPT
from .sales.classify import CLASSIFY_PROMPT, get_classify_prompt
from .sales.requirement import REQUIREMENT_PROMPT, get_requirement_prompt
from .sales.reply import GREETING_PROMPT, NEED_QUERY_PROMPT, get_reply_prompt
from .solution.intent import INTENT_PROMPT, get_intent_prompt
from .solution.recommend import RECOMMEND_PROMPT, get_recommend_prompt
from .solution.reflection import REFLECTION_PROMPT, get_reflection_prompt
from .rag.query_rewrite import QUERY_REWRITE_PROMPT
from .rag.rerank import RERANK_PROMPT

__all__ = [
    "SYSTEM_PROMPT",
    "FORMAT_PROMPT",
    "CLASSIFY_PROMPT",
    "get_classify_prompt",
    "REQUIREMENT_PROMPT",
    "get_requirement_prompt",
    "GREETING_PROMPT",
    "NEED_QUERY_PROMPT",
    "get_reply_prompt",
    "INTENT_PROMPT",
    "get_intent_prompt",
    "RECOMMEND_PROMPT",
    "get_recommend_prompt",
    "REFLECTION_PROMPT",
    "get_reflection_prompt",
    "QUERY_REWRITE_PROMPT",
    "RERANK_PROMPT",
]
