"""Solution prompts module."""
from .intent import INTENT_PROMPT, get_intent_prompt
from .recommend import RECOMMEND_PROMPT, get_recommend_prompt, get_follow_up_prompt
from .reflection import REFLECTION_PROMPT, get_reflection_prompt

__all__ = [
    "INTENT_PROMPT",
    "get_intent_prompt",
    "RECOMMEND_PROMPT",
    "get_recommend_prompt",
    "get_follow_up_prompt",
    "REFLECTION_PROMPT",
    "get_reflection_prompt",
]
