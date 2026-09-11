"""Sales prompts module."""
from .classify import CLASSIFY_PROMPT, get_classify_prompt
from .requirement import REQUIREMENT_PROMPT, get_requirement_prompt
from .reply import GREETING_PROMPT, NEED_QUERY_PROMPT, get_reply_prompt

__all__ = [
    "CLASSIFY_PROMPT",
    "get_classify_prompt",
    "REQUIREMENT_PROMPT",
    "get_requirement_prompt",
    "GREETING_PROMPT",
    "NEED_QUERY_PROMPT",
    "get_reply_prompt",
]
