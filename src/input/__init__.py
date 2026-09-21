"""v2.5：输入层（多消息聚合 → UserTurn）。"""
from .message_aggregator import (
    DEFAULT_DEBOUNCE_SECONDS,
    DEFAULT_MAX_WINDOW_SECONDS,
    MessageAggregator,
)
from .user_turn import UserTurn
from .turn_payload import TurnPayload, merge_message_parts, merge_request_payload

__all__ = [
    "DEFAULT_DEBOUNCE_SECONDS",
    "DEFAULT_MAX_WINDOW_SECONDS",
    "MessageAggregator",
    "TurnPayload",
    "UserTurn",
    "merge_message_parts",
    "merge_request_payload",
]
