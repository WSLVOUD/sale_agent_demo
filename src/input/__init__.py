"""v2.5：输入层（多消息聚合 → UserTurn）。"""
from .message_aggregator import (
    DEFAULT_DEBOUNCE_SECONDS,
    DEFAULT_MAX_WINDOW_SECONDS,
    MessageAggregator,
)
from .user_turn import UserTurn
from .message import CustomerMessage, collect_messages, join_text, new_message_id
from .message_inbox import MessageInbox, get_message_inbox
from .message_store import (
    InMemoryMessageStore,
    MessageStore,
    SQLiteMessageStore,
    get_message_store,
)
from .message_deduplicator import (
    DedupDecision,
    MessageDeduplicator,
    get_message_deduplicator,
)
from .turn_store import TurnRecord, TurnStore, get_turn_store
from .turn_manager import TurnBuffer, TurnBuilder, TurnManager
from .turn_executor import TurnExecutor, TurnOutcome

__all__ = [
    "DEFAULT_DEBOUNCE_SECONDS",
    "DEFAULT_MAX_WINDOW_SECONDS",
    "CustomerMessage",
    "DedupDecision",
    "InMemoryMessageStore",
    "MessageDeduplicator",
    "MessageInbox",
    "MessageAggregator",
    "MessageStore",
    "SQLiteMessageStore",
    "TurnBuffer",
    "TurnBuilder",
    "TurnExecutor",
    "TurnManager",
    "TurnOutcome",
    "TurnRecord",
    "TurnStore",
    "UserTurn",
    "collect_messages",
    "get_message_deduplicator",
    "get_message_inbox",
    "get_message_store",
    "get_turn_store",
    "join_text",
    "new_message_id",
]
