"""v2.7 Phase 1（§4）：Message Inbox —— 所有入口的统一消息登记处。

消息模型见 :mod:`src.input.message`（一条消息 = 一个对象）。

    RECEIVED → DEDUPLICATED → BUFFERED → ASSIGNED_TO_TURN → PROCESSED → COMMITTED

已经 PROCESSED / COMMITTED 的 message_id 再次到达 → 直接忽略，
**绝对不再进 Agent**（计划 §4.2）。
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from .message import (  # noqa: F401  (re-export：消息模型在 message.py)
    ALL_STATUSES,
    ASSIGNED_TO_TURN,
    BUFFERED,
    COMMITTED,
    DEDUPLICATED,
    FAILED,
    PROCESSED,
    RECEIVED,
    TERMINAL_STATUSES,
    CustomerMessage,
    new_message_id,
)


class MessageInbox:
    """按会话保存消息（进程内，线程安全）。"""

    def __init__(self, *, max_sessions: int = 512, max_keep_per_session: int = 200):
        self._lock = threading.RLock()
        self._messages: Dict[str, CustomerMessage] = {}
        self._by_session: Dict[str, List[str]] = {}
        self._by_turn: Dict[str, List[str]] = {}
        self._max_sessions = max(1, int(max_sessions))
        self._max_keep = max(10, int(max_keep_per_session))

    def receive(
        self,
        *,
        session_id: str,
        text: str = "",
        images: Optional[List[Any]] = None,
        message_id: str = "",
        source: str = "api",
        metadata: Optional[Dict[str, Any]] = None,
        status: str = RECEIVED,
    ) -> CustomerMessage:
        """登记一条消息（保证 message_id 唯一）。"""
        session = str(session_id or "")
        key = str(message_id or "").strip() or new_message_id()
        message = CustomerMessage(
            message_id=key,
            session_id=session,
            text=str(text or ""),
            images=list(images or []),
            source=str(source or "api"),
            status=str(status or RECEIVED),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._prune()
            self._messages[key] = message
            self._by_session.setdefault(session, []).append(key)
            if len(self._by_session[session]) > self._max_keep:
                dropped = self._by_session[session][:-self._max_keep]
                self._by_session[session] = self._by_session[session][-self._max_keep:]
                for old in dropped:
                    self._messages.pop(old, None)
        return message

    def mark(self, message_id: str, status: str, *, turn_id: str = "") -> Optional[CustomerMessage]:
        key = str(message_id or "")
        with self._lock:
            message = self._messages.get(key)
            if message is None:
                return None
            message.status = str(status or message.status)
            if turn_id:
                message.turn_id = str(turn_id)
                ids = self._by_turn.setdefault(str(turn_id), [])
                if key not in ids:
                    ids.append(key)
            return message

    def receive_message(self, message: CustomerMessage) -> CustomerMessage:
        """登记一个已经建好的 Message 对象（§4.3 的对象化消息）。"""
        return self.receive(
            session_id=message.session_id,
            text=message.text,
            images=list(message.images or []),
            message_id=message.message_id,
            source=message.source,
            metadata=dict(message.metadata or {}),
            status=message.status,
        )

    # ── 查询 ────────────────────────────────────────────────────────────
    def get(self, message_id: str) -> Optional[CustomerMessage]:
        with self._lock:
            return self._messages.get(str(message_id or ""))

    def status_of(self, message_id: str) -> str:
        message = self.get(message_id)
        return str(message.status) if message else ""

    def is_finished(self, message_id: str) -> bool:
        return self.status_of(message_id) in TERMINAL_STATUSES

    def messages_for_turn(self, turn_id: str) -> List[CustomerMessage]:
        with self._lock:
            ids = list(self._by_turn.get(str(turn_id or ""), []))
            return [self._messages[key] for key in ids if key in self._messages]

    def session_messages(self, session_id: str) -> List[CustomerMessage]:
        with self._lock:
            ids = list(self._by_session.get(str(session_id or ""), []))
            return [self._messages[key] for key in ids if key in self._messages]

    def reset(self, session_id: str = "") -> None:
        with self._lock:
            if not session_id:
                self._messages.clear()
                self._by_session.clear()
                self._by_turn.clear()
                return
            for key in self._by_session.pop(str(session_id), []):
                message = self._messages.pop(key, None)
                if message and message.turn_id:
                    ids = self._by_turn.get(message.turn_id)
                    if ids and key in ids:
                        ids.remove(key)

    def _prune(self) -> None:
        if len(self._by_session) <= self._max_sessions:
            return
        for session in list(self._by_session)[: max(1, len(self._by_session) - self._max_sessions)]:
            for key in self._by_session.pop(session, []):
                self._messages.pop(key, None)


INBOX = MessageInbox()


def get_message_inbox() -> MessageInbox:
    return INBOX


__all__ = [
    "ALL_STATUSES",
    "ASSIGNED_TO_TURN",
    "BUFFERED",
    "COMMITTED",
    "CustomerMessage",
    "DEDUPLICATED",
    "FAILED",
    "INBOX",
    "MessageInbox",
    "PROCESSED",
    "RECEIVED",
    "TERMINAL_STATUSES",
    "get_message_inbox",
    "new_message_id",
]
