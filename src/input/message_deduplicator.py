"""v2.7 Phase 2（§5）：Message Deduplication —— 消息级去重。

计划 §5.1：``message_id`` 已经处理过（PROCESSED）→ 不再进 Agent。
计划 §5.2：**不允许只使用内存 Dict** —— 去重状态必须落在可替换的持久化接口
（:mod:`src.input.message_store`：内存 / SQLite / 将来的 Redis / PostgreSQL）。

    ① message_id（主）：外部平台 ID（WhatsApp / n8n / Webhook）或后端生成；
    ② payload 指纹（辅）：同一会话、短时间内**完全相同**的文字 + 图片 → 视为重试。

计划 §47 明确：不能把并发/重复问题交给 Prompt 或多加一个 if。
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .message_store import (
    KIND_FINGERPRINT,
    KIND_MESSAGE_ID,
    MessageStore,
    get_message_store,
)

DEFAULT_FINGERPRINT_TTL_SECONDS = 600.0


@dataclass
class DedupDecision:
    """去重结论。"""

    is_duplicate: bool = False
    reason: str = ""
    message_id: str = ""
    fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_duplicate": self.is_duplicate,
            "reason": self.reason,
            "message_id": self.message_id,
            "fingerprint": self.fingerprint,
        }


class MessageDeduplicator:
    """消息去重器（状态存在 MessageStore 里，可换持久化实现）。"""

    def __init__(
        self,
        *,
        store: Optional[MessageStore] = None,
        fingerprint_ttl: float = DEFAULT_FINGERPRINT_TTL_SECONDS,
    ):
        self.store = store or get_message_store()
        self._ttl = max(0.0, float(fingerprint_ttl))

    def check(
        self,
        session_id: str,
        *,
        message_id: str = "",
        text: str = "",
        images: Optional[List[Any]] = None,
    ) -> DedupDecision:
        """判断这条消息是不是"已经见过"。"""
        key = str(message_id or "").strip()
        fingerprint = payload_fingerprint(session_id, text, images)
        if key and self.store.is_seen(key, kind=KIND_MESSAGE_ID):
            return DedupDecision(True, "duplicate_message_id", key, fingerprint)
        if self._ttl > 0 and self.store.is_seen(
            fingerprint, kind=KIND_FINGERPRINT, ttl=self._ttl
        ):
            return DedupDecision(True, "duplicate_payload_fingerprint", key, fingerprint)
        return DedupDecision(False, "", key, fingerprint)

    def remember(
        self,
        session_id: str,
        *,
        message_id: str = "",
        text: str = "",
        images: Optional[List[Any]] = None,
        fingerprint: str = "",
    ) -> None:
        key = str(message_id or "").strip()
        fp = fingerprint or payload_fingerprint(session_id, text, images)
        if key:
            self.store.add(key, kind=KIND_MESSAGE_ID, session_id=str(session_id or ""))
        if fp:
            self.store.add(fp, kind=KIND_FINGERPRINT, session_id=str(session_id or ""))

    def forget(self, *, message_id: str = "", fingerprint: str = "") -> None:
        """执行失败时撤掉登记（否则客户重试会被当成重复丢弃）。"""
        if message_id:
            self.store.forget(str(message_id), kind=KIND_MESSAGE_ID)
        if fingerprint:
            self.store.forget(str(fingerprint), kind=KIND_FINGERPRINT)

    def reset(self) -> None:
        self.store.reset()


def payload_fingerprint(session_id: str, text: str, images: Optional[List[Any]] = None) -> str:
    """同一会话 + 同一文字 + 同一图片集合 = 同一个指纹。"""
    hasher = hashlib.sha1()
    hasher.update(str(session_id or "").encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(" ".join(str(text or "").split()).lower().encode("utf-8"))
    for image in images or []:
        hasher.update(b"\x01")
        hasher.update(str(image)[:512].encode("utf-8", "ignore"))
    return hasher.hexdigest()[:32]


DEDUPLICATOR = MessageDeduplicator()


def get_message_deduplicator() -> MessageDeduplicator:
    return DEDUPLICATOR


__all__ = [
    "DEFAULT_FINGERPRINT_TTL_SECONDS",
    "DEDUPLICATOR",
    "DedupDecision",
    "MessageDeduplicator",
    "get_message_deduplicator",
    "payload_fingerprint",
]
