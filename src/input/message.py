"""v2.7 Phase 1（§4）：统一消息模型。

计划 §4.1/§4.2/§4.3：

    message_id + session_id + source + text + images + timestamp + metadata

并且明确要求：**禁止 text 和 message_id 分成两个独立数组**（容易错位）——
一条消息就是一个对象，`messages = [{message_id, text, images, timestamp}, ...]`。

message_id 优先用外部平台的（WhatsApp / n8n / Webhook），没有就后端生成。
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

# ── 消息状态机（§4.2 / §9）──────────────────────────────────────────────
RECEIVED = "RECEIVED"
DEDUPLICATED = "DEDUPLICATED"
BUFFERED = "BUFFERED"
ASSIGNED_TO_TURN = "ASSIGNED_TO_TURN"
PROCESSED = "PROCESSED"
COMMITTED = "COMMITTED"
FAILED = "FAILED"

ALL_STATUSES = (
    RECEIVED, DEDUPLICATED, BUFFERED, ASSIGNED_TO_TURN, PROCESSED, COMMITTED, FAILED,
)
TERMINAL_STATUSES = frozenset({PROCESSED, COMMITTED})


def new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:16]}"


@dataclass
class CustomerMessage:
    """一条客户消息（自带 message_id 与它自己的全部内容）。"""

    message_id: str = ""
    session_id: str = ""
    text: str = ""
    images: List[Any] = field(default_factory=list)
    source: str = "api"
    timestamp: float = field(default_factory=time.time)
    status: str = RECEIVED
    turn_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.message_id or "").strip():
            self.message_id = new_message_id()
        self.timestamp = float(getattr(self, "received_at", None) or self.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "session_id": self.session_id,
            "text": str(self.text)[:200],
            "image_count": len(self.images),
            "source": self.source,
            "status": self.status,
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
        }


def message_from_part(
    part: Any, *, session_id: str = "", source: str = "api"
) -> Optional[CustomerMessage]:
    """把请求里的一个 part 变成 Message 对象（一条消息 = 一个对象）。"""
    if isinstance(part, CustomerMessage):
        return part
    if isinstance(part, dict):
        text = str(part.get("text") or part.get("question") or "").strip()
        images = list(part.get("images") or [])
        message_id = str(part.get("message_id") or part.get("id") or "")
        timestamp = part.get("timestamp") or part.get("received_at") or 0
        metadata = dict(part.get("metadata") or {})
    else:  # pragma: no cover - 防御式
        text = str(getattr(part, "text", "") or "").strip()
        images = list(getattr(part, "images", None) or [])
        message_id = str(getattr(part, "message_id", "") or "")
        timestamp = float(getattr(part, "timestamp", 0) or 0)
        metadata = dict(getattr(part, "metadata", {}) or {})
    if not text and not images:
        return None
    message = CustomerMessage(
        message_id=message_id,
        session_id=str(session_id or ""),
        text=text,
        images=images,
        source=str(source or "api"),
        metadata=metadata,
    )
    if timestamp:
        message.timestamp = float(timestamp)
    return message


def collect_messages(
    *,
    session_id: str = "",
    source: str = "api",
    question: str = "",
    images: Optional[Iterable[Any]] = None,
    messages: Optional[Iterable[Any]] = None,
    message_ids: Optional[Iterable[str]] = None,
) -> List[CustomerMessage]:
    """把一次请求统一成 Message 列表（老客户端只发 question/images 也支持）。

    实测 bug（2026-09-21 真实日志）：前端同时发 `messages[]` 和兼容字段
    `question`（内容相同），旧实现把两者都收下 → 同一句话变成 2 条消息
    （日志里 `message_ids=["m-…-0","m-…-0"]`、文本 `"hi\\nhi"`）。
    这里按 message_id 去重，并且**兼容字段只在前端没发 messages 时使用**。
    """
    items: List[CustomerMessage] = []
    for part in messages or []:
        message = message_from_part(part, session_id=session_id, source=source)
        if message is not None:
            items.append(message)
    # 同一个 message_id 重复提交（前端重试 / 同一请求里重复放）只算一条
    seen_ids = set()
    unique: List[CustomerMessage] = []
    for message in items:
        key = str(message.message_id or "")
        if key and key in seen_ids:
            continue
        if key:
            seen_ids.add(key)
        unique.append(message)
    items = unique

    legacy_images = [image for image in (images or []) if image]
    legacy_text = str(question or "").strip()
    if items:
        # `messages[]` 已经带过内容 → 兼容字段只是同一批消息的冗余副本
        covered_text = " ".join(" ".join(item.text.split()) for item in items)
        legacy_covered = (
            not legacy_text
            or " ".join(legacy_text.split()) in covered_text
            or covered_text in " ".join(legacy_text.split())
        )
        images_covered = all(image in item.images for item in items for image in legacy_images)
        if legacy_covered and images_covered:
            return items
    if legacy_text or legacy_images:
        legacy_id = next((str(item) for item in (message_ids or []) if item), "")
        if legacy_id and legacy_id in seen_ids:
            return items
        items = items + [
            CustomerMessage(
                message_id=legacy_id,
                session_id=str(session_id or ""),
                text=legacy_text,
                images=legacy_images,
                source=str(source or "api"),
            )
        ]
    return [item for item in items if str(item.text).strip() or item.images]


def join_text(messages: Iterable[CustomerMessage]) -> str:
    """把多条消息的文本按到达顺序拼成这一轮的输入（Agent 只读一次）。"""
    return "\n".join(
        str(getattr(message, "text", "") or "").strip()
        for message in messages
        if str(getattr(message, "text", "") or "").strip()
    )


__all__ = [
    "ALL_STATUSES",
    "ASSIGNED_TO_TURN",
    "BUFFERED",
    "COMMITTED",
    "CustomerMessage",
    "DEDUPLICATED",
    "FAILED",
    "PROCESSED",
    "RECEIVED",
    "TERMINAL_STATUSES",
    "collect_messages",
    "join_text",
    "message_from_part",
    "new_message_id",
]
