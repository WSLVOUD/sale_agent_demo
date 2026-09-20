"""v2.5 Phase 1：UserTurn —— 多條客户消息聚合成一个"轮次"。

客户口径（2026-09-20）：客户连续发 3~5 条消息时，应该

    M1 M2 M3 M4 M5 → 一个 UserTurn → 一次需求理解 → 一次 Agent → 一次回复

而不是每条消息各触发一次 Agent。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class UserTurn:
    """一个 Turn 里聚起来的消息（保留原始 message_id）。"""

    session_id: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    text_parts: List[str] = field(default_factory=list)
    images: List[Any] = field(default_factory=list)
    message_ids: List[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    closed_at: Optional[float] = None

    # ── 常用视图 ────────────────────────────────────────────────────────
    @property
    def text(self) -> str:
        """聚合后的文字（按到达顺序拼接，去掉空白项）。"""
        return " ".join(part.strip() for part in self.text_parts if str(part).strip())

    @property
    def has_images(self) -> bool:
        return bool(self.images)

    @property
    def duration(self) -> float:
        return round((self.closed_at or time.time()) - self.started_at, 3)

    def add(self, message: Dict[str, Any]) -> None:
        self.messages.append(message)
        text = str(message.get("text") or message.get("content") or "").strip()
        if text:
            self.text_parts.append(text)
        for image in message.get("images") or []:
            if image not in self.images:
                self.images.append(image)
        image = message.get("image")
        if image and image not in self.images:
            self.images.append(image)
        message_id = str(message.get("message_id") or message.get("id") or "")
        if message_id:
            self.message_ids.append(message_id)

    def close(self) -> "UserTurn":
        self.closed_at = time.time()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "text": self.text,
            "images": list(self.images),
            "message_ids": list(self.message_ids),
            "message_count": len(self.messages),
            "duration": self.duration,
        }


__all__ = ["UserTurn"]
