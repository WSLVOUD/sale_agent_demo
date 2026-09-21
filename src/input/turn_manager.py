"""v2.7 Phase 3（§6~§9）：Turn Manager —— 后端聚合 + Turn 生命周期 + turn_id 生成。

计划 §6：Turn Manager 负责

    Message → 判断是否属于当前开放 Turn → 加入 Turn Buffer → 等待 → Seal → 生成 turn_id

计划 §7：**Turn 必须在进入 Agent 之前确定**（禁止 `orchestrator.process_message()`
里再生成 turn_id）—— 本模块是 turn_id 的唯一生成地。

计划 §8/§9：每个 session 维护一个 Open Turn；窗口内到达的消息进同一个 Turn
（M1 M2 M3 → T100，Agent 只执行一次）；生命周期

    OPEN → SEALED → PROCESSING → DECIDED → RESPONDED → COMMITTED

已 SEALED 的 Turn 不再接收消息，新消息进入下一个 Turn。
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .message import CustomerMessage, join_text
from .turn_store import OPEN, SEALED, TurnRecord, TurnStore, get_turn_store

DEFAULT_GRACE_MS = 400
DEFAULT_MAX_WINDOW_MS = 1800


def _env_ms(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default)) or default))
    except Exception:  # pragma: no cover - 防御式
        return default


def turn_grace_seconds() -> float:
    """静默多久算"这一轮说完了"（LED_RAG_TURN_GRACE_MS，默认 400ms）。"""
    return _env_ms("LED_RAG_TURN_GRACE_MS", DEFAULT_GRACE_MS) / 1000.0


def turn_max_window_seconds() -> float:
    """一个 Turn 最长能等多久（LED_RAG_TURN_MAX_WINDOW_MS，默认 1800ms）。"""
    return _env_ms("LED_RAG_TURN_MAX_WINDOW_MS", DEFAULT_MAX_WINDOW_MS) / 1000.0


@dataclass
class TurnBuffer:
    """一个会话当前正在聚合的 Turn。"""

    turn: TurnRecord
    messages: List[CustomerMessage] = field(default_factory=list)
    first_arrival: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    arrivals: int = 0

    def touch(self) -> None:
        self.last_activity = time.time()
        self.arrivals += 1

    def add(self, message: CustomerMessage) -> None:
        self.messages.append(message)
        if message.message_id and message.message_id not in self.turn.message_ids:
            self.turn.message_ids.append(message.message_id)
        for image in message.images or []:
            if image not in self.turn.images:
                self.turn.images.append(image)
        self.turn.text = join_text(self.messages)
        self.touch()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn.turn_id,
            "session_id": self.turn.session_id,
            "message_ids": list(self.turn.message_ids),
            "messages": self.arrivals,
            "status": self.turn.status,
            "open_ms": round((time.time() - self.first_arrival) * 1000, 1),
        }


class TurnManager:
    """按会话聚合消息 → 形成 Turn（线程安全；turn_id 在这里生成）。"""

    def __init__(
        self,
        store: Optional[TurnStore] = None,
        *,
        grace_seconds: Optional[float] = None,
        max_window_seconds: Optional[float] = None,
    ):
        self.store = store or get_turn_store()
        self._lock = threading.RLock()
        self._buffers: Dict[str, TurnBuffer] = {}
        self.grace_seconds = (
            turn_grace_seconds() if grace_seconds is None else max(0.0, float(grace_seconds))
        )
        self.max_window_seconds = (
            turn_max_window_seconds()
            if max_window_seconds is None
            else max(0.0, float(max_window_seconds))
        )

    # ── 聚合 ────────────────────────────────────────────────────────────
    def push_message(
        self,
        message: CustomerMessage,
        *,
        turn_id: str = "",
    ) -> Tuple[TurnRecord, bool]:
        """把一条消息放进当前 OPEN Turn（没有就新建，并生成 turn_id）。

        Returns:
            ``(turn, created)``；``created=True`` 表示这条消息**开了新的一轮**
            （调用者是 leader，负责 Seal 与执行）。
        """
        key = str(message.session_id or "")
        with self._lock:
            buffer = self._buffers.get(key)
            if buffer is not None and buffer.turn.status != OPEN:
                buffer = None  # 已 SEALED → 新消息进入下一个 Turn（§9.2）
            if buffer is None:
                turn = self.store.create(
                    key,
                    message_ids=[message.message_id] if message.message_id else [],
                    text=str(message.text or ""),
                    images=list(message.images or []),
                    source=message.source,
                    turn_id=turn_id,
                )
                buffer = TurnBuffer(turn=turn, messages=[message])
                buffer.touch()
                self._buffers[key] = buffer
                return turn, True

            buffer.add(message)
            self.store.update(
                buffer.turn.turn_id,
                message_ids=list(buffer.turn.message_ids),
                text=buffer.turn.text,
                images=list(buffer.turn.images),
            )
            return buffer.turn, False

    def push(
        self,
        session_id: str,
        *,
        message_id: str = "",
        text: str = "",
        images: Optional[List[Any]] = None,
        source: str = "api",
        turn_id: str = "",
    ) -> Tuple[TurnRecord, bool]:
        """兼容入口：给散装字段，内部合成一个 Message 对象再聚合。"""
        return self.push_message(
            CustomerMessage(
                message_id=message_id,
                session_id=str(session_id or ""),
                text=str(text or ""),
                images=list(images or []),
                source=str(source or "api"),
            ),
            turn_id=turn_id,
        )

    def current(self, session_id: str) -> Optional[TurnBuffer]:
        with self._lock:
            return self._buffers.get(str(session_id or ""))

    def buffer_of_turn(self, turn_id: str) -> Optional[TurnBuffer]:
        with self._lock:
            for buffer in self._buffers.values():
                if buffer.turn.turn_id == str(turn_id or ""):
                    return buffer
        return None

    def messages_of_turn(self, turn_id: str) -> List[CustomerMessage]:
        buffer = self.buffer_of_turn(turn_id)
        return list(buffer.messages) if buffer is not None else []

    # ── 封口 ────────────────────────────────────────────────────────────
    def should_seal(self, buffer: TurnBuffer, now: Optional[float] = None) -> bool:
        """静默超过 grace，或超过 max window → 封口。"""
        moment = time.time() if now is None else float(now)
        if moment - buffer.last_activity >= self.grace_seconds:
            return True
        return moment - buffer.first_arrival >= self.max_window_seconds

    def wait_until_sealed(self, buffer: TurnBuffer, *, poll_seconds: float = 0.02) -> TurnRecord:
        """leader 在这里等"这一轮说完了"（后端聚合的核心等待）。"""
        while True:
            now = time.time()
            with self._lock:
                current = self._buffers.get(buffer.turn.session_id)
                if current is None or current.turn.turn_id != buffer.turn.turn_id:
                    return self.store.get(buffer.turn.turn_id) or buffer.turn
                if self.should_seal(buffer, now):
                    break
            time.sleep(max(0.001, float(poll_seconds)))
        return self.seal(buffer)

    def seal(self, buffer: TurnBuffer) -> TurnRecord:
        with self._lock:
            buffer.turn.status = SEALED
            current = self._buffers.get(buffer.turn.session_id)
            if current is not None and current.turn.turn_id == buffer.turn.turn_id:
                self._buffers.pop(buffer.turn.session_id, None)
        return self.store.mark(buffer.turn.turn_id, SEALED) or buffer.turn

    def release(self, session_id: str, turn_id: str = "") -> None:
        with self._lock:
            current = self._buffers.get(str(session_id or ""))
            if current is not None and (not turn_id or current.turn.turn_id == turn_id):
                self._buffers.pop(str(session_id or ""), None)

    # ── 组装 ────────────────────────────────────────────────────────────
    @staticmethod
    def payload(turn: TurnRecord, messages: Optional[List[CustomerMessage]] = None) -> Dict[str, Any]:
        """Turn → 一次 Agent 执行需要的负载（messages 为对象列表）。"""
        items = list(messages or [])
        text = join_text(items) if items else "\n".join(
            part.strip() for part in str(turn.text or "").splitlines() if part.strip()
        )
        images = list(turn.images)
        if items:
            for message in items:
                for image in message.images or []:
                    if image not in images:
                        images.append(image)
        message_ids = [
            str(item)
            for item in (
                [message.message_id for message in items] or list(turn.message_ids)
            )
            if item
        ]
        return {
            "turn_id": turn.turn_id,
            "session_id": turn.session_id,
            "text": text,
            "images": images,
            "messages": items,
            "message_ids": message_ids,
            "message_count": max(1, len(message_ids)),
            "aggregated": len(message_ids) > 1,
            "source": turn.source,
        }


# 兼容旧名字（v2.7 早期叫 TurnBuilder；计划 §47 禁止新增第二套 Turn 系统）
TurnBuilder = TurnManager


__all__ = [
    "DEFAULT_GRACE_MS",
    "DEFAULT_MAX_WINDOW_MS",
    "TurnBuffer",
    "TurnBuilder",
    "TurnManager",
    "turn_grace_seconds",
    "turn_max_window_seconds",
]
