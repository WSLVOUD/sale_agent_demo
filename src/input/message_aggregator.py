"""v2.5 Phase 1：Message Aggregator（debounce + max window + 幂等 + 会话锁）。

客户口径（2026-09-20）：连续多条消息要**聚合成一个 UserTurn**，只触发一次 Agent。

    WhatsApp / API → Message → Message Aggregator → UserTurn → LangGraph

规则（计划第四节）：
    · debounce 1.5~2s：客户停止发送达到 debounce → 生成 UserTurn
    · max aggregation window 6~8s：超过窗口一定要发出去（不能无限等）
    · message_id 幂等：同一个 message_id 不重复处理
    · session lock：同一个会话同时只有一个 UserTurn 在处理
    · 支持 文字+文字 / 文字+图片 / 图片+图片 / 文字+图片+文字

本模块是纯逻辑（不依赖具体传输层），方便单测：

    aggregator = MessageAggregator(debounce_seconds=0, max_window_seconds=0)
    turn = aggregator.add(session_id, {"message_id": "m1", "text": "hi"})
    if turn is not None: ...  # 到点了，把 turn 交给 Agent
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .user_turn import UserTurn

logger = logging.getLogger(__name__)

# v2.5++++（《消息聚合与自然对话链路优化计划》§4.2）：**600ms 静默期 + 1800ms 上限**
# 客户连续快速发消息时尽量合成一个 turn；单条消息最多多等 0.6s，不会有明显延迟。
def _env_seconds(name: str, default: float) -> float:
    import os

    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(0.0, float(raw) / 1000.0)   # 环境变量按毫秒配
    except ValueError:
        return default


DEFAULT_DEBOUNCE_SECONDS = _env_seconds("LED_RAG_TURN_DEBOUNCE_MS", 0.6)
DEFAULT_MAX_WINDOW_SECONDS = _env_seconds("LED_RAG_TURN_MAX_WINDOW_MS", 1.8)


@dataclass
class _SessionBuffer:
    turn: Optional[UserTurn] = None
    last_message_at: float = 0.0
    first_message_at: float = 0.0
    processing: bool = False


class MessageAggregator:
    """把同一个会话里连续到达的消息聚成一个 UserTurn。"""

    def __init__(
        self,
        *,
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
        max_window_seconds: float = DEFAULT_MAX_WINDOW_SECONDS,
        now: Optional[Callable[[], float]] = None,
    ) -> None:
        self.debounce = float(debounce_seconds)
        self.max_window = float(max_window_seconds)
        self._now = now or time.time
        self._buffers: Dict[str, _SessionBuffer] = {}
        self._seen_ids: Dict[str, float] = {}
        self._lock = threading.RLock()

    # ── 主入口 ──────────────────────────────────────────────────────────
    def add(self, session_id: str, message: Dict[str, Any]) -> Optional[UserTurn]:
        """加入一条消息；如果"这一轮可以发出去了"就返回聚合好的 UserTurn。

        返回 None 表示还在等待（debounce 未到 / 已有 turn 在处理中）。
        """
        session = str(session_id or "")
        message_id = str(message.get("message_id") or message.get("id") or "")
        with self._lock:
            if message_id and message_id in self._seen_ids:
                logger.info("[Aggregator] 重复 message_id=%s → 忽略", message_id)
                return None
            if message_id:
                self._seen_ids[message_id] = self._now()
                self._prune_seen()

            buffer = self._buffers.setdefault(session, _SessionBuffer())
            now = self._now()
            completed: Optional[UserTurn] = None
            # 客户停顿超过 debounce 又发来新消息 → 上一轮就此结束（先把它发出去），
            # 新消息开始下一轮（这样"几条连续消息"才不会被拆进一个无限增长的 turn）。
            # 注意：会话还在处理中时只把消息并进缓冲，不在这里发出。
            if (
                buffer.turn is not None
                and now - buffer.last_message_at >= self.debounce
                and not buffer.processing
            ):
                completed = self._flush(session)
            if buffer.turn is None:
                buffer.turn = UserTurn(session_id=session, started_at=now)
                buffer.first_message_at = now
            buffer.turn.add(message)
            buffer.last_message_at = now

            if completed is not None:
                return completed
            # 会话锁：同一个会话的一个 turn 还在处理时，后续消息并入下一个 turn
            if buffer.processing:
                return None

            if not self._ready(buffer, now):
                return None
            return self._flush(session)

    def flush(self, session_id: str) -> Optional[UserTurn]:
        """强制把某个会话现在的缓冲发出去（测试 / 会话结束 / max window 兜底）。"""
        with self._lock:
            buffer = self._buffers.get(str(session_id or ""))
            if buffer is None or buffer.turn is None:
                return None
            if buffer.processing:
                return None
            return self._flush(str(session_id or ""))

    def pending_sessions(self) -> List[str]:
        with self._lock:
            return [
                session for session, buffer in self._buffers.items()
                if buffer.turn is not None and not buffer.processing
            ]

    def mark_processing(self, session_id: str, processing: bool) -> None:
        """会话锁：标记某个会话是否正在跑 Agent（同一会话不会并发跑两次）。"""
        with self._lock:
            buffer = self._buffers.setdefault(str(session_id or ""), _SessionBuffer())
            buffer.processing = bool(processing)

    def reset(self, session_id: str = "") -> None:
        with self._lock:
            if session_id:
                self._buffers.pop(str(session_id), None)
            else:
                self._buffers.clear()
                self._seen_ids.clear()

    def mark_seen(self, session_id: str, message_id: str) -> bool:
        """幂等登记：返回 True 表示这个 message_id 之前已经处理过（重复提交）。"""
        key = str(message_id or "")
        if not key:
            return False
        with self._lock:
            if key in self._seen_ids:
                return True
            self._seen_ids[key] = self._now()
            self._prune_seen()
            return False

    # ── 内部 ────────────────────────────────────────────────────────────
    def _ready(self, buffer: _SessionBuffer, now: float) -> bool:
        if buffer.turn is None:
            return False
        since_last = now - buffer.last_message_at
        since_first = now - buffer.first_message_at
        # max window 到点 → 必须发出；否则等 debounce
        return since_last >= self.debounce or since_first >= self.max_window

    def _flush(self, session_id: str) -> Optional[UserTurn]:
        buffer = self._buffers.get(session_id)
        if buffer is None or buffer.turn is None:
            return None
        turn = buffer.turn.close()
        buffer.turn = None
        buffer.first_message_at = 0.0
        logger.info(
            "[Aggregator] session=%s → UserTurn(messages=%d, images=%d, text=%r)",
            session_id, len(turn.messages), len(turn.images), turn.text[:60],
        )
        return turn

    def _prune_seen(self, keep_seconds: float = 600.0) -> None:
        now = self._now()
        stale = [key for key, stamp in self._seen_ids.items() if now - stamp > keep_seconds]
        for key in stale:
            self._seen_ids.pop(key, None)


__all__ = ["DEFAULT_DEBOUNCE_SECONDS", "DEFAULT_MAX_WINDOW_SECONDS", "MessageAggregator"]
