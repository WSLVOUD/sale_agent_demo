"""v2.5++++（消息聚合与自然对话链路优化 · 第十阶段）：LLM 调用统一统计。

实测问题：一轮里明明打了 4~5 次 DeepSeek，日志却是 `llm_calls=0` ——
因为只有 Orchestrator 里手工 `llm_calls += 1` 的那一处被统计到，
Sales 图内部（意图分类 / 需求抽取 / 话术生成 / 兜底改写）全都漏掉了。

本模块在 **LLM 调用入口**统一记账（`src/core/llm.get_llm` 返回的实例都会经过这里）：

    call_id / session_id / turn_id / agent / node / model
    start_time / end_time / latency_ms
    prompt_tokens / completion_tokens / total_tokens
    success / error

一轮（turn）结束后可以得到：

    turn_llm_calls / turn_llm_latency_ms / turn_tokens / turn_errors
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMCallRecord:
    """一次 LLM 调用的完整记录。"""

    call_id: str = ""
    session_id: str = ""
    turn_id: str = ""
    agent: str = ""
    node: str = ""
    model: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    success: bool = True
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "agent": self.agent,
            "node": self.node,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 1),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "success": self.success,
            "error": self.error[:200],
        }


@dataclass
class TurnStats:
    """一轮的 LLM 统计。"""

    turn_id: str = ""
    session_id: str = ""
    calls: int = 0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    errors: int = 0
    records: List[LLMCallRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "llm_calls": self.calls,
            "llm_latency_ms": round(self.latency_ms, 1),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_errors": self.errors,
        }


@dataclass
class TurnContext:
    """当前正在处理的一轮（挂在 ContextVar 上，线程/协程各自独立）。"""

    turn_id: str
    session_id: str = ""
    message_count: int = 1
    aggregated: bool = False
    started_at: float = field(default_factory=time.time)
    records: List[LLMCallRecord] = field(default_factory=list)

    def stats(self) -> TurnStats:
        stats = TurnStats(turn_id=self.turn_id, session_id=self.session_id)
        for record in self.records:
            stats.calls += 1
            stats.latency_ms += record.latency_ms
            stats.prompt_tokens += record.prompt_tokens
            stats.completion_tokens += record.completion_tokens
            stats.total_tokens += record.total_tokens
            if not record.success:
                stats.errors += 1
        stats.records = list(self.records)
        return stats


_current_turn: ContextVar[Optional[TurnContext]] = ContextVar(
    "led_rag_llm_turn", default=None
)


class LLMCallTracker:
    """LLM 调用登记簿（线程安全）。"""

    def __init__(self, keep_turns: int = 50) -> None:
        self._lock = threading.RLock()
        self._history: List[TurnStats] = []
        self._keep_turns = max(1, int(keep_turns))

    # ── turn 生命周期 ───────────────────────────────────────────────────
    def begin_turn(
        self,
        session_id: str = "",
        *,
        turn_id: str = "",
        message_count: int = 1,
        aggregated: bool = False,
    ) -> TurnContext:
        context = TurnContext(
            turn_id=turn_id or f"turn_{uuid.uuid4().hex[:12]}",
            session_id=str(session_id or ""),
            message_count=max(1, int(message_count or 1)),
            aggregated=bool(aggregated),
        )
        _current_turn.set(context)
        return context

    def current_turn(self) -> Optional[TurnContext]:
        return _current_turn.get()

    def set_actor(self, *, agent: str = "", node: str = "") -> None:
        """标记"接下来的这次调用是谁发起的"（可选，便于排查）。"""
        context = _current_turn.get()
        if context is None:
            return
        self._actor.set((str(agent or ""), str(node or "")))

    _actor: ContextVar[tuple] = ContextVar("led_rag_llm_actor", default=("", ""))

    def end_turn(self) -> Optional[TurnStats]:
        context = _current_turn.get()
        if context is None:
            return None
        stats = context.stats()
        with self._lock:
            self._history.append(stats)
            while len(self._history) > self._keep_turns:
                self._history.pop(0)
        _current_turn.set(None)
        self._actor.set(("", ""))
        return stats

    def last_turn(self) -> Optional[TurnStats]:
        with self._lock:
            return self._history[-1] if self._history else None

    def history(self) -> List[TurnStats]:
        with self._lock:
            return list(self._history)

    def reset(self) -> None:
        with self._lock:
            self._history.clear()
        _current_turn.set(None)

    # ── 记账 ────────────────────────────────────────────────────────────
    def record(self, record: LLMCallRecord) -> None:
        context = _current_turn.get()
        if context is not None:
            if not record.turn_id:
                record.turn_id = context.turn_id
            if not record.session_id:
                record.session_id = context.session_id
            if not record.agent:
                record.agent = self._actor.get()[0]
            if not record.node:
                record.node = self._actor.get()[1]
            with self._lock:
                context.records.append(record)
        if not record.success:
            logger.warning(
                "[LLM] call failed model=%s latency_ms=%.1f error=%s",
                record.model, record.latency_ms, record.error[:120],
            )

    def track(self, *, model: str = "", agent: str = "", node: str = ""):
        """上下文管理器：with tracker.track(model=...) as holder: ..."""
        return _TrackContext(self, model=model, agent=agent, node=node)

    # ── 便捷统计 ────────────────────────────────────────────────────────
    def turn_calls(self) -> int:
        context = _current_turn.get()
        return len(context.records) if context is not None else 0

    def turn_latency_ms(self) -> float:
        context = _current_turn.get()
        if context is None:
            return 0.0
        return sum(record.latency_ms for record in context.records)


class _TrackContext:
    """把一次调用包起来并自动记账。"""

    def __init__(self, tracker: LLMCallTracker, *, model="", agent="", node=""):
        self._tracker = tracker
        self._record = LLMCallRecord(
            call_id=f"call_{uuid.uuid4().hex[:10]}",
            model=str(model or ""),
            agent=str(agent or ""),
            node=str(node or ""),
        )

    def __enter__(self) -> LLMCallRecord:
        self._record.start_time = time.time()
        return self._record

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._record.end_time = time.time()
        self._record.latency_ms = (self._record.end_time - self._record.start_time) * 1000
        self._record.success = exc is None
        if exc is not None:
            self._record.error = f"{exc_type.__name__}: {exc}" if exc_type else str(exc)
        self._tracker.record(self._record)
        return False  # 不吞异常


_tracker_instance: Optional[LLMCallTracker] = None
_tracker_lock = threading.Lock()


def get_llm_tracker() -> LLMCallTracker:
    """全局登记簿（进程内单例）。"""
    global _tracker_instance
    with _tracker_lock:
        if _tracker_instance is None:
            _tracker_instance = LLMCallTracker()
        return _tracker_instance


def track_usage(record: LLMCallRecord, response: Any) -> None:
    """从 LangChain 的返回里取出 token 用量（各家字段名不同，能取多少取多少）。"""
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        record.prompt_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        record.completion_tokens = int(
            usage.get("output_tokens") or usage.get("completion_tokens") or 0
        )
        record.total_tokens = int(
            usage.get("total_tokens")
            or (record.prompt_tokens + record.completion_tokens)
        )
        return
    metadata = getattr(response, "response_metadata", None) or {}
    token_usage = metadata.get("token_usage") or {}
    if isinstance(token_usage, dict) and token_usage:
        record.prompt_tokens = int(token_usage.get("prompt_tokens") or 0)
        record.completion_tokens = int(token_usage.get("completion_tokens") or 0)
        record.total_tokens = int(
            token_usage.get("total_tokens")
            or (record.prompt_tokens + record.completion_tokens)
        )


__all__ = [
    "LLMCallRecord",
    "LLMCallTracker",
    "TurnContext",
    "TurnStats",
    "get_llm_tracker",
    "track_usage",
]
