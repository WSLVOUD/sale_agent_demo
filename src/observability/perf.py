"""v2.3.1 Phase 2：PerfTracker —— 每轮性能埋点（从 orchestrator 迁出）。

原来是 `orchestrator.py` 里的内部类；搬出来之后 Orchestrator 只 import 它。
逻辑一行未改。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class PerfTracker:
    """Lightweight performance tracker for one request cycle."""

    def __init__(self, session_id: str, message: str):
        self.session_id = session_id
        self.message = message
        self._t0: float = time.time()
        self._markers: Dict[str, float] = {"_start": time.time()}
        self.route: str = "unknown"
        self.solution_route: Optional[str] = None   # "fast" | "agent"
        self.intent: str = ""
        self.llm_calls: int = 0
        self.final_products: int = 0
        # ── v2.6 §27：统一可观测口径（一轮一个 turn_id）───────────────────
        self.turn_id: str = ""
        self.action: str = ""
        self.speech_act: str = ""
        self.question_slot: str = ""
        self.response_count: int = 0
        self.question_count: int = 0

    def mark(self, name: str) -> None:
        self._markers[name] = time.time()

    @property
    def total_ms(self) -> float:
        return (time.time() - self._t0) * 1000

    def live_llm_calls(self) -> int:
        """这一轮**到目前为止**的 LLM 调用数（还没 end_turn 时也能拿到真值）。

        实测问题：`PERF ... llm_calls=0` 而同一轮的 `[Turn] llm_calls=2` —— 因为
        PERF 那行是在 `end_turn()` 之前打的，只能看到手工计数器。日志口径必须一致，
        所以这里直接向 LLMCallTracker 要当前轮的真实计数。
        """
        try:
            from .llm_tracker import get_llm_tracker

            context = get_llm_tracker().current_turn()
            if context is not None:
                return int(context.stats().calls)
        except Exception:  # pragma: no cover - 统计失败不影响业务
            pass
        return int(self.llm_calls)

    def latency(self, after: str) -> float:
        """Milliseconds between marker 'after' and now."""
        t = self._markers.get(after)
        return (time.time() - t) * 1000 if t else 0.0

    def _span(self, a: str, b: str) -> float:
        """Milliseconds between two markers."""
        ta = self._markers.get(a)
        tb = self._markers.get(b)
        if ta is None or tb is None:
            return 0.0
        return (tb - ta) * 1000

    def summary(self) -> Dict[str, Any]:
        """Build a structured perf summary dict."""
        stats = self._llm_stats()
        return {
            "session_id": self.session_id,
            "route": self.route,
            "solution_route": self.solution_route,
            "intent": self.intent,
            "total_latency_ms": round(self.total_ms, 1),
            "sales_latency_ms": round(self._span("_start", "sales_done"), 1),
            "solution_latency_ms": round(self.latency("sales_done"), 1),
            # v2.5++++（计划 §15）：LLM 调用数由 LLMCallTracker 统一记账，
            # 手工 ++ 的计数只在没有登记簿时兜底（避免再出现"实际调了多次却 llm_calls=0"）
            "llm_calls": max(self.llm_calls, getattr(stats, "calls", 0)),
            "llm_latency_ms": round(getattr(stats, "latency_ms", 0.0), 1),
            "llm_tokens": getattr(stats, "total_tokens", 0),
            "llm_errors": getattr(stats, "errors", 0),
            "final_products": self.final_products,
            "first_contact_intro": getattr(self, "first_contact_intro", ""),
            "first_contact_messages": getattr(self, "first_contact_messages", []),
            # ── v2.6 §27 ───────────────────────────────────────────────────
            "turn_id": self.turn_id,
            "action": self.action,
            "speech_act": self.speech_act,
            "question_slot": self.question_slot,
            "response_count": self.response_count,
            "question_count": self.question_count,
        }

    def _llm_stats(self):
        """本轮（或刚结束的这一轮）的 LLM 统计。"""
        try:
            from .llm_tracker import get_llm_tracker

            tracker = get_llm_tracker()
            current = tracker.current_turn()
            if current is not None:
                return current.stats()
            stats = tracker.last_turn()
            if stats is not None and (
                not self.session_id or stats.session_id == self.session_id
            ):
                return stats
        except Exception as exc:  # pragma: no cover - 统计失败不影响业务
            logger.warning("LLM stats unavailable: %s", exc)
        return None
