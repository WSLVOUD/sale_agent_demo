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

    def mark(self, name: str) -> None:
        self._markers[name] = time.time()

    @property
    def total_ms(self) -> float:
        return (time.time() - self._t0) * 1000

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
        return {
            "session_id": self.session_id,
            "route": self.route,
            "solution_route": self.solution_route,
            "intent": self.intent,
            "total_latency_ms": round(self.total_ms, 1),
            "sales_latency_ms": round(self._span("_start", "sales_done"), 1),
            "solution_latency_ms": round(self.latency("sales_done"), 1),
            "llm_calls": self.llm_calls,
            "final_products": self.final_products,
            "first_contact_intro": getattr(self, "first_contact_intro", ""),
            "first_contact_messages": getattr(self, "first_contact_messages", []),
        }
