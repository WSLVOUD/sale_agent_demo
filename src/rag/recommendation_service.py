"""Recommendation Service —— 统一推荐入口（v2.3 §6）。

从 v2.3 起，真正的决策链路在 :mod:`src.rag.recommendation_coordinator`：

    Recommendation Ready Gate → Engineering Derivation → Provenance Guard
        → Recommendation Engine → Validation → Final Recommendation + 审计日志

本服务只是"对外兼容层"：保持历史返回字段（`recommendation_status` /
`recommendation_basis` / `gate` …）不变，同时把协调器的结果透出来
（`provenance` / `decision_audit` / `coordinator_status`）。

任何路径（Sales / Solution / Fast / Normal / Agent / API）都必须走这里，
不允许自己实现另一套推荐算法。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.models.requirement import RequirementProfile
from src.rag.recommendation_coordinator import (
    CONFLICT,
    DEGRADED,
    NEED_CLARIFICATION,
    RECOMMENDED,
    REJECTED,
    RecommendationCoordinator,
)

logger = logging.getLogger(__name__)


class RecommendationService:
    """Gate → Engine 的统一推荐服务。"""

    def __init__(self, engine=None):
        self.coordinator = RecommendationCoordinator(engine=engine)
        self.engine = self.coordinator.engine

    def recommend(
        self,
        profile: RequirementProfile,
        top_k: int = 3,
        *,
        session_id: str = "",
        customer_input: str = "",
        final_response: str = "",
    ) -> Dict[str, Any]:
        """走统一出口：冲突/未就绪 → 追问；来源不合法 → 拒绝；否则出推荐。"""
        outcome = self.coordinator.recommend(
            profile,
            top_k=top_k,
            session_id=session_id,
            customer_input=customer_input,
            final_response=final_response,
        )

        if outcome.status in (NEED_CLARIFICATION, CONFLICT, REJECTED):
            return {
                "recommendation_status": "NEED_CLARIFICATION",
                "coordinator_status": outcome.status,
                "missing_fields": outcome.missing_fields,
                "next_question": outcome.next_question,
                "gate": outcome.gate,
                "recommendations": [],
                "candidate_count": 0,
                "rejected_count": 0,
                "violations": [],
                "reject_reasons": outcome.reject_reasons,
                "conflicts": outcome.conflicts,
                "provenance": outcome.provenance,
                "decision_audit": outcome.audit,
            }

        result = dict(outcome.result)
        # ── Phase 14：推荐依据（Best-effort 时标记 degraded）──────────────────
        # 客户"不知道"的字段不再阻塞推荐，但推荐结果必须能说清：
        #   - 依据了哪些已确认信息
        #   - 哪些字段是系统推断的
        #   - 哪些字段客户不知道（可能影响最终选型）
        degraded = outcome.status == DEGRADED
        unknown_slots = list((outcome.gate or {}).get("unknown_slots") or [])
        try:
            basis = profile.requirement_basis()
        except Exception:  # pragma: no cover - 防御式
            basis = {"confirmed": [], "inferred": [], "unknown": unknown_slots}
        for slot in unknown_slots:
            if slot not in basis.get("unknown", []):
                basis.setdefault("unknown", []).append(slot)

        result["recommendation_status"] = "DEGRADED" if degraded else "RECOMMENDED"
        result["missing_fields"] = []
        result["recommendation_basis"] = {
            "recommendation_status": result["recommendation_status"],
            "confirmed_requirements": list(basis.get("confirmed") or []),
            "inferred_requirements": list(basis.get("inferred") or []),
            "unknown_requirements": list(basis.get("unknown") or []),
        }
        result["unknown_requirements"] = list(basis.get("unknown") or [])
        result["gate"] = outcome.gate
        result["coordinator_status"] = outcome.status
        result["provenance"] = outcome.provenance
        result["decision_audit"] = outcome.audit
        result["reject_reasons"] = outcome.reject_reasons
        return result


__all__ = ["RecommendationService"]
