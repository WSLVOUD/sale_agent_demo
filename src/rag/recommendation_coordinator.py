"""v2.3 §6：推荐的**唯一出口**。

所有路径（Fast / Normal / Agent / Sales / Solution / API）最终都从这里拿推荐：

    RequirementProfile
        ↓
    Conflict 检查（§9：冲突未解决不推荐）
        ↓
    Recommendation Ready Gate
        ↓
    Engineering Derivation（统一在 src/engineering/）
        ↓
    Provenance Guard（§5：关键参数没有合法来源 → 不推荐）
        ↓
    RecommendationEngine
        ↓
    Validation（一致性校验）
        ↓
    Final Recommendation + 决策审计日志（§16）

任何模块都不允许自己再实现一套推荐算法。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.engineering import (
    check_provenance,
    conflict_message,
    detect_engineering_conflicts,
)
from src.models.requirement import RequirementProfile
from src.observability.decision_log import record_from_recommendation
from src.rag.parameter_inference import infer_technical_parameters
from src.rag.readiness import check_recommendation_ready

logger = logging.getLogger(__name__)

RECOMMENDED = "RECOMMENDED"
DEGRADED = "DEGRADED"
NEED_CLARIFICATION = "NEED_CLARIFICATION"
CONFLICT = "CONFLICT"
REJECTED = "REJECTED"


@dataclass
class RecommendationOutcome:
    """统一出口的返回结构（RecommendationService 再包一层兼容旧字段）。"""

    status: str = NEED_CLARIFICATION
    result: Dict[str, Any] = field(default_factory=dict)
    gate: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    audit: Dict[str, Any] = field(default_factory=dict)
    next_question: Optional[str] = None
    missing_fields: List[str] = field(default_factory=list)
    reject_reasons: List[str] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def recommendations(self) -> List[Dict[str, Any]]:
        return list(self.result.get("recommendations") or [])


class RecommendationCoordinator:
    """Gate + 推导 + 来源守卫 + 引擎 + 审计 的唯一入口。"""

    def __init__(self, engine: Any = None):
        if engine is None:
            from src.rag.recommendation_engine import RecommendationEngine

            engine = RecommendationEngine()
        self.engine = engine

    # ── 主入口 ──────────────────────────────────────────────────────────
    def recommend(
        self,
        profile: RequirementProfile,
        top_k: int = 3,
        *,
        session_id: str = "",
        customer_input: str = "",
        final_response: str = "",
    ) -> RecommendationOutcome:
        technical = infer_technical_parameters(profile.to_facts()) if profile else {}
        gate = check_recommendation_ready(profile) if profile else None
        gate_dict = gate.to_dict() if gate is not None else {}

        # ── §9：冲突未解决 → 不推荐，先澄清 ──────────────────────────────
        conflicts = [item.to_dict() for item in detect_engineering_conflicts(profile)]
        stored_conflicts = list(getattr(profile, "conflicts", None) or [])
        if conflicts or stored_conflicts:
            outcome = RecommendationOutcome(
                status=CONFLICT,
                result={},
                gate={**gate_dict, "status": CONFLICT, "ready": False},
                next_question=conflict_message(profile),
                missing_fields=list(getattr(profile, "conflict_slots", None) or []),
                reject_reasons=[item["message"] for item in conflicts] + stored_conflicts,
                conflicts=conflicts,
            )
            outcome.audit = self._audit(
                profile, technical, outcome, session_id, customer_input, final_response
            )
            return outcome

        # ── Gate：需求没齐 → 继续追问 ────────────────────────────────────
        if gate is None or not gate.ready:
            outcome = RecommendationOutcome(
                status=NEED_CLARIFICATION,
                result={},
                gate=gate_dict,
                next_question=(gate.next_question if gate is not None else None),
                missing_fields=list(gate.missing) if gate is not None else [],
                reject_reasons=[gate.reason] if gate is not None and gate.reason else [],
            )
            outcome.audit = self._audit(
                profile, technical, outcome, session_id, customer_input, final_response
            )
            return outcome

        # ── §5：Provenance Guard ─────────────────────────────────────────
        # 客户直接点名型号 / 系列时，"来源"就是客户自己那句话（型号自带参数），
        # 不需要再要求工程参数的来源；其余情况必须能说清点间距与室内外的依据。
        named_model = bool(
            getattr(profile, "model", None) or getattr(profile, "series_id", None)
        )
        required = () if named_model else ("pixel_pitch_mm", "environment")
        report = check_provenance(profile, technical, required=required)
        provenance = report.to_dict()
        if not report.ok:
            logger.warning(
                "RecommendationCoordinator: provenance guard rejected: %s",
                report.reject_reasons,
            )
            outcome = RecommendationOutcome(
                status=REJECTED,
                result={},
                gate=gate_dict,
                provenance=provenance,
                reject_reasons=list(report.reject_reasons),
                missing_fields=list(report.missing) + list(report.illegal),
            )
            outcome.audit = self._audit(
                profile, technical, outcome, session_id, customer_input, final_response
            )
            return outcome

        # ── 确定性引擎 ───────────────────────────────────────────────────
        result = self.engine.recommend(profile=profile, top_k=top_k, require_ready=True)
        violations = list(result.get("violations") or [])
        status = (
            NEED_CLARIFICATION
            if result.get("recommendation_status") == NEED_CLARIFICATION
            else DEGRADED
            if gate.status == "DEGRADED_READY"
            else RECOMMENDED
        )
        if violations:
            status = REJECTED
        outcome = RecommendationOutcome(
            status=status,
            result=result,
            gate=gate_dict,
            provenance=provenance,
            next_question=result.get("next_question"),
            missing_fields=list(result.get("missing_fields") or []),
            reject_reasons=[str(item) for item in violations],
        )
        outcome.audit = self._audit(
            profile, technical, outcome, session_id, customer_input, final_response
        )
        logger.info(
            "[RecommendationCoordinator] status=%s selected=%s provenance_ok=%s",
            outcome.status,
            outcome.audit.get("selected_model"),
            report.ok,
        )
        return outcome

    # ── 审计 ────────────────────────────────────────────────────────────
    def _audit(
        self,
        profile: Any,
        technical: Dict[str, Any],
        outcome: RecommendationOutcome,
        session_id: str,
        customer_input: str,
        final_response: str,
    ) -> Dict[str, Any]:
        try:
            record = record_from_recommendation(
                profile,
                session_id=session_id,
                customer_input=customer_input,
                technical=technical,
                result=outcome.result,
                provenance=outcome.provenance,
                gate=outcome.gate,
                validation={"status": outcome.status, "reject_reasons": outcome.reject_reasons},
                final_response=final_response,
            )
            return record.emit()
        except Exception as exc:  # pragma: no cover - 审计失败不能影响业务
            logger.warning("Decision audit failed: %s", exc)
            return {}


def recommend(profile: RequirementProfile, top_k: int = 3, **kwargs: Any) -> RecommendationOutcome:
    """便捷函数：一次性走完统一出口。"""
    return RecommendationCoordinator().recommend(profile, top_k=top_k, **kwargs)


__all__ = [
    "CONFLICT",
    "DEGRADED",
    "NEED_CLARIFICATION",
    "RECOMMENDED",
    "REJECTED",
    "RecommendationCoordinator",
    "RecommendationOutcome",
    "recommend",
]
