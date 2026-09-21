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
    check_feasibility,
    check_model_feasibility,
    check_provenance,
    conflict_message,
    detect_engineering_conflicts,
    finest_pitch_in,
)
from src.engineering import CONFLICT as FEASIBILITY_CONFLICT
from src.engineering import NEED_ADJUSTMENT as FEASIBILITY_NEED_ADJUSTMENT
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

        # ── v2.5 Phase 7 / v2.5+：Engineering Feasibility（不可绕过）─────
        #   尺寸 / P值 / 分辨率 / 箱体几何统一判断。客户口径（2026-09-21）：
        #   分辨率**只按"屏体大约能不能达到"处理** —— 不区分输入/屏体、不提澄清问题。
        #     · 数据自相矛盾 → 先澄清
        #     · 这个尺寸即使最细点间距也达不到 → 直接给结论话术：
        #         "需要更大尺寸（标准尺寸 X×Y）/ 需要更细的点间距 / 或者降低分辨率"
        #     · 换成更细的 P 值就能做到 → 不阻塞，继续走引擎（由引擎在更细档位选型）
        feasibility = check_feasibility(
            profile, finest_pitch_mm=self._finest_pitch_mm()
        )
        if not feasibility.feasible:
            status = CONFLICT if feasibility.status == FEASIBILITY_CONFLICT else REJECTED
            logger.info(
                "[Feasibility] status=%s conflicts=%d alternatives=%d",
                feasibility.status, len(feasibility.conflicts), len(feasibility.alternatives),
            )
            outcome = RecommendationOutcome(
                status=status,
                result={},
                gate={**gate_dict, "status": status, "ready": False},
                # 这里给客户的是"结论 + 调整方向"，不是提问
                next_question=feasibility.message
                or (conflict_message(profile) if status == CONFLICT else ""),
                missing_fields=[],
                reject_reasons=[
                    item.get("message", "") for item in feasibility.conflicts
                ] + list(feasibility.alternatives),
            )
            outcome.audit = self._audit(
                profile, technical, outcome, session_id, customer_input, final_response
            )
            outcome.audit["feasibility"] = feasibility.to_dict()
            return outcome
        if feasibility.status == FEASIBILITY_NEED_ADJUSTMENT:
            # 当前 P 值达不到，但目录里有更细的点间距能做到 → 交给引擎去选，
            # 不在这里打断对话（型号级可行性会在下一步逐款复核）
            logger.info("[Feasibility] %s", "；".join(feasibility.notes))

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

        # ── v2.5：型号级分辨率可行性（客户有 DISPLAY 级分辨率要求时）───────
        resolution_checks = []
        if feasibility.resolution_result is not None or (
            getattr(profile, "resolution_requirement", None) or {}
        ):
            from src.models.product import CanonicalModel  # noqa: F401  (类型提示)

            keep = []
            checks_by_model: Dict[str, Dict[str, Any]] = {}
            for item in result.get("recommendations") or []:
                model = self._model_by_name(str(item.get("model") or ""))
                if model is None:
                    keep.append(item)
                    continue
                check = check_model_feasibility(profile, model)
                resolution_checks.append(check)
                checks_by_model[str(getattr(model, "model", ""))] = check
                if not check.get("applicable") or check.get("acceptable"):
                    keep.append(item)
                else:
                    logger.info(
                        "[Feasibility] 型号 %s 分辨率不达标 → 剔除（%s）",
                        model.model, check.get("resolution_fit", {}).get("fit_level"),
                    )
            if keep:
                result["recommendations"] = keep
                # 客户口径（2026-09-21）："x × y" 不区分哪边是宽 —— 把可行性层挑出来的
                # 摆法透传给表达层，让箱体 / 尺寸数字跟可行性判断一致。
                top_check = checks_by_model.get(str(keep[0].get("model") or ""))
                if top_check and top_check.get("applicable"):
                    result["size_orientation"] = top_check.get("orientation") or "as_given"
                    result["size_layout"] = top_check.get("layout")
            else:
                outcome = RecommendationOutcome(
                    status=REJECTED,
                    result=result,
                    gate=gate_dict,
                    provenance=provenance,
                    # 逐款都不达标时，把可行性层给出的"要多大尺寸 / 多细的点间距 /
                    # 或者降低分辨率"直接讲给客户
                    next_question=feasibility.message
                    or self._resolution_shortfall_message(profile, feasibility),
                    reject_reasons=["没有型号能在当前约束下拼到足够接近的目标分辨率"]
                    + list(feasibility.alternatives),
                    missing_fields=["resolution"],
                )
                outcome.audit = self._audit(
                    profile, technical, outcome, session_id, customer_input, final_response
                )
                outcome.audit["feasibility"] = feasibility.to_dict()
                return outcome
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
        outcome.audit["feasibility"] = feasibility.to_dict()
        if resolution_checks:
            outcome.audit["resolution_checks"] = resolution_checks
        logger.info(
            "[RecommendationCoordinator] status=%s selected=%s provenance_ok=%s",
            outcome.status,
            outcome.audit.get("selected_model"),
            report.ok,
        )
        return outcome

    # ── 内部：按型号名取目录里的型号（用于型号级可行性）──────────────────
    def _model_by_name(self, name: str) -> Any:
        if not name:
            return None
        for model in getattr(self.engine, "models", []) or []:
            if str(getattr(model, "model", "")) == name:
                return model
        return None

    # ── 内部：目录里最细的点间距（可行性判断的物理极限）──────────────────
    def _finest_pitch_mm(self) -> Optional[float]:
        return finest_pitch_in(getattr(self.engine, "models", None))

    # ── 内部：型号逐款都不达标时给客户的兜底话术（结论，不是提问）────────
    def _resolution_shortfall_message(
        self, profile: RequirementProfile, feasibility: Any
    ) -> str:
        target = (feasibility.resolution_result.target if feasibility.resolution_result else None)
        if not target:
            requirement = getattr(profile, "resolution_requirement", None) or {}
            width_px = requirement.get("target_width")
            height_px = requirement.get("target_height")
            target = (width_px, height_px) if width_px and height_px else None
        if not target:
            return ""
        width_m = float(getattr(profile, "target_width_m", 0) or 0)
        height_m = float(getattr(profile, "target_height_m", 0) or 0)
        size_text = (
            f" at {width_m:.2f}m x {height_m:.2f}m" if width_m and height_m else ""
        )
        return (
            f"None of our models can reach {target[0]}x{target[1]} pixels{size_text} "
            "within the other requirements you gave. Going a bit larger on the screen, "
            "allowing a finer pitch, or aiming slightly lower on resolution would all work "
            "— tell me which one you'd prefer and I'll rebuild it for you."
        )

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
