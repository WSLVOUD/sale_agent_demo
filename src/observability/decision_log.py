"""v2.3 §16：决策审计日志。

每次推荐记一条结构化记录，回答"为什么是这个型号"：

    Turn ID / Customer Input / Extracted Facts / Field Decisions / Derived Parameters
    / Constraints / Gate Decision / Candidate Models / Rejected Models + Reasons
    / Selected Model / Validation Result / Final Response

记录只写日志（JSON 一行），不影响业务流程；拿不到某个字段就留空。
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _safe(value: Any) -> Any:
    """把任意对象转成可 JSON 序列化的形式（只保留可读的关键信息）。"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    for attr in ("model_dump", "to_dict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                return _safe(method())
            except Exception:  # pragma: no cover - 防御式
                break
    return str(value)[:300]


@dataclass
class DecisionRecord:
    """一次推荐/追问的完整决策记录。"""

    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: str = ""
    customer_input: str = ""
    extracted_facts: Dict[str, Any] = field(default_factory=dict)
    field_decisions: Dict[str, str] = field(default_factory=dict)
    derived_parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    gate_decision: Dict[str, Any] = field(default_factory=dict)
    candidate_models: List[str] = field(default_factory=list)
    rejected_models: List[Dict[str, Any]] = field(default_factory=list)
    selected_model: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)
    validation_result: Dict[str, Any] = field(default_factory=dict)
    final_response: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "customer_input": self.customer_input[:400],
            "extracted_facts": _safe(self.extracted_facts),
            "field_decisions": _safe(self.field_decisions),
            "derived_parameters": _safe(self.derived_parameters),
            "constraints": _safe(self.constraints),
            "gate_decision": _safe(self.gate_decision),
            "candidate_models": list(self.candidate_models),
            "rejected_models": _safe(self.rejected_models)[:20],
            "selected_model": self.selected_model,
            "provenance": _safe(self.provenance),
            "validation_result": _safe(self.validation_result),
            "final_response": self.final_response[:400],
            "created_at": self.created_at,
        }

    def emit(self) -> Dict[str, Any]:
        payload = self.to_dict()
        logger.info("[DecisionAudit] %s", json.dumps(payload, ensure_ascii=False))
        return payload


def record_from_recommendation(
    profile: Any,
    *,
    session_id: str = "",
    customer_input: str = "",
    technical: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    gate: Optional[Dict[str, Any]] = None,
    validation: Optional[Dict[str, Any]] = None,
    final_response: str = "",
) -> DecisionRecord:
    """用一次调用的输入/输出拼出审计记录。"""
    result = dict(result or {})
    technical = dict(technical or result.get("technical_parameters") or {})
    recommendations = list(result.get("recommendations") or [])
    record = DecisionRecord(
        session_id=str(session_id or ""),
        customer_input=str(customer_input or ""),
        extracted_facts=_safe(profile.to_facts()) if hasattr(profile, "to_facts") else {},
        field_decisions=dict(getattr(profile, "field_decisions", {}) or {}),
        derived_parameters={
            "viewing_distance_m": technical.get("viewing_distance_m"),
            "viewing_distance_estimate": technical.get("viewing_distance_estimate"),
            "pitch_window": [
                technical.get("pixel_pitch_min_mm"),
                technical.get("pixel_pitch_max_mm"),
            ],
            "pitch_target_mm": technical.get("pitch_target_mm"),
            "backbone_source": technical.get("source"),
        },
        constraints=_safe(result.get("hard_constraints") or {}),
        gate_decision=_safe(gate or result.get("gate") or {}),
        candidate_models=[str(item.get("model")) for item in recommendations],
        rejected_models=list(result.get("rejected") or result.get("rejected_models") or []),
        selected_model=str(recommendations[0].get("model")) if recommendations else "",
        provenance=_safe(provenance or result.get("provenance") or {}),
        validation_result=_safe(validation or {
            "violations": result.get("violations") or [],
            "status": result.get("recommendation_status"),
        }),
        final_response=str(final_response or ""),
    )
    return record


__all__ = ["DecisionRecord", "record_from_recommendation"]
