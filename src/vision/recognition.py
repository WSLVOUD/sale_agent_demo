"""图片识别结果（ImageRecognitionResult）—— 只输出**候选**事实，不触发推荐。

《LCD_IFP_需求链路工程化整改计划》Phase 3 / 第三章：

    图片 → 多模态模型 → ImageRecognitionResult → 客户确认 → ConfirmedRequirementProfile

硬性约束（本模块只做"第一跳"）：

    · 只描述"图片看到了什么/证据支持什么"，字段一律标 candidate；
    · **禁止**直接触发产品推荐 / Gate / 计算（不导入推荐或对话模块）；
    · 看见 ≠ 需要：``camera_observed`` 不等于 ``camera_required``，
      客户确认由 Image Confirmation Layer（`src/dialogue/image_confirmation.py`）负责。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 只有这些字段允许成为"图片候选事实"（计划 §三/§四/§五）
CANDIDATE_FIELDS: tuple[str, ...] = (
    "display_type",       # LED / LCD / IFP
    "installation",       # fixed / rental
    "environment",        # indoor / outdoor
    "is_splicing",
    "screen_size_inch",
    "splicing_layout",
    "screen_count",
    "bezel_mm",
    "camera_observed",
    "touch_observed",
    "handwriting_observed",
    "application",
)

# 客户可确认/可纠正的字段（确认层按这个顺序问，最多问一项）
CONFIRMABLE_FIELDS: tuple[str, ...] = (
    "environment",
    "is_splicing",
    "installation",
    "camera_observed",
    "screen_size_inch",
    "touch_observed",
    "handwriting_observed",
    "application",
)


@dataclass
class ImageRecognitionResult:
    """一张（或多张合并后的）图片的结构化候选结果。"""

    display_type: str = "UNKNOWN"
    installation: str = "unknown"
    environment: str = "unknown"
    is_splicing: Optional[bool] = None
    screen_size_inch: Optional[float] = None
    splicing_layout: str = ""
    screen_count: Optional[int] = None
    bezel_mm: Optional[float] = None
    camera_observed: Optional[bool] = None
    touch_observed: Optional[bool] = None
    handwriting_observed: Optional[bool] = None
    application: str = ""
    confidence: float = 0.0
    evidence: Dict[str, str] = field(default_factory=dict)
    notes: str = ""
    model: str = ""

    # ── 构造 ────────────────────────────────────────────────────────────
    @classmethod
    def from_payload(cls, payload: Any) -> "ImageRecognitionResult":
        """从多模态模型 / VisionRequirement 的字典形态构造（缺字段一律留空）。"""
        data = _as_dict(payload)
        result = cls(
            display_type=str(data.get("display_type") or "UNKNOWN").upper(),
            installation=str(data.get("installation") or "unknown").lower(),
            environment=str(data.get("environment") or "unknown").lower(),
            splicing_layout=str(data.get("splicing_layout") or ""),
            application=str(data.get("application") or ""),
            notes=str(data.get("notes") or ""),
            model=str(data.get("model") or ""),
            confidence=float(data.get("confidence") or 0.0),
        )
        for name in (
            "is_splicing",
            "camera_observed",
            "touch_observed",
            "handwriting_observed",
        ):
            value = _bool_or_none(data.get(name))
            if value is not None:
                setattr(result, name, value)
        for name in ("screen_size_inch", "bezel_mm"):
            value = _float_or_none(data.get(name))
            if value is not None:
                setattr(result, name, value)
        count = _int_or_none(data.get("screen_count"))
        if count is not None:
            result.screen_count = count
        evidence = data.get("evidence")
        if isinstance(evidence, dict):
            result.evidence = {str(k): str(v) for k, v in evidence.items()}
        if not result.screen_count and result.splicing_layout:
            result.screen_count = _count_from_layout(result.splicing_layout)
        return result

    # ── 输出 ────────────────────────────────────────────────────────────
    def candidates(self) -> Dict[str, Any]:
        """图片"候选事实"（只含有值/有证据的字段）。"""
        out: Dict[str, Any] = {}
        for name in CANDIDATE_FIELDS:
            value = getattr(self, name, None)
            if value in (None, "", [], {}, "unknown", "UNKNOWN"):
                continue
            out[name] = value
        return out

    def has_evidence(self, name: str) -> bool:
        return bool(str(self.evidence.get(name) or "").strip())

    def is_empty(self) -> bool:
        return not self.candidates()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "display_type": self.display_type,
            "installation": self.installation,
            "environment": self.environment,
            "is_splicing": self.is_splicing,
            "screen_size_inch": self.screen_size_inch,
            "splicing_layout": self.splicing_layout,
            "screen_count": self.screen_count,
            "bezel_mm": self.bezel_mm,
            "camera_observed": self.camera_observed,
            "touch_observed": self.touch_observed,
            "handwriting_observed": self.handwriting_observed,
            "application": self.application,
            "confidence": self.confidence,
            "evidence": dict(self.evidence),
            "notes": self.notes,
            "model": self.model,
        }


# ── 内部工具 ────────────────────────────────────────────────────────────────
def _as_dict(payload: Any) -> Dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return dict(payload)
    if hasattr(payload, "model_dump"):
        return dict(payload.model_dump())
    if hasattr(payload, "to_dict"):
        return dict(payload.to_dict())
    return {}


def _unwrap(value: Any) -> Any:
    """VisionField / {"value": …} → 值本身。"""
    if isinstance(value, dict) and "value" in value:
        return value.get("value")
    return getattr(value, "value", value)


def _bool_or_none(value: Any) -> Optional[bool]:
    value = _unwrap(value)
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "yes", "y", "1", "是", "有"):
        return True
    if text in ("false", "no", "n", "0", "否", "没有", "无"):
        return False
    return None


def _float_or_none(value: Any) -> Optional[float]:
    value = _unwrap(value)
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> Optional[int]:
    number = _float_or_none(value)
    return int(number) if number is not None else None


def _count_from_layout(layout: str) -> Optional[int]:
    """\"6x2\" → 12（图片只看到排布时，屏数按排布算）。"""
    import re

    match = re.match(r"^\s*(\d+)\s*[x×*]\s*(\d+)\s*$", str(layout or ""))
    if not match:
        return None
    return int(match.group(1)) * int(match.group(2))


__all__ = ["CANDIDATE_FIELDS", "CONFIRMABLE_FIELDS", "ImageRecognitionResult"]
