"""v2.3 §3.2 / §5：字段级 provenance（来源 + 公式）与推荐前的来源守卫。

规则（计划 §5）：

  任何最终推荐用到的关键工程参数都必须有来源：

      customer  客户明确说的
      confirmed 客户确认过的（含已核对通过的图片结论）
      derived   Python 用公式推出来的（必须带 formula_id）
      inferred 系统估算（可用于打分，但不能伪装成客户事实）
      vision   图片识别结果（必须经过确认，见 §11）

  没有合法 provenance → **不推荐**（Recommendation Reject）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 允许的来源
CUSTOMER = "customer"
CONFIRMED = "confirmed"
DERIVED = "derived"
INFERRED = "inferred"
VISION = "vision"
LEGAL_SOURCES: Tuple[str, ...] = (CUSTOMER, CONFIRMED, DERIVED, INFERRED, VISION)

# 推导公式编号（出现的每个公式都要有编号，便于审计）
FORMULA_VIEWING_DISTANCE = "GEOMETRY_TO_VIEWING_DISTANCE_V1"
FORMULA_PITCH_WINDOW = "DISTANCE_TO_PITCH_WINDOW_V1"
FORMULA_PITCH_TARGET = "ENVIRONMENT_DISTANCE_TO_PITCH_TARGET_V1"
FORMULA_PITCH_FALLBACK = "ENVIRONMENT_FALLBACK_PITCH_BAND_V1"

# 档案 sources 里的来源 → 合法来源 + 字段状态
_SOURCE_MAP: Dict[str, Tuple[str, str]] = {
    "explicit": (CUSTOMER, "CONFIRMED"),
    "confirmed": (CONFIRMED, "CONFIRMED"),
    "vision_accepted": (CONFIRMED, "CONFIRMED"),
    "scenario_derived": (DERIVED, "INFERRED"),
    "vision_explicit": (VISION, "INFERRED"),
    "vision_inferred": (VISION, "INFERRED"),
    "inferred": (INFERRED, "INFERRED"),
    "default": (INFERRED, "INFERRED"),
}

# 推荐必须能说清来源的关键参数
KEY_PARAMETERS: Tuple[str, ...] = (
    "environment",
    "installation",
    "pixel_pitch_mm",
    "viewing_distance_m",
    "screen_size",
)


@dataclass(frozen=True)
class FieldValue:
    """带来源的字段值（计划 §3.2 的统一字段结构）。"""

    value: Any = None
    unit: str = ""
    source: str = ""
    status: str = "MISSING"
    confidence: float = 0.0
    source_fields: Tuple[str, ...] = ()
    formula_id: str = ""

    @property
    def has_provenance(self) -> bool:
        return self.source in LEGAL_SOURCES

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "status": self.status,
            "confidence": round(float(self.confidence), 2),
            "source_fields": list(self.source_fields),
            "formula_id": self.formula_id,
        }


@dataclass
class ProvenanceReport:
    """推荐前的来源检查结果。"""

    ok: bool = True
    entries: Dict[str, FieldValue] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)          # 完全没有来源的参数
    illegal: List[str] = field(default_factory=list)          # 来源不在允许清单里
    reject_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "missing": list(self.missing),
            "illegal": list(self.illegal),
            "reject_reasons": list(self.reject_reasons),
            "entries": {key: value.to_dict() for key, value in self.entries.items()},
        }


def slot_provenance(profile: Any, slot: str) -> FieldValue:
    """从 RequirementProfile 的 sources / 字段状态推出一个 FieldValue。"""
    if profile is None:
        return FieldValue()
    from src.models.requirement import canonical_slot

    key = canonical_slot(slot)
    source_tag = ""
    try:
        source_tag = str(profile.slot_source(key) or "")
    except Exception:  # pragma: no cover - 防御式
        source_tag = str((getattr(profile, "sources", {}) or {}).get(key) or "")
    if not source_tag:
        try:
            source_tag = str((getattr(profile, "sources", {}) or {}).get(key) or "")
        except Exception:  # pragma: no cover
            source_tag = ""

    mapped = _SOURCE_MAP.get(source_tag)
    status = "MISSING"
    try:
        status = str(profile.field_decision(key) or "MISSING")
    except Exception:  # pragma: no cover - 防御式
        pass

    if mapped:
        source, mapped_status = mapped
        # provenance 的 source / status **只由来源标记决定**：
        #   explicit / confirmed / vision_accepted → customer / confirmed（CONFIRMED）
        #   scenario_derived                       → derived（INFERRED，不能算客户确认）
        #   vision_*                               → vision（INFERRED）
        #   inferred / default                     → inferred（INFERRED）
        # 字段决策状态（field_decision）仍然按原规则用于 Gate 判断，但两者不能混为一谈。
        status = mapped_status
    else:
        # 没有来源标记 → 没有 provenance（不允许当推荐依据）。
        # 注意：只有真正由工程公式推导出来的值才允许标 derived，
        # 那是由 build_provenance() 显式构造的，不在这里"猜"。
        source = ""

    value = None
    for attr in ("pixel_pitch_mm", "viewing_distance_m", "target_width_m",
                 "target_height_m", "environment", "installation"):
        if canonical_slot(attr) == key or attr == key:
            value = getattr(profile, attr, None)
            if value is not None:
                break
    if value is None:
        value = getattr(profile, key, None)

    unit = {
        "pixel_pitch_mm": "mm",
        "viewing_distance_m": "m",
        "target_width_m": "m",
        "target_height_m": "m",
    }.get(key, "")
    confidence = 1.0 if source in (CUSTOMER, CONFIRMED) else 0.7 if source == VISION else 0.6
    return FieldValue(
        value=value,
        unit=unit,
        source=source,
        status=status,
        confidence=confidence if value is not None else 0.0,
    )


def build_provenance(profile: Any, technical: Optional[Dict[str, Any]] = None) -> Dict[str, FieldValue]:
    """关键参数的 provenance 表（供推荐/审计使用）。"""
    technical = dict(technical or {})
    entries: Dict[str, FieldValue] = {}

    for slot in ("environment", "installation"):
        entries[slot] = slot_provenance(profile, slot)

    pitch = slot_provenance(profile, "pixel_pitch")
    if pitch.value is None and technical.get("pixel_pitch_min_mm") is not None:
        band_source = str((technical.get("source") or {}).get("pixel_pitch") or "")
        formula = {
            "fallback_environment_default": FORMULA_PITCH_FALLBACK,
        }.get(band_source, FORMULA_PITCH_WINDOW)
        pitch = FieldValue(
            value=technical.get("pitch_target_mm") or technical.get("pixel_pitch_min_mm"),
            unit="mm",
            source=DERIVED,
            status="INFERRED",
            confidence=0.7,
            source_fields=tuple(
                name for name in ("viewing_distance_m", "audience_count",
                                  "room_area_sqm", "room_depth_m",
                                  "target_width_m", "target_height_m")
                if (profile is not None and getattr(profile, name, None) is not None)
            ),
            formula_id=formula,
        )
    entries["pixel_pitch_mm"] = pitch

    distance = slot_provenance(profile, "viewing_distance")
    if distance.value is None and technical.get("viewing_distance_m") is not None:
        estimate = technical.get("viewing_distance_estimate") or {}
        distance = FieldValue(
            value=technical.get("viewing_distance_m"),
            unit="m",
            source=DERIVED,
            status="INFERRED",
            confidence=0.65,
            source_fields=(str(estimate.get("source") or ""),) if estimate else (),
            formula_id=FORMULA_VIEWING_DISTANCE,
        )
    entries["viewing_distance_m"] = distance

    if technical.get("pitch_target_mm") is not None and entries["pixel_pitch_mm"].source:
        entries["pitch_target_mm"] = FieldValue(
            value=technical.get("pitch_target_mm"),
            unit="mm",
            source=entries["pixel_pitch_mm"].source,
            status="INFERRED",
            confidence=entries["pixel_pitch_mm"].confidence,
            source_fields=entries["pixel_pitch_mm"].source_fields,
            formula_id=FORMULA_PITCH_TARGET,
        )

    width = getattr(profile, "target_width_m", None) if profile is not None else None
    height = getattr(profile, "target_height_m", None) if profile is not None else None
    if width and height:
        entries["screen_size"] = FieldValue(
            value=[width, height],
            unit="m",
            source=slot_provenance(profile, "width").source or slot_provenance(profile, "size").source,
            status=slot_provenance(profile, "size").status,
            confidence=1.0 if slot_provenance(profile, "size").source in (CUSTOMER, CONFIRMED) else 0.7,
        )
    return entries


def check_provenance(
    profile: Any,
    technical: Optional[Dict[str, Any]] = None,
    required: Sequence[str] = ("pixel_pitch_mm", "environment"),
) -> ProvenanceReport:
    """推荐前的来源守卫：关键参数没有合法来源 → 不允许推荐。"""
    entries = build_provenance(profile, technical)
    report = ProvenanceReport(entries=entries)
    for name in required:
        entry = entries.get(name)
        if entry is None or entry.value in (None, "", [], {}):
            report.missing.append(name)
            report.reject_reasons.append(f"缺少 {name} 的来源（没有任何依据）")
            continue
        if entry.source not in LEGAL_SOURCES:
            report.illegal.append(name)
            report.reject_reasons.append(
                f"{name} 的来源不合法（{entry.source or '空'}）：不允许 LLM 直接生成关键工程参数"
            )
    report.ok = not report.missing and not report.illegal
    return report


__all__ = [
    "CONFIRMED",
    "CUSTOMER",
    "DERIVED",
    "FieldValue",
    "FORMULA_PITCH_FALLBACK",
    "FORMULA_PITCH_TARGET",
    "FORMULA_PITCH_WINDOW",
    "FORMULA_VIEWING_DISTANCE",
    "INFERRED",
    "KEY_PARAMETERS",
    "LEGAL_SOURCES",
    "ProvenanceReport",
    "VISION",
    "build_provenance",
    "check_provenance",
    "slot_provenance",
]
