"""v2.3 §14：统一 Validation —— LLM / Vision / 规则输出的**入档闸门**。

    Understanding / Vision / Rewrite
                ↓
            Validation          ← 本模块
                ↓
        RequirementProfile

校验内容（计划 §14）：字段是否合法、单位 / 数值范围是否合法、source 是否存在、
是否与已有事实冲突、是否允许更新。禁止 LLM 的 JSON 直接改核心状态。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.engineering.provenance import LEGAL_SOURCES, _SOURCE_MAP

logger = logging.getLogger(__name__)

# 已知槽位（其余字段一律拒绝入档）
KNOWN_SLOTS: Tuple[str, ...] = (
    "display_type", "environment", "purpose", "content_type", "installation",
    "price_preference", "budget_level", "viewing_distance_m", "distance",
    "target_width_mm", "target_height_mm", "screen_size_hint_mm", "size_axis",
    "pixel_pitch_mm", "pixel_pitch", "brightness_min", "brightness_min_nit",
    "brightness_max", "brightness_max_nit", "audience_count", "room_area_sqm",
    "room_depth_m", "series_id", "model", "interaction", "waterproof", "cob",
    "hdr", "gob", "flexible", "size", "target_size", "usage", "location_type",
    "indoor", "outdoor", "semi_outdoor", "is_rental", "brightness", "resolution",
)

# 允许的标记 / 派生键（抽取链路内部使用，不参与数值校验）
MARKER_KEYS: Tuple[str, ...] = (
    "_explicit_keys", "_inferred_slots", "_default_slots", "_scenario_derived",
    "_semantic_conflicts", "_raw_message",
)

# 数值范围：字段 -> (最小值, 最大值, 单位)
RANGES: Dict[str, Tuple[float, float, str]] = {
    "viewing_distance_m": (0.3, 200, "m"),
    "distance": (0.3, 200, "m"),
    "target_width_mm": (100, 200000, "mm"),
    "target_height_mm": (100, 200000, "mm"),
    "screen_size_hint_mm": (10, 200000, "mm"),
    "pixel_pitch_mm": (0.3, 20, "mm"),
    "pixel_pitch": (0.3, 20, "mm"),
    "brightness_min": (1, 20000, "nit"),
    "brightness_max": (1, 20000, "nit"),
    "brightness_min_nit": (1, 20000, "nit"),
    "brightness_max_nit": (1, 20000, "nit"),
    "audience_count": (1, 100000, "people"),
    "room_area_sqm": (1, 100000, "sqm"),
    "room_depth_m": (1, 200, "m"),
}

ENUMS: Dict[str, Tuple[str, ...]] = {
    "environment": ("indoor", "outdoor", "semi_outdoor"),
    "installation": ("fixed", "rental"),
    "display_type": ("LED", "LCD", "IFP", "BOTH"),
    "content_type": ("video", "image", "mixed"),
    "budget_level": ("low", "mid", "medium", "high"),
    "price_preference": ("price", "quality", "both"),
    "size_axis": ("width", "height", "diagonal"),
}


@dataclass
class FactValidation:
    """一次入档校验的结果。"""

    ok: bool = True
    accepted: Dict[str, Any] = field(default_factory=dict)
    rejected: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "accepted": {key: str(value)[:80] for key, value in self.accepted.items()},
            "rejected": dict(self.rejected),
            "notes": list(self.notes),
        }


def _range_error(slot: str, value: Any) -> Optional[str]:
    spec = RANGES.get(slot)
    if not spec:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"{slot} 不是数字：{value!r}"
    low, high, unit = spec
    if not (low <= number <= high):
        return f"{slot} 超出合理范围（{low}~{high} {unit}）：{number}"
    return None


def validate_incoming_facts(
    facts: Dict[str, Any],
    *,
    source: str = "rule",
    profile: Any = None,
) -> FactValidation:
    """校验一批准备入档的事实。

    ``source``：rule / llm / vision / customer —— 用来判断"是否允许更新"。
    """
    result = FactValidation()
    accepted: Dict[str, Any] = {}
    mapped = _SOURCE_MAP.get(str(source or "").lower())
    source_tag = mapped[0] if mapped else ""

    for key, value in dict(facts or {}).items():
        name = str(key)
        if name in MARKER_KEYS or name.startswith("_"):
            accepted[name] = value
            continue
        if value in (None, "", [], {}):
            continue
        if name not in KNOWN_SLOTS:
            result.rejected[name] = "未知字段（不允许入档）"
            result.notes.append(f"dropped unknown field {name!r}")
            continue
        if name in ENUMS and str(value).lower() not in [v.lower() for v in ENUMS[name]]:
            result.rejected[name] = f"{name} 取值不合法：{value!r}（允许 {ENUMS[name]}）"
            continue
        range_error = _range_error(name, value)
        if range_error:
            result.rejected[name] = range_error
            continue
        if profile is not None and source in ("llm", "vision"):
            try:
                from src.models.requirement import canonical_slot

                slot = canonical_slot(name)
                if profile.slot_is_confirmed(slot):
                    current = getattr(profile, slot, None)
                    if current not in (None, "", [], {}) and str(current) != str(value):
                        result.rejected[name] = (
                            f"客户已确认 {slot}={current!r}，{source} 来源不能覆盖"
                        )
                        result.notes.append(f"kept customer value for {slot}")
                        continue
            except Exception:  # pragma: no cover - 防御式
                pass
        accepted[name] = value

    result.accepted = accepted
    result.ok = not result.rejected
    if result.rejected:
        logger.warning(
            "[FactValidation] source=%s rejected=%s", source or "-", result.rejected
        )
    if source_tag and source_tag not in LEGAL_SOURCES:  # pragma: no cover - 防御式
        result.notes.append(f"source {source!r} 不在允许清单里")
    return result


def validate_and_filter(
    facts: Dict[str, Any], *, source: str = "rule", profile: Any = None
) -> Dict[str, Any]:
    """便捷函数：返回校验后的字段（非法项直接丢掉）。"""
    return validate_incoming_facts(facts, source=source, profile=profile).accepted


__all__ = [
    "ENUMS",
    "FactValidation",
    "KNOWN_SLOTS",
    "MARKER_KEYS",
    "RANGES",
    "validate_and_filter",
    "validate_incoming_facts",
]
