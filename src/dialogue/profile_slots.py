"""需求档案 → 对话层槽位名（v2.7 §18 Answer Coverage 的输入）。

为什么单独成一个模块：v2.3.1 的边界不变量（
``tests/test_v231_orchestrator_boundary.py``）明确规定
**Orchestrator 只做编排、不持有业务参数名**（不得出现点间距 / 像素密度之类），
所以"档案字段 → 槽位名"这张业务映射表放在对话层，由它转出去。

实测 bug（2026-09-22）：原先 ``newly_filled_slots`` 来自 ``profile.to_facts()``，
而它**不包含** ``price_preference`` / ``content_type`` / ``budget_level`` →
客户答了"只在意质量"，档案里其实已经有值，但收口层看不到 →
「承接上限」逻辑误判成"客户一直没回答那个问题" → 每轮都把问题压下去，
结果既不问缺的 installation、也不推荐，对话卡死。

所以这里直接读档案的**全部字段**（pydantic 模型 / dict），再做槽位名映射。
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

# 档案字段名 → 对话层槽位名（一个槽位可能由多个字段构成，例如尺寸）
FIELD_TO_SLOT: Dict[str, str] = {
    "viewing_distance_m": "viewing_distance",
    "pixel_pitch_mm": "pixel_pitch",
    "target_width_m": "size",
    "target_height_m": "size",
    "screen_size_hint_mm": "size",
    "brightness_min_nit": "brightness",
}

# 留痕 / 元信息字段：不是"客户给过的需求值"
_META_FIELDS = frozenset({
    "sources",
    "conflicts",
    "conflict_slots",
    "ask_counts",
    "field_decisions",
})


def _raw_fields(profile: Any) -> Dict[str, Any]:
    """把档案摊平成字段字典（拿不到就返回空字典，不影响主流程）。"""
    if profile is None:
        return {}
    try:
        if hasattr(profile, "model_dump"):
            return dict(profile.model_dump())
        if isinstance(profile, Mapping):  # pragma: no cover - 防御式
            return dict(profile)
        return dict(
            profile.to_facts() if hasattr(profile, "to_facts") else {}
        )  # pragma: no cover - 防御式
    except Exception:  # pragma: no cover - 防御式
        return {}


def profile_slot_map(profile: Any) -> Dict[str, Any]:
    """档案里"已经有值"的槽位 → 槽位值。"""
    slots: Dict[str, Any] = {}
    for key, value in _raw_fields(profile).items():
        name = str(key)
        if name.startswith("_") or name.startswith("vision_"):
            continue
        if name in _META_FIELDS or value in (None, "", [], {}):
            continue
        slots[FIELD_TO_SLOT.get(name, name)] = value
    return slots


__all__ = ["FIELD_TO_SLOT", "profile_slot_map"]
