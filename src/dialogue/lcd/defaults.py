"""LCD 可选字段的业务默认值（纯 helper）。

只提供"补什么值"，**不决定"什么时候补"** —— 那个判断在 lcd_decision 里。
搬迁自 lcd_decision.py（整改计划 §十一 第 6 条）。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable

logger = logging.getLogger(__name__)


_FILLABLE_DEFAULTS: Dict[str, Any] = {
    "lcd_bezel": 3.5,
    "lcd_touch": False,
    "lcd_handwriting": False,
    "lcd_ops": False,
    "lcd_camera": False,
    "lcd_tender": False,
}
_FILLABLE_FIELD: Dict[str, str] = {
    "lcd_bezel": "lcd_bezel_mm",
    "lcd_touch": "lcd_touch_required",
    "lcd_handwriting": "lcd_handwriting_required",
    "lcd_ops": "lcd_ops_required",
    "lcd_camera": "lcd_camera_required",
    "lcd_tender": "lcd_tender_project",
}

# ── 防死循环（客户口径 2026-09-30："还是没有触发推荐"）──────────────────────
# 实测：视频墙对话里客户答的排布是 "3x3"，旧解析没记下来 → 需求链在"问排布"
# 上无限循环，永远不推荐。除了修解析，这里再加一道**收口**规则：
#
#   同一个槽位问满 _SLOT_ASK_LIMIT 次还拿不到答案 → 绝不再问第三次。
#   · 有安全业务默认值的（LCD 几乎全是室内屏）→ 按默认补齐（来源标 recommended）
#   · 没有安全默认值的（排布）→ 不阻塞推荐，也不编造：直接带着"待确认排布"推荐
#
# LED 链路不经过这个函数（计划 §二十九 LED Chain = Frozen），所以 LED 口径不受影响。
_SLOT_ASK_LIMIT = 2
_DEGRADED_DEFAULTS: Dict[str, Any] = {
    "environment": "indoor",
}
_DEGRADED_FIELD: Dict[str, str] = {
    "environment": "environment",
}
_DEGRADABLE_SLOTS: frozenset[str] = frozenset({"lcd_layout"})

def _apply_degraded_default(profile: Any, slot: str) -> bool:
    """问满上限仍无答案 → 按业务默认补齐（来源标 recommended，不伪装成客户要求）。"""
    if slot not in _DEGRADED_DEFAULTS:
        return False
    field_name = _DEGRADED_FIELD.get(slot, slot)
    if getattr(profile, field_name, None) not in (None, "", [], {}):
        return False
    value = _DEGRADED_DEFAULTS[slot]
    setattr(profile, field_name, value)
    sources = dict(getattr(profile, "sources", None) or {})
    sources[field_name] = "recommended"
    profile.sources = sources
    logger.info(
        "[LCD] slot=%s 问了 %d 次仍未答 → 按业务默认收口：%s=%r",
        slot, _SLOT_ASK_LIMIT, field_name, value,
    )
    return True

def fill_defaults_for_recommendation(profile: Any, missing: Iterable[str]) -> Dict[str, Any]:
    """客户要推荐 / 可选字段已经问过 → 用业务默认值补齐（来源标 recommended）。"""
    applied: Dict[str, Any] = {}
    if profile is None:
        return applied
    sources = dict(getattr(profile, "sources", None) or {})
    field_map = {
        "lcd_bezel": "lcd_bezel_mm",
        "lcd_touch": "lcd_touch_required",
        "lcd_handwriting": "lcd_handwriting_required",
        "lcd_ops": "lcd_ops_required",
        "lcd_camera": "lcd_camera_required",
        "lcd_tender": "lcd_tender_project",
    }
    for slot in list(missing or []):
        if slot not in _FILLABLE_DEFAULTS:
            continue
        field_name = field_map[slot]
        if getattr(profile, field_name, None) not in (None, "", [], {}):
            continue
        setattr(profile, field_name, _FILLABLE_DEFAULTS[slot])
        sources[field_name] = "recommended"
        applied[field_name] = _FILLABLE_DEFAULTS[slot]
    profile.sources = sources
    return applied


__all__ = [
    "fill_defaults_for_recommendation",
]
