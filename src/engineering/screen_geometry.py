"""v2.3 §3.2 / §12：屏体几何的统一计算（宽高、参考尺寸）。

只放"屏体本身"的换算：宽高（米）→ 尺寸、观看距离 → 参考屏幕尺寸、
客户授权尺寸时的确定性参考尺寸。观看距离/点间距在各自模块里。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .constants import (
    DELEGATED_SIZE_ASPECT,
    DELEGATED_SIZE_DISTANCE_FACTOR,
    DELEGATED_SIZE_MAX_H_M,
    DELEGATED_SIZE_MIN_H_M,
    SCREEN_SIZE_BY_DISTANCE,
)


def screen_dims_m(facts: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """从事实里取屏体宽高（米）；只有一条边时另一条按 16:9 估。"""
    facts = dict(facts or {})
    width = facts.get("target_width_m")
    height = facts.get("target_height_m")
    if width is None and facts.get("target_width_mm") is not None:
        width = float(facts["target_width_mm"]) / 1000
    if height is None and facts.get("target_height_mm") is not None:
        height = float(facts["target_height_mm"]) / 1000
    hint = facts.get("screen_size_hint_mm")
    if width is None and height is None and hint:
        width = float(hint) / 1000
    width = float(width) if width else None
    height = float(height) if height else None
    if width and not height:
        height = width * 9 / 16
    elif height and not width:
        width = height * 16 / 9
    return width, height


def screen_size_for_distance(distance_m: Optional[float]) -> Optional[str]:
    """观看距离 → 建议屏幕尺寸（参考值，主要用于 LCD / IFP 场景）。"""
    if distance_m is None:
        return None
    try:
        value = float(distance_m)
    except (TypeError, ValueError):
        return None
    for limit, size in SCREEN_SIZE_BY_DISTANCE:
        if value <= limit:
            return size
    return None


def suggest_screen_size(facts: Any) -> Optional[Tuple[float, float]]:
    """客户授权 AI 决定尺寸时，按观看距离给出**参考尺寸**（米）。

    这是 Python 的确定性推导，不是 LLM 猜数字；推导结果只用于工程计算与话术参考，
    **不会写进客户的确认事实**。
    """
    from .viewing_distance import parse_distance

    if facts is None:
        return None
    if hasattr(facts, "to_facts"):
        facts = facts.to_facts()
    distance = None
    if isinstance(facts, dict):
        distance = facts.get("viewing_distance_m") or facts.get("distance")
    distance = parse_distance(distance)
    if not distance or distance <= 0:
        return None
    height = max(
        DELEGATED_SIZE_MIN_H_M,
        min(DELEGATED_SIZE_MAX_H_M, float(distance) / DELEGATED_SIZE_DISTANCE_FACTOR),
    )
    width = height * DELEGATED_SIZE_ASPECT
    return round(width, 2), round(height, 2)


__all__ = ["screen_dims_m", "screen_size_for_distance", "suggest_screen_size"]
