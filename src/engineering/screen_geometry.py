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


def stitching_geometry(
    *,
    target_width_mm: float,
    target_height_mm: float,
    cabinet_width_mm: float,
    cabinet_height_mm: float,
    modules_per_cabinet: int = 1,
    module_width_mm: Optional[float] = None,
    module_height_mm: Optional[float] = None,
) -> Dict[str, Any]:
    """由目标物理尺寸 + 箱体几何算"实际能拼出来的尺寸 / 箱体数 / 模组数"。

    这是 v2.5 Phase 4 的纯几何部分（不依赖产品目录）：宽高各自向下取整到
    箱体（横拼）或旋转 90°（竖拼）能拼出的最大整数箱体数。
    """
    if min(target_width_mm, target_height_mm, cabinet_width_mm, cabinet_height_mm) <= 0:
        return {}
    columns = max(1, int(float(target_width_mm) // float(cabinet_width_mm)))
    rows = max(1, int(float(target_height_mm) // float(cabinet_height_mm)))
    actual_width = columns * float(cabinet_width_mm)
    actual_height = rows * float(cabinet_height_mm)
    cabinets = columns * rows
    modules = cabinets * max(1, int(modules_per_cabinet or 1))
    result = {
        "columns": columns,
        "rows": rows,
        "cabinet_count": cabinets,
        "module_count": modules,
        "actual_width_mm": round(actual_width, 1),
        "actual_height_mm": round(actual_height, 1),
        "actual_aspect_ratio": round(actual_width / actual_height, 4) if actual_height else None,
    }
    if module_width_mm and module_height_mm:
        result["module_columns"] = max(1, int(actual_width // float(module_width_mm)))
        result["module_rows"] = max(1, int(actual_height // float(module_height_mm)))
    return result


def actual_pixel_resolution(
    *,
    actual_width_mm: float,
    actual_height_mm: float,
    pixel_pitch_mm: float,
) -> Optional[Tuple[int, int]]:
    """实际物理尺寸 + 点间距 → 实际可拼接的像素分辨率。"""
    if min(float(actual_width_mm), float(actual_height_mm), float(pixel_pitch_mm)) <= 0:
        return None
    width_px = int(round(float(actual_width_mm) / float(pixel_pitch_mm)))
    height_px = int(round(float(actual_height_mm) / float(pixel_pitch_mm)))
    if width_px <= 0 or height_px <= 0:
        return None
    return width_px, height_px


def size_orientations(width_mm: float, height_mm: float) -> Tuple[Tuple[str, float, float], ...]:
    """客户报的 "x × y" 尺寸的两种摆法（**不区分哪一边是宽**）。

    客户口径（2026-09-21）：
        客户说 "3*5" 时不要再追问"哪个是宽哪个是高" —— 两种摆法都算，
        哪种摆法能拼到客户要的分辨率就按哪种（做屏本来就是可以横着装也可以竖着装）。

    返回 ``(("as_given", x, y), ("swapped", y, x))``；正方形只返回一种。
    """
    width = float(width_mm or 0)
    height = float(height_mm or 0)
    if min(width, height) <= 0:
        return ()
    options = [("as_given", width, height)]
    if abs(width - height) > 1e-6:
        options.append(("swapped", height, width))
    return tuple(options)


def min_achievable_deviation(
    *,
    target_width_px: int,
    target_height_px: int,
    module_width_px: int,
    module_height_px: int,
) -> Optional[float]:
    """几何上"最小可能偏差"：半个模组的像素数 ÷ 目标像素数。

    客户口径：不是所有项目都用同一个固定容差 —— 例如目标是 3840、
    模组宽 320px 时，最少只能差到半个模组（160px ≈ 4.2%），
    这种情况应当算 NEAR_MATCH（已经拼到最接近），而不是直接判失败。
    """
    if min(target_width_px, target_height_px) <= 0:
        return None
    h = (max(1, int(module_width_px)) / 2) / target_width_px
    v = (max(1, int(module_height_px)) / 2) / target_height_px
    return round(max(h, v), 4)


__all__ = [
    "actual_pixel_resolution",
    "min_achievable_deviation",
    "screen_dims_m",
    "screen_size_for_distance",
    "size_orientations",
    "stitching_geometry",
    "suggest_screen_size",
]
