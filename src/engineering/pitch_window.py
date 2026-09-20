"""v2.3 §4 / §12：观看距离 → 点间距窗口（唯一实现）。

三件事都只在这里做一次：

  1. `preferred_pitch_for_environment()` —— 客户口径的业务规则（环境 + 距离 → 区间 + 首选）
  2. `pitch_window_for_distances()`      —— 物理窗口（最远 ÷ 5 ~ 最远 ÷ 1.5，再用最近收紧）
  3. `fallback_pitch_band()`             —— 连距离都推不出来时的保守兜底档

`infer_technical_parameters()` 只负责把三者的结果收口，不再自己维护规则表。
"""
from __future__ import annotations

from typing import Optional, Tuple

from .constants import (
    DEFAULT_FALLBACK_PITCH_BAND,
    FALLBACK_PITCH_BAND,
    INDOOR_PITCH_TABLE,
    OUTDOOR_PITCH_TABLE,
    PITCH_FAR_LIMIT_M_PER_MM,
    PITCH_NEAR_LIMIT_M_PER_MM,
    PITCH_OPTIMAL_M_PER_MM,
    VIEWING_DISTANCE_PITCH_TABLE,
)


def preferred_pitch_for_environment(
    environment: Optional[str],
    distance_m: Optional[float],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """按"环境 + 观看距离"给出 (点间距下限, 上限, 首选值)，单位 mm。

    这是客户口径的业务规则，比通用距离表优先；客户自己点名点间距时不使用。
    """
    env = str(environment or "").strip().lower()
    if distance_m is None:
        return None, None, None
    try:
        value = float(distance_m)
    except (TypeError, ValueError):
        return None, None, None
    if value <= 0:
        return None, None, None

    # semi_outdoor（半户外）按室外口径处理（客户口径里"室外"包含门头/半户外）
    table = (
        INDOOR_PITCH_TABLE
        if env == "indoor"
        else OUTDOOR_PITCH_TABLE
        if env in ("outdoor", "semi_outdoor")
        else None
    )
    if not table:
        return None, None, None
    # 业务口径按"以内"理解（例如"6~20m 以内 P5 都合适"→ 20m 也归 P5 档）
    for limit, low, high, target in table:
        if value <= limit:
            return low, high, target
    last = table[-1]
    return last[1], last[2], last[3]


def pitch_range_for_distance(
    distance_m: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """观看距离（米）→ 推荐点间距区间（mm，通用距离表）。"""
    if distance_m is None:
        return None, None
    try:
        value = float(distance_m)
    except (TypeError, ValueError):
        return None, None
    if value <= 0:
        return None, None
    for index, (limit, pitch_min, pitch_max) in enumerate(VIEWING_DISTANCE_PITCH_TABLE):
        # 第一档（近距离）取闭区间：客户常说"2 米以内"，2.0m 应归入最细点间距档；
        # 其余档位保持左闭右开（4m 属于 4~8m 档），与 v1.0 的工程规则一致。
        if (value <= limit) if index == 0 else (value < limit):
            return pitch_min, pitch_max
    return 6.0, 10.0


def pitch_window_for_distances(
    nearest_m: Optional[float],
    farthest_m: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """观看距离区间 → 点间距窗口 (下限, 上限)，单位 mm。

    下限 = 最远观看距离 ÷ 5   （比这更细，远端观众根本看不出差别 → 白花钱）
    上限 = 最远观看距离 ÷ 1.5 （比这更粗，远端观众就看不清）
    再用最近观众收紧上限：点间距不能粗过"最近观众 ÷ 1m/mm"。
    """
    if not farthest_m or float(farthest_m) <= 0:
        return None, None
    farthest = float(farthest_m)
    low = farthest / PITCH_FAR_LIMIT_M_PER_MM
    high = farthest / (PITCH_OPTIMAL_M_PER_MM / 2)     # ÷1.5
    if nearest_m and float(nearest_m) > 0:
        high = min(high, float(nearest_m) / PITCH_NEAR_LIMIT_M_PER_MM)
    if high < low:
        low = high
    return round(max(low, 0.5), 2), round(max(high, 0.5), 2)


def fallback_pitch_band(environment: Optional[str]) -> Tuple[float, float, float]:
    """连观看距离都推不出来时的兜底档（按环境给保守区间，偏粗不偏细）。"""
    env = str(environment or "").strip().lower()
    return FALLBACK_PITCH_BAND.get(env, DEFAULT_FALLBACK_PITCH_BAND)


__all__ = [
    "fallback_pitch_band",
    "pitch_range_for_distance",
    "pitch_window_for_distances",
    "preferred_pitch_for_environment",
]
