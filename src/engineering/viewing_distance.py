"""v2.3 §4 / §12：观看距离的统一推导（唯一实现）。

客户给的东西千变万化（人数 / 面积 / 进深 / 屏尺寸 / 直接说距离），但都能折算成
同一个物理量：观看距离区间 [最近观众, 最远观众]。折算只用常量模块里的那几组公式，
所以以后客户换说法**不需要新增推荐规则**，只需要能解析出数字。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .constants import (
    FRONT_OFFSET_M,
    SCREEN_FAR_FACTOR,
    SCREEN_NEAR_FACTOR,
    SEAT_ROW_DEPTH_M,
    SEAT_WIDTH_M,
)
from .screen_geometry import screen_dims_m


@dataclass(frozen=True)
class ViewingDistanceEstimate:
    """推导出来的观看距离区间（米）。``source`` 说明这个数是从哪来的。"""

    nearest_m: Optional[float]
    farthest_m: float
    source: str          # room_depth / room_area / audience / screen_size

    @property
    def typical_m(self) -> float:
        if self.nearest_m:
            return round((float(self.nearest_m) + float(self.farthest_m)) / 2, 2)
        return round(float(self.farthest_m) * 0.6, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nearest_m": round(float(self.nearest_m), 2) if self.nearest_m else None,
            "farthest_m": round(self.farthest_m, 2),
            "typical_m": self.typical_m,
            "source": self.source,
        }


def estimate_viewing_distance(facts: Dict[str, Any]) -> Optional[ViewingDistanceEstimate]:
    """把"人数 / 面积 / 进深 / 屏尺寸"折算成观看距离区间（纯公式、确定性）。

    优先级（数字越靠前越直接）：

        room_depth  → 最远 ≈ 进深 − 0.5m（最后排贴着后墙）
        room_area   → 进深 ≈ 面积 ÷ 屏宽（没有屏宽时 ≈ √(面积 ÷ 2)）
        audience    → 每排座位数 ≈ 屏宽 ÷ 0.6m，排数 = 人数 ÷ 每排，
                      进深 ≈ 排数 × 0.9m + 屏前留距 2.5m
        screen_size → 最近 ≈ 1.5 × 屏高，最远 ≈ 3 × 屏高

    客户已经明确说了观看距离时返回 None（那条路径直接用客户给的数）。
    """
    if not facts:
        return None
    if facts.get("viewing_distance_m") or facts.get("distance"):
        return None

    width_m, height_m = screen_dims_m(dict(facts))
    source = ""
    farthest: Optional[float] = None

    depth = facts.get("room_depth_m")
    if depth:
        farthest = max(1.0, float(depth) - 0.5)
        source = "room_depth"
    if farthest is None:
        area = facts.get("room_area_sqm")
        if area and width_m:
            derived_depth = float(area) / float(width_m)
            if 1.0 <= derived_depth <= 200.0:
                farthest = max(1.0, derived_depth - 0.5)
                source = "room_area"
    if farthest is None:
        people = facts.get("audience_count")
        if people:
            # 知道屏宽 → 每排座位数 = 屏宽 ÷ 0.6m；不知道屏宽 → 按"座位区宽深比 2:1"
            # 的通用假设直接算排数（√(人数 ÷ 2)）。两者都是确定性公式。
            if width_m:
                seats_per_row = max(1, int(float(width_m) / SEAT_WIDTH_M))
                rows = int(-(-int(people) // seats_per_row))
            else:
                rows = max(1, int(round((float(people) / 2.0) ** 0.5)))
            farthest = rows * SEAT_ROW_DEPTH_M + FRONT_OFFSET_M
            source = "audience"
    if farthest is None:
        area_only = facts.get("room_area_sqm")
        if area_only:
            derived_depth = (float(area_only) / 2.0) ** 0.5
            if 1.0 <= derived_depth <= 200.0:
                farthest = max(1.0, derived_depth - 0.5)
                source = "room_area"
    if farthest is None and height_m:
        farthest = SCREEN_FAR_FACTOR * float(height_m)
        source = "screen_size"
    if farthest is None:
        return None

    # 最近观众：至少 1.5 × 屏高（否则看不全整屏），但不粗过最远观众的 70%。
    # 屏高未知时不做假设 —— 宁可只按"最远观众"给窗口，也不要凭空收紧下限。
    nearest: Optional[float] = None
    if height_m:
        nearest = max(1.0, min(SCREEN_NEAR_FACTOR * float(height_m), farthest * 0.7))
    return ViewingDistanceEstimate(nearest_m=nearest, farthest_m=farthest, source=source)


def parse_distance(text: Any) -> Optional[float]:
    """从 "4米" / "4m" / "4000mm" 这类文本解析观看距离（米）。"""
    if text is None or text == "":
        return None
    if isinstance(text, (int, float)):
        return float(text) if text else None
    match = re.search(r"(\d+(?:\.\d+)?)", str(text))
    if not match:
        return None
    value = float(match.group(1))
    lowered = str(text).lower()
    if "mm" in lowered or "毫米" in lowered or value > 100:
        value = value / 1000
    return value or None


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "是")


def environment_from_facts(facts: Dict[str, Any]) -> Optional[str]:
    """从事实字典推导使用环境（兼容 indoor/outdoor 布尔与 environment 字符串两套写法）。"""
    facts = dict(facts or {})
    environment = facts.get("environment")
    if environment in ("indoor", "outdoor", "semi_outdoor"):
        return environment
    if _truthy(facts.get("semi_outdoor")):
        return "semi_outdoor"
    if _truthy(facts.get("outdoor")):
        return "outdoor"
    if _truthy(facts.get("indoor")):
        return "indoor"
    return None


__all__ = [
    "ViewingDistanceEstimate",
    "environment_from_facts",
    "estimate_viewing_distance",
    "parse_distance",
]
