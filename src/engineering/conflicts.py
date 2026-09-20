"""v2.3 §9：Conflict 独立状态。

矛盾信息不要简单当成 UNKNOWN —— 要单独记成 CONFLICT，并且**冲突解决前禁止推荐**：

    Conflict → Recommendation Block → Clarification

这里只做**确定性**的工程冲突检测（物理上不可能 / 与业务口径直接矛盾），
客户语义上的矛盾（"说过室内又说室外"）仍然由既有链路写入 profile.conflicts。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

# 室内环境下，超过这个点间距就属于"明显不匹配"（业务口径室内 >30m 才允许 P10）
INDOOR_COARSE_PITCH_MM = 8.0
INDOOR_COARSE_DISTANCE_M = 30.0


@dataclass(frozen=True)
class EngineeringConflict:
    slot: str
    message: str
    evidence: str = ""

    def to_dict(self) -> dict:
        return {"slot": self.slot, "message": self.message, "evidence": self.evidence}


def detect_engineering_conflicts(profile: Any) -> List[EngineeringConflict]:
    """返回确定性工程冲突（可能为空）。

    目前覆盖三类：

      1. 场地面积 < 屏体面积（屏比房间还大）
      2. 室内 + 很粗的点间距（P8+）但观看距离并不远（< 30m）
      3. 屏幕宽度明显超出场地（宽 > 场地面积 ÷ 屏高）
    """
    if profile is None:
        return []
    conflicts: List[EngineeringConflict] = []

    width = getattr(profile, "target_width_m", None)
    height = getattr(profile, "target_height_m", None)
    area = getattr(profile, "room_area_sqm", None)
    depth = getattr(profile, "room_depth_m", None)

    if area and width and height:
        screen_area = float(width) * float(height)
        if screen_area > float(area) * 1.05:
            conflicts.append(EngineeringConflict(
                slot="size",
                message=(
                    f"屏体面积（{screen_area:.1f}㎡）比场地面积（{float(area):.1f}㎡）还大"
                ),
                evidence=f"target={width}x{height}m room_area={area}sqm",
            ))
    if area and width:
        # 场地面积 ÷ 屏宽 = 场地进深；屏宽明显超过它就是自相矛盾
        implied_depth = float(area) / float(width)
        if implied_depth < 1.0:
            conflicts.append(EngineeringConflict(
                slot="size",
                message=(
                    f"按面积反推的场地进深只有 {implied_depth:.1f}m，放不下 {float(width):.1f}m 宽的屏"
                ),
                evidence=f"room_area={area}sqm screen_width={width}m",
            ))
    if depth and height and float(depth) < float(height):
        conflicts.append(EngineeringConflict(
            slot="size",
            message=(
                f"场地进深（{float(depth):.1f}m）比屏高（{float(height):.1f}m）还小，视线距离不成立"
            ),
            evidence=f"room_depth={depth}m screen_height={height}m",
        ))

    environment = str(getattr(profile, "environment", "") or "").lower()
    pitch = getattr(profile, "pixel_pitch_mm", None)
    distance = getattr(profile, "viewing_distance_m", None)
    if (
        environment == "indoor"
        and pitch is not None
        and float(pitch) >= INDOOR_COARSE_PITCH_MM
        and (distance is None or float(distance) < INDOOR_COARSE_DISTANCE_M)
    ):
        conflicts.append(EngineeringConflict(
            slot="pixel_pitch",
            message=(
                f"室内使用却选了 {float(pitch):g}mm 这么粗的点间距（观看距离 "
                f"{'未知' if distance is None else format(float(distance), '.1f') + 'm'}）"
            ),
            evidence=f"environment=indoor pitch={pitch} distance={distance}",
        ))
    return conflicts


def conflict_slots(profile: Any) -> List[str]:
    """冲突字段（去重、保持顺序）。"""
    slots: List[str] = []
    for slot in (getattr(profile, "conflict_slots", None) or []):
        if slot and slot not in slots:
            slots.append(str(slot))
    for item in detect_engineering_conflicts(profile):
        if item.slot not in slots:
            slots.append(item.slot)
    return slots


def has_conflict(profile: Any) -> bool:
    return bool((getattr(profile, "conflicts", None) or []) or detect_engineering_conflicts(profile))


def conflict_message(profile: Any) -> Optional[str]:
    """给客户看的一句话（英文，客户口径：回复必须英文）。"""
    conflicts = detect_engineering_conflicts(profile)
    if conflicts:
        return (
            "Before I can pick a model I need to clear one thing up: "
            + conflicts[0].message
            + ". Could you confirm the numbers?"
        )
    stored = list(getattr(profile, "conflicts", None) or [])
    if stored:
        slot = (getattr(profile, "conflict_slots", None) or ["environment"])[0]
        return (
            f"Just to make sure I have it right, could you confirm the {slot}? "
            f"({stored[0]})"
        )
    return None


__all__ = [
    "EngineeringConflict",
    "conflict_message",
    "conflict_slots",
    "detect_engineering_conflicts",
    "has_conflict",
]
