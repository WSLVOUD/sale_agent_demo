"""v2.5 Phase 6：Engineering Feasibility Engine（工程可行性统一判断）。

链路（计划第十三 / 十四节）：

    Hard Constraint → Feasibility → Filter → Feasible Candidates
        → Recommendation Engine（只负责在可行集合里排序）

本模块把 尺寸 / P值 / 分辨率 / 比例 / 箱体几何 / 观看距离 / 亮度 / 环境 / 安装方式
统一纳入判断，输出：

    FeasibilityResult(
        feasible, status, resolution_result, aspect_ratio_result,
        geometry_result, conflicts, alternatives, notes,
    )

状态取值：FEASIBLE / CONFLICT / IMPOSSIBLE / NEED_CLARIFICATION
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .constants import ASPECT_DEVIATION_ACCEPTABLE
from .conflicts import detect_engineering_conflicts
from .resolution import (
    ACCEPTABLE,
    DISPLAY,
    INPUT,
    NEAR_MATCH,
    UNKNOWN,
    ResolutionFitResult,
    ResolutionRequirement,
    fit_resolution,
    parse_resolution,
)
from .screen_geometry import (
    actual_pixel_resolution,
    min_achievable_deviation,
    stitching_geometry,
)

FEASIBLE = "FEASIBLE"
CONFLICT = "CONFLICT"
IMPOSSIBLE = "IMPOSSIBLE"
NEED_CLARIFICATION = "NEED_CLARIFICATION"

CLARIFY_4K_INPUT_OR_DISPLAY = (
    "Do you need the LED itself to match the target resolution, or is it enough that the "
    "control system accepts that signal as an input?"
)


@dataclass
class FeasibilityResult:
    feasible: bool = True
    status: str = FEASIBLE
    resolution_result: Optional[ResolutionFitResult] = None
    aspect_ratio_deviation: Optional[float] = None
    geometry_result: Dict[str, Any] = field(default_factory=dict)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    question: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feasible": self.feasible,
            "status": self.status,
            "resolution_result": self.resolution_result.to_dict() if self.resolution_result else None,
            "aspect_ratio_deviation": self.aspect_ratio_deviation,
            "geometry_result": dict(self.geometry_result),
            "conflicts": list(self.conflicts),
            "alternatives": list(self.alternatives),
            "question": self.question,
            "notes": list(self.notes),
        }


def _physical_aspect(profile: Any) -> Optional[float]:
    width = getattr(profile, "target_width_m", None)
    height = getattr(profile, "target_height_m", None)
    if width and height:
        return float(width) / float(height)
    return None


def _resolution_requirement(profile: Any) -> Optional[ResolutionRequirement]:
    """从档案里取分辨率需求（若有）。"""
    requirement = getattr(profile, "resolution_requirement", None)
    if isinstance(requirement, ResolutionRequirement):
        return requirement
    if isinstance(requirement, dict) and requirement:
        try:
            from dataclasses import fields as _fields

            allowed = {item.name for item in _fields(ResolutionRequirement)}
            return ResolutionRequirement(
                **{key: value for key, value in requirement.items() if key in allowed}
            )
        except Exception:  # pragma: no cover - 防御式
            return None
    raw = str(getattr(profile, "resolution_raw", "") or "")
    return parse_resolution(raw) if raw else None


def check_feasibility(profile: Any, *, resolution: Optional[ResolutionRequirement] = None) -> FeasibilityResult:
    """档案级工程可行性判断（不针对具体型号）。"""
    result = FeasibilityResult()
    if profile is None:
        return result

    # ① 已知工程冲突（屏比房间大、室内+P10、进深 < 屏高…）
    conflicts = [item.to_dict() for item in detect_engineering_conflicts(profile)]
    if conflicts or (getattr(profile, "conflicts", None) or []):
        result.feasible = False
        result.status = CONFLICT
        result.conflicts = conflicts or [{"message": str(m)} for m in profile.conflicts]
        result.notes.append("存在工程冲突，先澄清再推荐")
        return result

    requirement = resolution or _resolution_requirement(profile)
    if requirement is None:
        return result

    # ② 客户只说"要 4K"，含义不明 → 允许澄清（影响推荐时才问）
    if requirement.mode == UNKNOWN:
        result.status = NEED_CLARIFICATION
        result.question = CLARIFY_4K_INPUT_OR_DISPLAY
        result.notes.append("分辨率含义不明（INPUT / DISPLAY），不猜")
        result.resolution_result = ResolutionFitResult(
            target=requirement.target, fit_level="",
        )
        return result

    # ③ INPUT 需求：只要求控制系统能收这个信号 → 不进入 LED 拼接计算
    if requirement.mode == INPUT:
        result.notes.append("客户要的是 4K 输入能力，不要求 LED 本身达到该像素数")
        return result

    if requirement.mode != DISPLAY or requirement.target is None:
        return result

    # ④ DISPLAY 需求：先看物理比例是否冲突（尺寸定死时比例可能对不上）
    target_aspect = requirement.aspect_ratio
    physical_aspect = _physical_aspect(profile)
    if target_aspect and physical_aspect:
        deviation = abs(physical_aspect - target_aspect) / target_aspect
        result.aspect_ratio_deviation = round(deviation, 4)
        if deviation > ASPECT_DEVIATION_ACCEPTABLE:
            result.feasible = False
            result.status = CONFLICT
            result.conflicts.append({
                "slot": "resolution",
                "message": (
                    f"物理比例 {round(physical_aspect, 3)}:1 与目标分辨率比例 "
                    f"{round(target_aspect, 3)}:1 冲突"
                ),
            })
            result.question = (
                "Do you need the LED itself to match the target display resolution, "
                "or is the physical size the fixed requirement?"
            )
            result.notes.append("比例冲突：需要客户确认以哪个为准")
            return result

    # ⑤ 客户给了 P 值与尺寸 → 直接算出"用这个 P 值最多能做到多少像素"
    pitch = getattr(profile, "pixel_pitch_mm", None)
    width_mm = (getattr(profile, "target_width_m", None) or 0) * 1000
    height_mm = (getattr(profile, "target_height_m", None) or 0) * 1000
    if pitch and width_mm and height_mm:
        actual = actual_pixel_resolution(
            actual_width_mm=width_mm, actual_height_mm=height_mm, pixel_pitch_mm=float(pitch)
        )
        target = requirement.target
        if actual and target:
            fit = fit_resolution(
                target, actual, tolerance=requirement.tolerance,
            )
            result.resolution_result = fit
            result.geometry_result = {
                "target_resolution": list(target),
                "pitch_limited_resolution": list(actual),
                "pitch_mm": float(pitch),
            }
            if fit.fit_level in ("EXACT", NEAR_MATCH, ACCEPTABLE):
                result.notes.append(
                    f"按 {float(pitch):g}mm 点间距实际可做到 {actual[0]}x{actual[1]}（{fit.fit_level}）"
                )
                return result
            # ⑥ 达不到 → 给出调整方向（不自动改客户的尺寸/P值/分辨率）
            needed_pitch = width_mm / target[0]
            result.feasible = False
            result.status = IMPOSSIBLE
            result.alternatives = [
                f"用更细的点间距（约 P{needed_pitch:.2f} 或更细）才能到 {target[0]}x{target[1]}",
                "或者把屏幕尺寸做大",
                "或者把目标分辨率降下来",
                "如果只是要 4K 输入能力，重新确认需求即可",
            ]
            result.question = (
                f"With a {float(pitch):g}mm pitch and this size the panel can reach about "
                f"{actual[0]}x{actual[1]}, not {target[0]}x{target[1]}. "
                "Which is fixed for you: the pitch, the size, or the resolution?"
            )
            result.notes.append("当前条件无法同时满足，需要客户选择调整方向")
            return result

    # ⑦ 只有分辨率目标、还没尺寸/P值 → 等需求齐了再判断（不阻塞）
    result.notes.append("分辨率目标已记录；等尺寸 / P 值确定后再做拼接可行性判断")
    return result


def check_model_feasibility(
    profile: Any,
    model: Any,
    *,
    target_width_mm: Optional[float] = None,
    target_height_mm: Optional[float] = None,
    resolution: Optional[ResolutionRequirement] = None,
) -> Dict[str, Any]:
    """型号级可行性：这个型号 + 目标尺寸 + 客户分辨率目标，能不能拼到足够接近。"""
    requirement = resolution or _resolution_requirement(profile)
    width_mm = float(
        target_width_mm
        or (getattr(profile, "target_width_m", None) or 0) * 1000
        or 0
    )
    height_mm = float(
        target_height_mm
        or (getattr(profile, "target_height_m", None) or 0) * 1000
        or 0
    )
    cabinet_w = float(getattr(model, "cabinet_width_mm", 0) or 0)
    cabinet_h = float(getattr(model, "cabinet_height_mm", 0) or 0)
    pitch = float(getattr(model, "pixel_pitch_mm", 0) or 0)
    if min(width_mm, height_mm, cabinet_w, cabinet_h, pitch) <= 0:
        return {"applicable": False, "reason": "缺少尺寸 / 箱体 / 点间距信息"}
    if requirement is None or not requirement.is_display:
        return {"applicable": False, "reason": "客户没有 DISPLAY 级分辨率要求"}

    geometry = stitching_geometry(
        target_width_mm=width_mm, target_height_mm=height_mm,
        cabinet_width_mm=cabinet_w, cabinet_height_mm=cabinet_h,
        modules_per_cabinet=getattr(model, "modules_per_cabinet", 1) or 1,
        module_width_mm=getattr(model, "module_width_mm", None),
        module_height_mm=getattr(model, "module_height_mm", None),
    )
    actual = actual_pixel_resolution(
        actual_width_mm=geometry.get("actual_width_mm", width_mm),
        actual_height_mm=geometry.get("actual_height_mm", height_mm),
        pixel_pitch_mm=pitch,
    )
    fit = fit_resolution(
        requirement.target, actual, tolerance=requirement.tolerance,
        min_achievable_deviation=min_achievable_deviation(
            target_width_px=requirement.target[0], target_height_px=requirement.target[1],
            module_width_px=int(getattr(model, "module_width_mm", 0) or 0) // max(1, int(pitch)),
            module_height_px=int(getattr(model, "module_height_mm", 0) or 0) // max(1, int(pitch)),
        ),
    )
    return {
        "applicable": True,
        "model": getattr(model, "model", ""),
        "geometry": geometry,
        "resolution_fit": fit.to_dict(),
        "acceptable": fit.acceptable,
    }


__all__ = [
    "CLARIFY_4K_INPUT_OR_DISPLAY",
    "CONFLICT",
    "FEASIBLE",
    "FeasibilityResult",
    "IMPOSSIBLE",
    "NEED_CLARIFICATION",
    "check_feasibility",
    "check_model_feasibility",
]
