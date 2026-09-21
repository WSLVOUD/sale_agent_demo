"""v2.5 Phase 6 / v2.5+：Engineering Feasibility Engine（工程可行性统一判断）。

客户口径（2026-09-21，关于分辨率）：

  · 分辨率**一律按"屏体大约能拼到"处理** —— 与"4K 输入 / 4K 屏体"无关，
    也不再向客户提任何澄清问题；
  · 只要这个尺寸的屏体（用到目录里最细的点间距）**大约**能达到客户要的分辨率
    就算可行，"达到或超过"同样算可行；
  · 如果连最细的点间距也达不到 → 不再推荐，而是直接告诉客户
    "这个尺寸下达不到你要的分辨率"，并给出可执行的调整方向：
        ① 需要的点间距  ② 标准尺寸（目标分辨率 × 点间距）  ③ 或者降低分辨率；
  · 客户没提分辨率 → 不做任何分辨率约束，按默认推荐。

状态：CONFLICT（数据自相矛盾）/ IMPOSSIBLE（这个尺寸做不到）/
      NEED_ADJUSTMENT（换成更细的 P 值就能做到，不阻塞选择） / FEASIBLE
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .conflicts import detect_engineering_conflicts
from .resolution import (
    ResolutionFitResult,
    ResolutionRequirement,
    fit_resolution,
    parse_resolution,
)
from .screen_geometry import actual_pixel_resolution, min_achievable_deviation

FEASIBLE = "FEASIBLE"
NEED_ADJUSTMENT = "NEED_ADJUSTMENT"
IMPOSSIBLE = "IMPOSSIBLE"
CONFLICT = "CONFLICT"

# 目录里拿不到模组几何时，用业界通用模组尺寸估"几何上最小能做到的偏差"
# （真正精确的判断在型号级 check_model_feasibility，这里只用于档位划分）。
DEFAULT_MODULE_WIDTH_MM = 320.0
DEFAULT_MODULE_HEIGHT_MM = 160.0
# 标准尺寸大到这里还没意义就只提"更细的点间距 / 降低分辨率"
MAX_USEFUL_STANDARD_WIDTH_M = 30.0


@dataclass
class FeasibilityResult:
    feasible: bool = True
    status: str = FEASIBLE
    resolution_result: Optional[ResolutionFitResult] = None
    geometry_result: Dict[str, Any] = field(default_factory=dict)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    # 给客户的**结论话术**（不是提问）：做不到时说清"为什么 + 怎么调整"
    message: str = ""
    required_pitch_mm: Optional[float] = None
    standard_size_m: Optional[Tuple[float, float]] = None
    achievable_resolution: Optional[Tuple[int, int]] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feasible": self.feasible,
            "status": self.status,
            "resolution_result": self.resolution_result.to_dict() if self.resolution_result else None,
            "geometry_result": dict(self.geometry_result),
            "conflicts": list(self.conflicts),
            "alternatives": list(self.alternatives),
            "message": self.message,
            "required_pitch_mm": self.required_pitch_mm,
            "standard_size_m": list(self.standard_size_m) if self.standard_size_m else None,
            "achievable_resolution": (
                list(self.achievable_resolution) if self.achievable_resolution else None
            ),
            "notes": list(self.notes),
        }


# ── 纯计算辅助 ──────────────────────────────────────────────────────────────
def required_pitch_mm(
    target: Optional[Tuple[int, int]],
    width_mm: float,
    height_mm: float,
) -> Optional[float]:
    """要达到目标分辨率，点间距最多能有多大（mm）——取宽高两个方向里更严的那个。"""
    if not target or not width_mm or not height_mm:
        return None
    width_px, height_px = int(target[0]), int(target[1])
    if min(width_px, height_px) <= 0 or min(width_mm, height_mm) <= 0:
        return None
    return round(min(float(width_mm) / width_px, float(height_mm) / height_px), 3)


def standard_size_for_resolution(
    target: Optional[Tuple[int, int]],
    pitch_mm: Optional[float],
) -> Optional[Tuple[float, float]]:
    """按目标分辨率 + 给定点间距给出"标准尺寸"（米）。"""
    if not target or not pitch_mm or float(pitch_mm) <= 0:
        return None
    width_m = int(target[0]) * float(pitch_mm) / 1000
    height_m = int(target[1]) * float(pitch_mm) / 1000
    return round(width_m, 2), round(height_m, 2)


def finest_pitch_in(models: Any) -> Optional[float]:
    """产品目录里最细的点间距（mm）；目录为空时返回 None。"""
    pitches = [
        float(getattr(model, "pixel_pitch_mm", 0) or 0)
        for model in (models or [])
    ]
    pitches = [value for value in pitches if value > 0]
    return min(pitches) if pitches else None


def _physical_size_mm(profile: Any) -> Tuple[float, float]:
    """档案里的目标尺寸（统一成毫米）；客户只给了一边就返回 0。"""
    width = getattr(profile, "target_width_m", None)
    height = getattr(profile, "target_height_m", None)
    if not width and getattr(profile, "target_width_mm", None):
        width = float(profile.target_width_mm) / 1000
    if not height and getattr(profile, "target_height_mm", None):
        height = float(profile.target_height_mm) / 1000
    return float(width or 0) * 1000, float(height or 0) * 1000


def _resolution_requirement(profile: Any) -> Optional[ResolutionRequirement]:
    """从档案里取分辨率需求（"大约达到"的目标值）。

    注意：v2.5+ 不再区分 INPUT / DISPLAY —— 只要客户提了分辨率，就按
    "屏体大约要达到"处理。
    """
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


def _module_pixels(
    module_width_mm: Optional[float],
    module_height_mm: Optional[float],
    pitch_mm: Optional[float],
) -> Tuple[int, int]:
    pitch = float(pitch_mm) if pitch_mm else 0.0
    if pitch <= 0:
        return 1, 1
    width = float(module_width_mm or DEFAULT_MODULE_WIDTH_MM)
    height = float(module_height_mm or DEFAULT_MODULE_HEIGHT_MM)
    return max(1, int(round(width / pitch))), max(1, int(round(height / pitch)))


def check_feasibility(
    profile: Any,
    *,
    resolution: Optional[ResolutionRequirement] = None,
    finest_pitch_mm: Optional[float] = None,
    module_width_mm: Optional[float] = None,
    module_height_mm: Optional[float] = None,
) -> FeasibilityResult:
    """档案级工程可行性判断（不针对具体型号）。

    ``finest_pitch_mm``：产品目录里最细的点间距（由调用方传入 —— 工程层不依赖目录）。
    """
    result = FeasibilityResult()
    if profile is None:
        return result

    # ① 已知工程冲突（屏比房间大、室内 + P10、进深 < 屏高…）→ 先澄清
    conflicts = [item.to_dict() for item in detect_engineering_conflicts(profile)]
    stored_conflicts = list(getattr(profile, "conflicts", None) or [])
    if conflicts or stored_conflicts:
        result.feasible = False
        result.status = CONFLICT
        result.conflicts = conflicts or [{"message": str(item)} for item in stored_conflicts]
        result.message = (
            "Those numbers can't all be true at the same time, so I can't size a screen "
            "from them yet."
        )
        result.notes.append("存在工程冲突，先澄清再推荐")
        return result

    requirement = resolution or _resolution_requirement(profile)
    if requirement is None or requirement.target is None:
        # 客户没提分辨率 → 不做任何分辨率约束，按默认推荐
        return result

    target = requirement.target
    width_mm, height_mm = _physical_size_mm(profile)
    pitch = getattr(profile, "pixel_pitch_mm", None)
    tolerance = requirement.tolerance
    module_px = _module_pixels(
        module_width_mm, module_height_mm, finest_pitch_mm or pitch
    )
    deviation = min_achievable_deviation(
        target_width_px=target[0], target_height_px=target[1],
        module_width_px=module_px[0], module_height_px=module_px[1],
    )

    # ② 尺寸已知 + 知道目录最细点间距：
    #    先看"这个尺寸的物理极限（最细 P 值）能不能大约达到" —— 达不到就没救，
    #    直接告诉客户要多大尺寸 / 要多细的点间距 / 或者降低分辨率。
    if width_mm and height_mm and finest_pitch_mm:
        ceiling = actual_pixel_resolution(
            actual_width_mm=width_mm,
            actual_height_mm=height_mm,
            pixel_pitch_mm=float(finest_pitch_mm),
        )
        if ceiling:
            best = fit_resolution(
                target, ceiling, tolerance=tolerance, min_achievable_deviation=deviation
            )
            result.resolution_result = best
            result.achievable_resolution = ceiling
            result.geometry_result = {
                "target_resolution": list(target),
                "target_size_mm": [round(width_mm, 1), round(height_mm, 1)],
                "finest_pitch_mm": float(finest_pitch_mm),
                "best_resolution_at_this_size": list(ceiling),
            }
            if not best.acceptable:
                standard = standard_size_for_resolution(target, finest_pitch_mm)
                if standard and standard[0] > MAX_USEFUL_STANDARD_WIDTH_M:
                    standard = None
                result.feasible = False
                result.status = IMPOSSIBLE
                result.standard_size_m = standard
                result.required_pitch_mm = required_pitch_mm(target, width_mm, height_mm)
                result.message = _size_too_small_message(
                    target, ceiling, width_mm, height_mm, standard
                )
                result.alternatives = _size_alternatives(ceiling, standard)
                result.notes.append("这个尺寸（即使最细点间距）达不到客户要的分辨率")
                return result

    # ③ 用客户当前（或系统推断）的 P 值再核一遍：够 → 可行；
    #    不够但换更细的 P 值能做到、且客户没把 P 值定死 → 交给引擎去更细的档里选。
    if pitch and width_mm and height_mm:
        actual = actual_pixel_resolution(
            actual_width_mm=width_mm, actual_height_mm=height_mm, pixel_pitch_mm=float(pitch)
        )
        if actual:
            fit = fit_resolution(
                target, actual, tolerance=tolerance, min_achievable_deviation=deviation
            )
            result.resolution_result = fit
            result.achievable_resolution = actual
            result.geometry_result.update({
                "target_resolution": list(target),
                "current_pitch_mm": float(pitch),
                "current_resolution": list(actual),
            })
            if fit.acceptable:
                result.notes.append(
                    f"按 {float(pitch):g}mm 点间距实际拼接分辨率 {actual[0]}x{actual[1]} "
                    f"≈ 目标（{fit.fit_level}）"
                )
                return result

            needed = required_pitch_mm(target, width_mm, height_mm)
            result.required_pitch_mm = needed
            standard = standard_size_for_resolution(target, float(pitch))
            if standard and standard[0] > MAX_USEFUL_STANDARD_WIDTH_M:
                standard = None
            result.standard_size_m = standard
            pitch_is_fixed = bool(
                getattr(profile, "slot_is_confirmed", lambda *_: False)("pixel_pitch")
            )
            finer_available = bool(
                needed and finest_pitch_mm and float(finest_pitch_mm) <= float(needed) + 1e-6
            )
            if needed and not pitch_is_fixed and finer_available:
                # 客户没把 P 值定死，而且更细的 P 值能做到 → 不阻塞，交给引擎
                result.status = NEED_ADJUSTMENT
                result.notes.append(
                    f"当前 {float(pitch):g}mm 点间距只能做到 {actual[0]}x{actual[1]}；"
                    f"要达到 {target[0]}x{target[1]} 需要 P≤{float(needed):.2f}，"
                    "由引擎在更细的档位里选型号"
                )
                return result

            # 客户把 P 值定死了，或者连最细的也做不到 → 直说 + 给调整方向
            result.feasible = False
            result.status = IMPOSSIBLE
            result.message = _pitch_or_size_message(
                target, actual, float(pitch), needed, standard
            )
            result.alternatives = _pitch_alternatives(actual, needed, standard)
            result.notes.append("当前 P 值 / 尺寸达不到客户要的分辨率")
            return result

    # ④ 只有分辨率目标，尺寸或点间距还没给 → 先记录，不阻塞
    result.notes.append("分辨率目标已记录；等尺寸 / 点间距确定后再做拼接可行性判断")
    return result


# ── 给客户的话术（结论，不是提问）──────────────────────────────────────────
def _size_too_small_message(
    target: Tuple[int, int],
    achievable: Tuple[int, int],
    width_mm: float,
    height_mm: float,
    standard: Optional[Tuple[float, float]],
) -> str:
    text = (
        f"A {width_mm / 1000:.2f}m x {height_mm / 1000:.2f}m screen tops out at about "
        f"{achievable[0]}x{achievable[1]} pixels, even with our finest pitch, so this size "
        f"can't reach {target[0]}x{target[1]}."
    )
    if standard:
        text += (
            f" For a true {target[0]}x{target[1]} the screen would need to be about "
            f"{standard[0]}m x {standard[1]}m."
        )
    text += (
        f" Otherwise we keep your size and go with about {achievable[0]}x{achievable[1]} "
        "— which is plenty for normal viewing."
    )
    return text


def _pitch_or_size_message(
    target: Tuple[int, int],
    actual: Tuple[int, int],
    pitch_mm: float,
    needed_pitch_mm: Optional[float],
    standard_size: Optional[Tuple[float, float]],
) -> str:
    text = (
        f"At {pitch_mm:g}mm pitch this screen comes to about {actual[0]}x{actual[1]} pixels, "
        f"short of the {target[0]}x{target[1]} you're after."
    )
    if needed_pitch_mm:
        text += (
            f" Hitting {target[0]}x{target[1]} at this size needs a pitch of about "
            f"P{needed_pitch_mm:.2f} or finer"
        )
    if standard_size:
        text += (
            f"; if the pitch has to stay at {pitch_mm:g}mm, the screen itself would need to be "
            f"about {standard_size[0]}m x {standard_size[1]}m"
        )
    text += (
        f". Or we keep this size and settle for about {actual[0]}x{actual[1]} pixels, "
        "which is still a clean, sharp picture."
    )
    return text


def _size_alternatives(
    achievable: Tuple[int, int],
    standard: Optional[Tuple[float, float]],
) -> List[str]:
    items: List[str] = []
    if standard:
        items.append(f"enlarge the screen to about {standard[0]}m x {standard[1]}m")
    items.append(f"or stay at this size with about {achievable[0]}x{achievable[1]} pixels")
    items.append("or tell us a lower resolution target you'd be happy with")
    return items


def _pitch_alternatives(
    actual: Tuple[int, int],
    needed_pitch_mm: Optional[float],
    standard: Optional[Tuple[float, float]],
) -> List[str]:
    items: List[str] = []
    if needed_pitch_mm:
        items.append(f"use a finer pitch (about P{needed_pitch_mm:.2f} or finer)")
    else:
        items.append("use a finer pitch")
    if standard:
        items.append(f"or enlarge the screen to about {standard[0]}m x {standard[1]}m")
    items.append(f"or settle for about {actual[0]}x{actual[1]} at this size")
    return items


# ── 型号级可行性 ────────────────────────────────────────────────────────────
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
    if requirement is None or requirement.target is None:
        return {"applicable": False, "reason": "客户没有分辨率要求"}

    width_mm = float(target_width_mm or _physical_size_mm(profile)[0] or 0)
    height_mm = float(target_height_mm or _physical_size_mm(profile)[1] or 0)
    cabinet_w = float(getattr(model, "cabinet_width_mm", 0) or 0)
    cabinet_h = float(getattr(model, "cabinet_height_mm", 0) or 0)
    pitch = float(getattr(model, "pixel_pitch_mm", 0) or 0)
    if min(width_mm, height_mm, cabinet_w, cabinet_h, pitch) <= 0:
        return {"applicable": False, "reason": "缺少尺寸 / 箱体 / 点间距信息"}

    # 横拼与竖拼（旋转 90°）两种排布都算一遍，取更接近目标分辨率的那种 ——
    # 客户口径：尺寸可以横着拼也可以竖着拼，两种都要能算。
    layouts = _model_layouts(width_mm, height_mm, cabinet_w, cabinet_h, pitch)
    best_key, best = min(layouts.items(), key=lambda item: item[1]["rank"])
    target = requirement.target
    module_px = _module_pixels(
        getattr(model, "module_width_mm", None),
        getattr(model, "module_height_mm", None),
        pitch,
    )
    fit = fit_resolution(
        target, best["resolution"], tolerance=requirement.tolerance,
        min_achievable_deviation=min_achievable_deviation(
            target_width_px=target[0], target_height_px=target[1],
            module_width_px=module_px[0], module_height_px=module_px[1],
        ),
    )
    return {
        "applicable": True,
        "model": getattr(model, "model", ""),
        "geometry": best["geometry"],
        "best_layout": best_key,
        "layouts": {key: value["geometry"] for key, value in layouts.items()},
        "resolution_fit": fit.to_dict(),
        "acceptable": fit.acceptable,
    }


def _model_layouts(
    width_mm: float,
    height_mm: float,
    cabinet_w: float,
    cabinet_h: float,
    pitch: float,
) -> Dict[str, Dict[str, Any]]:
    """横拼 / 竖拼（箱体旋转 90°）两种排布的实际尺寸与像素分辨率。"""
    layouts: Dict[str, Dict[str, Any]] = {}
    for key, (box_w, box_h) in (
        ("landscape", (cabinet_w, cabinet_h)),
        ("portrait", (cabinet_h, cabinet_w)),
    ):
        columns = max(1, int(width_mm // box_w))
        rows = max(1, int(height_mm // box_h))
        actual_w = columns * box_w
        actual_h = rows * box_h
        resolution = actual_pixel_resolution(
            actual_width_mm=actual_w, actual_height_mm=actual_h, pixel_pitch_mm=pitch
        )
        rank = _layout_rank(resolution)
        layouts[key] = {
            "columns": columns,
            "rows": rows,
            "actual_width_mm": round(actual_w, 1),
            "actual_height_mm": round(actual_h, 1),
            "resolution": resolution,
            "rank": rank,
            "geometry": {
                "columns": columns,
                "rows": rows,
                "actual_width_mm": round(actual_w, 1),
                "actual_height_mm": round(actual_h, 1),
            },
        }
    return layouts


def _layout_rank(resolution: Optional[Tuple[int, int]]) -> float:
    """横拼 / 竖拼两种排布的排序键：像素总数越多越好（越接近/超过目标越稳）。"""
    if not resolution:
        return 0.0
    return -float(resolution[0]) * float(resolution[1])


__all__ = [
    "CONFLICT",
    "FEASIBLE",
    "FeasibilityResult",
    "IMPOSSIBLE",
    "NEED_ADJUSTMENT",
    "check_feasibility",
    "check_model_feasibility",
    "finest_pitch_in",
    "required_pitch_mm",
    "standard_size_for_resolution",
]
