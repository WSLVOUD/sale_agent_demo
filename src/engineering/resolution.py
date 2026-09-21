"""v2.5 Phase 3 / 5：分辨率需求（Resolution Requirement）与"接近程度"判断。

客户口径（2026-09-21 修订）：

  · 分辨率**一律按"屏体大约能达到"处理** —— 和"这是输入信号还是屏体指标"无关，
    所以不再区分含义、也不再问澄清问题（``mode`` 仅作为记录/排查用）；
  · 屏体拼出来的像素数 **达到或超过** 客户要的分辨率即可，不要求像素级严丝合缝
    （MEETS_OR_EXCEEDS）；略低一点但偏差在容差内也算（NEAR_MATCH / ACCEPTABLE）；
  · 偏差大到明显达不到时，由可行性层直接告诉客户：需要更细的点间距 / 更大的尺寸，
    或者降低分辨率目标；
  · 1080P / 2K / 1440P / 4K / 8K / 自定义 WxH 全部支持；
  · 阈值集中放在 constants 里，后续用 Golden Dataset 调整。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .constants import (
    RESOLUTION_FIT_ACCEPTABLE,
    RESOLUTION_FIT_NEAR,
    RESOLUTION_FIT_NOT_ACCEPTABLE,
    ASPECT_DEVIATION_ACCEPTABLE,
)

INPUT = "INPUT"
DISPLAY = "DISPLAY"
UNKNOWN = "UNKNOWN"

EXACT = "EXACT"
MEETS_OR_EXCEEDS = "MEETS_OR_EXCEEDS"
NEAR_MATCH = "NEAR_MATCH"
ACCEPTABLE = "ACCEPTABLE"
NOT_ACCEPTABLE = "NOT_ACCEPTABLE"
IMPOSSIBLE = "IMPOSSIBLE"

# 常见分辨率（2K 取 DCI 2048×1080 —— 与 1080P 区分开，客户口径把两者分开列）
RESOLUTION_PRESETS: Dict[str, Tuple[int, int]] = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "fhd": (1920, 1080),
    "2k": (2048, 1080),
    "1440p": (2560, 1440),
    "qhd": (2560, 1440),
    "4k": (3840, 2160),
    "uhd": (3840, 2160),
    "8k": (7680, 4320),
}

_TOKEN_RE = re.compile(
    r"\b(8k|4k|2k|1440p|1080p|720p|uhd|qhd|fhd)\b", re.IGNORECASE
)
_WXH_RE = re.compile(r"\b(\d{3,5})\s*[x×*]\s*(\d{3,5})\b", re.IGNORECASE)
_INPUT_RE = re.compile(
    r"\b(?:support(?:s)?|input|accept(?:s)?|feed|signal|source|interface)\b|输入|接入|信号",
    re.IGNORECASE,
)
_DISPLAY_RE = re.compile(
    r"\b(?:led itself|panel itself|the screen itself|display|pixels?|resolution of the screen|"
    r"native)\b|屏体本身|屏幕本身|实际分辨率|像素",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ResolutionRequirement:
    """客户的分辨率需求（计划第七节的结构）。"""

    target_width: Optional[int] = None
    target_height: Optional[int] = None
    mode: str = UNKNOWN
    tolerance: Optional[float] = None
    source: str = "customer"
    status: str = "CONFIRMED"
    raw: str = ""

    @property
    def target(self) -> Optional[Tuple[int, int]]:
        if self.target_width and self.target_height:
            return int(self.target_width), int(self.target_height)
        return None

    @property
    def aspect_ratio(self) -> Optional[float]:
        if not self.target:
            return None
        width, height = self.target
        return round(width / height, 4)

    @property
    def is_display(self) -> bool:
        return self.mode == DISPLAY and self.target is not None

    @property
    def constrains_screen(self) -> bool:
        """客户提到分辨率时是否构成对**屏体**的约束。

        v2.5+ 客户口径：只要客户说了分辨率，就按"屏体大约要达到"处理 ——
        不再区分 INPUT / DISPLAY，也不问澄清问题。
        """
        return self.target is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_width": self.target_width,
            "target_height": self.target_height,
            "target": list(self.target) if self.target else None,
            "mode": self.mode,
            "tolerance": self.tolerance,
            "source": self.source,
            "status": self.status,
            "raw": self.raw,
        }


def parse_resolution(text: Any) -> Optional[ResolutionRequirement]:
    """从客户原话里解析分辨率需求（含 INPUT / DISPLAY / UNKNOWN 判定）。"""
    source = str(text or "")
    if not source.strip():
        return None

    target: Optional[Tuple[int, int]] = None
    match = _WXH_RE.search(source)
    if match:
        target = (int(match.group(1)), int(match.group(2)))
    if target is None:
        token = _TOKEN_RE.search(source)
        if token:
            target = RESOLUTION_PRESETS.get(token.group(1).lower())
    if target is None:
        return None

    if _INPUT_RE.search(source) and not _DISPLAY_RE.search(source):
        mode = INPUT
    elif _DISPLAY_RE.search(source):
        mode = DISPLAY
    else:
        mode = UNKNOWN   # "we need 4k" → 不猜，必要时澄清

    return ResolutionRequirement(
        target_width=target[0], target_height=target[1],
        mode=mode, source="customer", status="CONFIRMED",
        raw=source.strip()[:120],
    )


@dataclass
class ResolutionFitResult:
    """目标分辨率 vs 实际可拼接分辨率（计划第九节）。"""

    target: Optional[Tuple[int, int]] = None
    actual: Optional[Tuple[int, int]] = None
    horizontal_deviation: float = 0.0
    vertical_deviation: float = 0.0
    pixel_count_deviation: float = 0.0
    aspect_ratio_deviation: float = 0.0
    tolerance: float = RESOLUTION_FIT_NEAR
    fit_level: str = IMPOSSIBLE
    notes: list = field(default_factory=list)

    @property
    def acceptable(self) -> bool:
        return self.fit_level in (EXACT, MEETS_OR_EXCEEDS, NEAR_MATCH, ACCEPTABLE)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": list(self.target) if self.target else None,
            "actual": list(self.actual) if self.actual else None,
            "horizontal_deviation": round(self.horizontal_deviation, 4),
            "vertical_deviation": round(self.vertical_deviation, 4),
            "pixel_count_deviation": round(self.pixel_count_deviation, 4),
            "aspect_ratio_deviation": round(self.aspect_ratio_deviation, 4),
            "tolerance": round(self.tolerance, 4),
            "fit_level": self.fit_level,
            "notes": list(self.notes),
        }


def fit_resolution(
    target: Optional[Tuple[int, int]],
    actual: Optional[Tuple[int, int]],
    *,
    tolerance: Optional[float] = None,
    min_achievable_deviation: Optional[float] = None,
    aspect_tolerance: float = ASPECT_DEVIATION_ACCEPTABLE,
) -> ResolutionFitResult:
    """计算目标与实际拼接分辨率的偏差，并给出 EXACT / NEAR_MATCH / … 等级。

    不做 `actual == target` 的硬判断；容差优先级：
        1. 客户给的 tolerance
        2. 几何上"最小可做到"的偏差（min_achievable_deviation，例如半个模组）
        3. constants 里的默认阈值（后续用 Golden Dataset 调）
    """
    result = ResolutionFitResult(target=target, actual=actual)
    if not target or not actual:
        result.fit_level = IMPOSSIBLE
        result.notes.append("缺少目标分辨率或实际拼接分辨率")
        return result

    tw, th = int(target[0]), int(target[1])
    aw, ah = int(actual[0]), int(actual[1])
    if min(tw, th, aw, ah) <= 0:
        result.fit_level = IMPOSSIBLE
        result.notes.append("分辨率数值不合法")
        return result

    result.horizontal_deviation = abs(aw - tw) / tw
    result.vertical_deviation = abs(ah - th) / th
    result.pixel_count_deviation = abs(aw * ah - tw * th) / (tw * th)
    result.aspect_ratio_deviation = abs((aw / ah) - (tw / th)) / (tw / th)

    tol = float(tolerance) if tolerance else float(RESOLUTION_FIT_NEAR)
    if min_achievable_deviation:
        tol = max(tol, float(min_achievable_deviation))
    result.tolerance = tol

    worst = max(result.horizontal_deviation, result.vertical_deviation)
    if aw == tw and ah == th:
        result.fit_level = EXACT
    elif aw >= tw and ah >= th:
        # 客户口径：屏体贴出来的像素数够（甚至更多）就算达到目标分辨率 ——
        # 客户要的是"这块屏能到 4K"，比 4K 更高当然也算到。
        result.fit_level = MEETS_OR_EXCEEDS
    elif worst <= tol:
        result.fit_level = NEAR_MATCH
    elif worst <= RESOLUTION_FIT_ACCEPTABLE and result.aspect_ratio_deviation <= aspect_tolerance:
        result.fit_level = ACCEPTABLE
    elif worst <= RESOLUTION_FIT_NOT_ACCEPTABLE:
        result.fit_level = NOT_ACCEPTABLE
    else:
        result.fit_level = IMPOSSIBLE
    return result


__all__ = [
    "ACCEPTABLE",
    "DISPLAY",
    "EXACT",
    "IMPOSSIBLE",
    "INPUT",
    "MEETS_OR_EXCEEDS",
    "NEAR_MATCH",
    "NOT_ACCEPTABLE",
    "RESOLUTION_PRESETS",
    "ResolutionFitResult",
    "ResolutionRequirement",
    "UNKNOWN",
    "fit_resolution",
    "parse_resolution",
]
