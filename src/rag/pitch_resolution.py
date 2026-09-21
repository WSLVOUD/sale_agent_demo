"""v2.5+++（对话决策与输出链路优化 · Phase 3）：Pitch Resolution。

计划 §4：客户说 P3、系统推 P2.9 时，必须能解释"requested pitch"与"resolved pitch"
的关系 —— 不能一句 "P3 it is." 之后无声地给出 P2.9。

责任边界（不变）：点间距由 Python 决定（hard filter → 有效窗口 → 打分 → 选中型号），
LLM 只负责把这个**已经定好的结果**解释清楚。

匹配类型：
    EXACT                 目录里就有这个点间距
    NEAREST_AVAILABLE     目录没有完全一致的，选中的是最接近的那一档
    WITHIN_VALID_WINDOW   客户没点名字间距，选中的落在"环境+视距"的有效窗口内
    OUTSIDE_VALID_WINDOW  选中的偏离了客户要求 / 有效窗口（例如被其他硬条件挤走）
    UNKNOWN               没有足够信息判断
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import re

EXACT = "EXACT"
NEAREST_AVAILABLE = "NEAREST_AVAILABLE"
WITHIN_VALID_WINDOW = "WITHIN_VALID_WINDOW"
OUTSIDE_VALID_WINDOW = "OUTSIDE_VALID_WINDOW"
UNKNOWN = "UNKNOWN"

ALL_MATCH_TYPES = (
    EXACT,
    NEAREST_AVAILABLE,
    WITHIN_VALID_WINDOW,
    OUTSIDE_VALID_WINDOW,
    UNKNOWN,
)

# 客户看到的点间距标签（型号名里的 P3 / P2.9）；数值只用于"谁更接近"
EXACT_TOLERANCE_MM = 0.06

_MODEL_PITCH_RE = re.compile(r"-P(\d+(?:\.\d+)?)", re.IGNORECASE)


def pitch_label(value: Optional[float]) -> str:
    """数值 → 客户看得懂的标签：3.0 → P3，2.976 → P2.98（型号名会覆盖它）。"""
    if not value:
        return ""
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return f"P{text}"


def model_pitch_label(model: Any) -> str:
    """型号名里的点间距标签（客户实际看到的就是它：TW11-IR-P2.9 → P2.9）。"""
    match = _MODEL_PITCH_RE.search(str(getattr(model, "model", "") or ""))
    if match:
        value = match.group(1).rstrip("0").rstrip(".")
        return f"P{value}" if value else ""
    return pitch_label(getattr(model, "pixel_pitch_mm", None))


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return f"P{text}"


@dataclass
class PitchResolution:
    """requested pitch → resolved pitch 的显式说明。"""

    requested_pitch: Optional[float] = None
    resolved_pitch: Optional[float] = None
    match_type: str = UNKNOWN
    reason: str = ""
    band_min: Optional[float] = None
    band_max: Optional[float] = None
    nearest_available: Optional[float] = None
    requested_label: str = ""
    resolved_label: str = ""

    @property
    def requested_text(self) -> str:
        return self.requested_label or _fmt(self.requested_pitch)

    @property
    def resolved_text(self) -> str:
        return self.resolved_label or _fmt(self.resolved_pitch)

    @property
    def needs_explanation(self) -> bool:
        """是否需要向客户解释（requested 与 resolved 不一致时必须解释）。"""
        if self.requested_pitch is None or self.resolved_pitch is None:
            return False
        if self.match_type == EXACT:
            return False
        if self.requested_label and self.resolved_label:
            return self.requested_label.upper() != self.resolved_label.upper()
        return abs(float(self.requested_pitch) - float(self.resolved_pitch)) > EXACT_TOLERANCE_MM

    def explain(self) -> str:
        """一句可以直接说给客户的话（措辞交给 LLM，事实来自这里）。"""
        if not self.needs_explanation:
            return ""
        requested = self.requested_text or "that pitch"
        resolved = self.resolved_text or "the closest one"
        if self.match_type == NEAREST_AVAILABLE:
            return (
                f"{requested} isn't an exact option in this range — "
                f"{resolved} is the closest available match."
            )
        if self.match_type == OUTSIDE_VALID_WINDOW:
            return (
                f"{requested} isn't available for this configuration, so we'd go with "
                f"{resolved} instead."
            )
        return f"We'd use {resolved} rather than {requested} for this configuration."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested_pitch": self.requested_pitch,
            "resolved_pitch": self.resolved_pitch,
            "requested_pitch_text": self.requested_text,
            "resolved_pitch_text": self.resolved_text,
            "requested_label": self.requested_label,
            "resolved_label": self.resolved_label,
            "pitch_match_type": self.match_type,
            "pitch_resolution_reason": self.reason,
            "needs_explanation": self.needs_explanation,
            "band_min_mm": self.band_min,
            "band_max_mm": self.band_max,
            "nearest_available_mm": self.nearest_available,
        }


def _nearest(pitches: Iterable[float], target: float) -> Optional[float]:
    values = [float(item) for item in pitches if item]
    if not values:
        return None
    return min(values, key=lambda value: abs(value - float(target)))


def resolve_pitch(
    profile: Any,
    model: Any,
    *,
    available_pitches: Optional[Iterable[float]] = None,
    band_min: Optional[float] = None,
    band_max: Optional[float] = None,
) -> PitchResolution:
    """算出这个型号的 requested / resolved 关系（纯计算，无 LLM）。"""
    resolved = getattr(model, "pixel_pitch_mm", None)
    resolved = float(resolved) if resolved else None
    requested = getattr(profile, "pixel_pitch_mm", None)
    requested = float(requested) if requested else None
    # 客户看到的标签：requested 用客户给的值；resolved 优先用**型号名里的标签**
    requested_label = pitch_label(requested) if requested else ""
    resolved_label = model_pitch_label(model) if resolved else ""

    if band_min is None or band_max is None:
        # 允许调用方只传 technical 字典
        technical = getattr(profile, "_technical", None)
        if isinstance(technical, dict):
            band_min = band_min if band_min is not None else technical.get("pixel_pitch_min_mm")
            band_max = band_max if band_max is not None else technical.get("pixel_pitch_max_mm")

    if resolved is None:
        return PitchResolution(
            requested_pitch=requested, resolved_pitch=None, match_type=UNKNOWN,
            reason="型号没有点间距信息", band_min=band_min, band_max=band_max,
            requested_label=requested_label, resolved_label="",
        )

    if requested is None:
        within = (
            band_min is not None
            and band_max is not None
            and float(band_min) - 1e-6 <= resolved <= float(band_max) + 1e-6
        )
        if band_min is None and band_max is None:
            return PitchResolution(
                requested_pitch=None, resolved_pitch=resolved, match_type=UNKNOWN,
                reason="客户没有指定点间距，也没有可用窗口",
                band_min=band_min, band_max=band_max,
                resolved_label=resolved_label,
            )
        return PitchResolution(
            requested_pitch=None, resolved_pitch=resolved,
            match_type=WITHIN_VALID_WINDOW if within else OUTSIDE_VALID_WINDOW,
            reason=(
                "客户没有指定点间距；该值落在环境+视距的有效窗口内"
                if within
                else "客户没有指定点间距；该值超出环境+视距的有效窗口"
            ),
            band_min=band_min, band_max=band_max,
            resolved_label=resolved_label,
        )

    # 客户看的标签一样（P3 要 → 型号名也是 P3）→ 完全匹配
    resolved_from_name = bool(_MODEL_PITCH_RE.search(str(getattr(model, "model", "") or "")))
    if resolved_from_name:
        labels_match = bool(
            requested_label and resolved_label
            and requested_label.upper() == resolved_label.upper()
        )
    else:
        # 型号名里没有点间距标签（非常规型号）→ 用数值容差兜底
        labels_match = abs(requested - resolved) <= EXACT_TOLERANCE_MM
    if labels_match:
        return PitchResolution(
            requested_pitch=requested, resolved_pitch=resolved, match_type=EXACT,
            reason="目录里就是这个点间距", band_min=band_min, band_max=band_max,
            nearest_available=resolved, requested_label=requested_label,
            resolved_label=resolved_label,
        )

    nearest = _nearest(available_pitches or [], requested)
    if nearest is not None and abs(nearest - resolved) <= EXACT_TOLERANCE_MM:
        return PitchResolution(
            requested_pitch=requested, resolved_pitch=resolved,
            match_type=NEAREST_AVAILABLE,
            reason=(
                f"{_fmt(requested)} 不是目录里的现成点间距，"
                f"{_fmt(resolved)} 是最接近的一档"
            ),
            band_min=band_min, band_max=band_max, nearest_available=nearest,
            requested_label=requested_label, resolved_label=resolved_label,
        )

    within = (
        band_min is not None
        and band_max is not None
        and float(band_min) - 1e-6 <= resolved <= float(band_max) + 1e-6
    )
    return PitchResolution(
        requested_pitch=requested, resolved_pitch=resolved,
        match_type=WITHIN_VALID_WINDOW if within else OUTSIDE_VALID_WINDOW,
        reason=(
            f"客户要 {_fmt(requested)}，但该配置下选了 {_fmt(resolved)}"
            + ("（落在有效窗口内）" if within else "（超出有效窗口，需向客户解释）")
        ),
        band_min=band_min, band_max=band_max, nearest_available=nearest,
        requested_label=requested_label, resolved_label=resolved_label,
    )


__all__ = [
    "ALL_MATCH_TYPES",
    "EXACT",
    "EXACT_TOLERANCE_MM",
    "NEAREST_AVAILABLE",
    "OUTSIDE_VALID_WINDOW",
    "PitchResolution",
    "UNKNOWN",
    "WITHIN_VALID_WINDOW",
    "model_pitch_label",
    "pitch_label",
    "resolve_pitch",
]
