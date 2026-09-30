"""LCD / IFP 的确定性选型（《LCD_IFP 需求链路工程化整改计划》Phase 6 / §十二）。

为什么要单独一条：LED 的 `RecommendationEngine` 是按点间距 / 箱体 / 亮度选型的，
对 LCD（英寸尺寸 + 分辨率 + 拼缝 + 拼接）完全不适用 —— 实测 2026-09-30：
LCD 需求齐了，Solution 侧仍然用 LED 的 Gate 反问 "Is it a permanent install,
or is it for rental/events?"，客户永远等不到推荐。

这里只做"从检索到的 LCD 候选里挑最合适的那一个"：不猜、不编，
打分依据全部来自产品数据与客户档案（尺寸 → 分辨率 → 拼缝 → 亮度）。
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _size_of(metadata: Dict[str, Any]) -> Optional[float]:
    raw = str(metadata.get("display_size_inch") or "").strip().strip('"')
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _bezel_of(metadata: Dict[str, Any]) -> Optional[float]:
    raw = str(metadata.get("bazel_mm") or "").replace("mm", "").strip()
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _resolution_of(metadata: Dict[str, Any]) -> str:
    text = str(metadata.get("resolution") or "").upper()
    if "3840" in text or "4K" in text:
        return "4K"
    if "1920" in text or "1080" in text or "2K" in text:
        return "2K"
    return ""


def _is_splicing_product(metadata: Dict[str, Any]) -> bool:
    """是不是"为拼接而设计"的视频墙面板。

    注意区分：普通商用显示器 `splicing_supported=True`（能拼，但拼缝明显），
    `is_splicing=False`；视频墙系列 `is_splicing=True`。客户要拼接墙时优先前者。
    """
    return bool(metadata.get("is_splicing"))


def score_lcd_candidate(profile: Any, metadata: Dict[str, Any]) -> Tuple[float, List[str]]:
    """给一个 LCD 候选打分（越大越合适）+ 命中理由。"""
    score = 0.0
    reasons: List[str] = []

    want_size = getattr(profile, "lcd_size_inch", None)
    size = _size_of(metadata)
    if want_size and size:
        diff = abs(float(want_size) - size)
        if diff < 0.5:
            score += 6.0
            reasons.append(f"{int(size)}-inch panels match your screen size")
        elif diff <= 3:
            score += 3.0
            reasons.append(f"{int(size)}-inch panels are the closest size available")
        else:
            score -= diff          # 尺寸差太多 → 扣分

    want_res = str(getattr(profile, "lcd_resolution", None) or "").upper()
    res = _resolution_of(metadata)
    if want_res and res:
        if want_res == res:
            score += 3.0
            reasons.append(f"{res} resolution as required")
        else:
            score -= 2.0

    if getattr(profile, "lcd_is_splicing", None) is True:
        if _is_splicing_product(metadata):
            score += 4.0
            bezel = _bezel_of(metadata)
            if bezel is not None:
                reasons.append(f"video-wall panels with {bezel:g}mm bezel")
        else:
            score -= 3.0
    elif getattr(profile, "lcd_is_splicing", None) is False:
        if _is_splicing_product(metadata):
            score -= 1.5
        else:
            score += 1.5
            reasons.append("designed as standalone displays")

    want_bezel = getattr(profile, "lcd_bezel_mm", None)
    bezel = _bezel_of(metadata)
    if want_bezel and bezel is not None:
        if bezel <= float(want_bezel) + 0.05:
            score += 2.0
            reasons.append(f"bezel {bezel:g}mm meets your seam requirement")
        else:
            score -= 1.5

    brightness = metadata.get("brightness_nit") or metadata.get("brightness_min_cd")
    if brightness:
        reasons.append(f"{int(brightness)}nit brightness")
    return score, reasons


def select_lcd_candidate(
    profile: Any,
    candidates: List[Any],
    *,
    to_dict=None,
) -> Optional[Tuple[Any, Dict[str, Any], List[str]]]:
    """从检索到的候选里挑一个 LCD 型号；挑不出返回 None。"""
    if not candidates:
        return None
    best: Optional[Tuple[float, Any, Dict[str, Any], List[str]]] = None
    for item in candidates:
        metadata = dict(getattr(item, "metadata", None) or (item.get("metadata") if isinstance(item, dict) else {}) or {})
        if str(metadata.get("display_type") or "").upper() not in ("LCD", "IFP"):
            continue
        score, reasons = score_lcd_candidate(profile, metadata)
        if best is None or score > best[0]:
            best = (score, item, metadata, reasons)
    if best is None:
        return None
    logger.info(
        "LCD/IFP select: model=%s score=%.1f reasons=%s",
        best[2].get("model"), best[0], best[3],
    )
    return best[1], best[2], best[3]


def layout_text(profile: Any) -> str:
    """拼接排布的确定性描述（客户给了排布就不再问，计划 §十）。"""
    layout = str(getattr(profile, "lcd_splicing_layout", None) or "").strip()
    count = getattr(profile, "lcd_screen_count", None)
    if not layout:
        return ""
    if count:
        return f"{layout} layout ({int(count)} panels)"
    match = re.match(r"^(\d+)\s*[x×*]\s*(\d+)$", layout)
    if match:
        return f"{layout} layout ({int(match.group(1)) * int(match.group(2))} panels)"
    return f"{layout} layout"


__all__ = ["layout_text", "score_lcd_candidate", "select_lcd_candidate"]
