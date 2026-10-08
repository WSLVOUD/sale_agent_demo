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
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    exclude: Iterable[str] = (),
) -> Optional[Tuple[Any, Dict[str, Any], List[str]]]:
    """从检索到的候选里挑一个 LCD 型号；挑不出返回 None。

    ``exclude``：已经给客户看过的型号 —— 客户问"还有其他推荐吗"时要**换一个**
    （客户口径 2026-10）。排除后如果没有候选了，就退回不排除（宁可重复给一次，
    也不能因为"都推荐过"而给不出任何型号）。
    """
    if not candidates:
        return None

    def _model_of(metadata: Dict[str, Any]) -> str:
        return str(metadata.get("model") or metadata.get("product_id") or "").strip()

    def _best(skip: set) -> Optional[Tuple[float, Any, Dict[str, Any], List[str]]]:
        found = None
        for item in candidates:
            metadata = dict(
                getattr(item, "metadata", None)
                or (item.get("metadata") if isinstance(item, dict) else {})
                or {}
            )
            if str(metadata.get("display_type") or "").upper() not in ("LCD", "IFP"):
                continue
            if skip and _model_of(metadata) in skip:
                continue
            score, reasons = score_lcd_candidate(profile, metadata)
            if found is None or score > found[0]:
                found = (score, item, metadata, reasons)
        return found

    excluded = {str(name).strip() for name in (exclude or ()) if str(name).strip()}
    best = _best(excluded) if excluded else None
    if best is None:
        best = _best(set())
    if best is None:
        return None
    logger.info(
        "LCD/IFP select: model=%s score=%.1f reasons=%s%s",
        best[2].get("model"), best[0], best[3],
        f" (excluded {sorted(excluded)})" if excluded else "",
    )
    return best[1], best[2], best[3]


def layout_text(profile: Any) -> str:
    """拼接排布的确定性描述（客户给了排布就不再问，计划 §十）。

    客户口径（2026-10）：**只有"客户要拼接屏"时才允许出现排布 / 箱体数**。
    IFP（会议平板）根本无法拼接，任何情况下都不许出现"3x3 / 9 panels"这种话。

    这里用 ``lcd_is_splicing`` 作为唯一前置条件（而不是去 import dialogue 层的
    ``is_ifp_requirement`` —— 架构护栏规定 rag 不能伸手进 dialogue / agents）。
    不是拼接需求 → 一律返回空。
    """
    if not bool(getattr(profile, "lcd_is_splicing", None)):
        return ""
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
