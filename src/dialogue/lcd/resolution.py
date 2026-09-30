"""LCD 分辨率规则（纯 helper，不做决策）。

    · 客户明确说 4K / 2K / 1080P / 1920x1080 / 3840x2160 → 用客户的
    · 否则按尺寸规则：<65" → 2K，>65" → 4K
    · 恰好 65"：产品库只有一种规格就用它；2K/4K 都有 → 默认 4K（不再反问客户）

"什么时候用这个默认值"仍然由 lcd_decision 决定。
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


def resolution_for_size(
    size_inch: Optional[float],
    *,
    customer_resolution: str = "",
    catalog_resolutions: Optional[Iterable[str]] = None,
) -> Tuple[str, str]:
    """返回 ``(resolution, source)``。

    优先级：客户明确 > 尺寸规则（<65"→2K，>65"→4K）> 产品库实际规格（65" 边界）。

    客户口径（2026-09-30）：**分辨率不许反问客户**。"客户不说，就按大于 65 选 4K、
    小于 65 选 2K / 1080p"。恰好 65" 时先看产品库：只有一种规格就用它；
    两种都有（现在就是 2K/4K 都有）就按"大屏走 4K"的口径默认 4K，不再问客户。
    """
    explicit = _normalize_resolution(customer_resolution)
    if explicit:
        return explicit, "customer_explicit"
    if size_inch is None:
        return "", "unknown"
    size = float(size_inch)
    if size < 65:
        return "2K", "size_rule_lt_65"
    if size > 65:
        return "4K", "size_rule_gt_65"
    catalog = {_normalize_resolution(item) for item in (catalog_resolutions or [])}
    catalog.discard("")
    if len(catalog) == 1:
        return catalog.pop(), "product_catalog"
    return "4K", "size_default_4k_65"


def _normalize_resolution(value: Any) -> str:
    text = str(value or "").strip().upper().replace(" ", "")
    if not text:
        return ""
    if text in ("2K", "1080P", "FHD", "1920X1080", "1920*1080", "FULLHD"):
        return "2K"
    if text in ("4K", "2160P", "UHD", "3840X2160", "3840*2160", "4KUHD"):
        return "4K"
    return ""


def catalog_resolutions_for_size(size_inch: float, data_dir: str = "") -> List[str]:
    """产品库里该尺寸档已有的分辨率（65" 边界用；查不到就返回空）。"""
    try:
        import json
        import os

        from src.config import config

        path = os.path.join(data_dir or config.DATA_DIR, "lcd_products.json")
        with open(path, encoding="utf-8") as handle:
            products = json.load(handle)["products"]
    except Exception:  # pragma: no cover - 防御式
        return []
    found: List[str] = []
    for item in products:
        try:
            if abs(float(str(item.get("display_size_inch", "")).strip('"')) - float(size_inch)) > 0.01:
                continue
        except (TypeError, ValueError):
            continue
        for token in str(item.get("resolution") or "").replace("/", " ").split():
            normalized = _normalize_resolution(token)
            if normalized and normalized not in found:
                found.append(normalized)
    return found


__all__ = [
    "catalog_resolutions_for_size",
    "resolution_for_size",
]
