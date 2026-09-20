"""v2.3 §4 / §15：产品约束的统一派生（亮度、安装方式等）。

约束只在这里派生一次；推荐引擎与硬过滤都从这里取，避免同一条业务规则在多个
模块里各写一遍。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .constants import BRIGHTNESS_BY_ENVIRONMENT


def brightness_range_for_environment(
    environment: Optional[str],
) -> Tuple[Optional[int], Optional[int]]:
    """使用环境 → 亮度区间（nit）。"""
    if not environment:
        return None, None
    return BRIGHTNESS_BY_ENVIRONMENT.get(str(environment), (None, None))


def rental_from_facts(facts: Dict[str, Any]) -> Optional[bool]:
    """从事实里取固装 / 租赁（没有信息时返回 None，由调用方决定默认值）。"""
    facts = dict(facts or {})
    installation = facts.get("installation")
    if installation in ("fixed", "rental"):
        return installation == "rental"
    if facts.get("is_rental") is not None:
        return bool(facts.get("is_rental"))
    return None


__all__ = ["brightness_range_for_environment", "rental_from_facts"]
