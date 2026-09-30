"""IFP 判定（纯 helper）—— 整改计划 §八：IFP 判断必须统一。

最终只能由 LCD Decision Center 用这里的口径判 IFP：
会议 / 教育 + 客户要手写 / 白板（或要触控）→ IFP；会议室本身**不等于** IFP。

这里只回答"客户要的是不是交互平板"，不决定"下一步问什么"。
"""
from __future__ import annotations

import logging
from typing import Any

from .parsing import CONFERENCE_EDUCATION

logger = logging.getLogger(__name__)


def is_ifp_requirement(profile: Any) -> bool:
    """客户要的是不是**交互平板（IFP）**（计划 §十四/§十五）。

    客户口径（2026-09-30）："客户要求很明显是可手写的会议室使用的 IFP，
    为什么会推荐普通的可触摸的 LCD？" —— 所以判定口径是：

        · 产品类型已经是 IFP；
        · 或者客户要**手写 / 白板**（这就是交互平板，不是普通 LCD）；
        · 或者会议 / 教育场景下客户要**触控**。

    判定为真以后，检索与选型都要按 IFP 来（否则会从 LCD 语料里挑出一台
    普通的商用显示器 —— 实测就是这样把 P65 推给了要手写白板的会议室）。
    """
    display_type = str(getattr(profile, "display_type", None) or "").upper()
    if display_type == "IFP":
        return True
    if getattr(profile, "lcd_handwriting_required", None) is True:
        return True
    if (
        getattr(profile, "lcd_touch_required", None) is True
        and str(getattr(profile, "lcd_category", "") or "") == CONFERENCE_EDUCATION
    ):
        return True
    return False


def effective_display_type(profile: Any) -> str:
    """检索 / 选型实际该按哪个产品类型走（LED / LCD / IFP）。

    客户明说 LCD、但要手写白板时，档案里的 display_type 仍是 LCD（那是客户原话），
    可**选型口径**必须是 IFP —— 这里给出"实际该查哪一类产品"。
    """
    display_type = str(getattr(profile, "display_type", None) or "").upper()
    if display_type == "LED":
        return "LED"
    if is_ifp_requirement(profile):
        return "IFP"
    return display_type


def _is_ifp_branch(profile: Any) -> bool:
    """会议 / 教育 + 需要手写或互动 → IFP 分支（计划 §十五/§十六）。"""
    return is_ifp_requirement(profile)


__all__ = [
    "effective_display_type",
    "is_ifp_requirement",
]
