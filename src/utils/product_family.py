"""从需求档案判断"这一轮属于哪条产品链路"，给兜底话术选口径用。

客户口径（2026-09-30）：LCD 会话里绝不能出现 LED 口径的兜底话术
（"if one of the requirements can be relaxed, for example the pixel pitch…"），
反过来也一样。所以凡是"面向客户但内容通用"的兜底句，都要先问一下这里是哪条链路。
"""
from __future__ import annotations

from typing import Any


def product_family_of(profile: Any) -> str:
    """返回 ``"led"`` / ``"lcd"`` / ``"ifp"``；判断不了返回空串。"""
    if profile is None:
        return ""
    if isinstance(profile, dict):
        display_type = str(profile.get("display_type") or "").upper()
        handwriting = profile.get("lcd_handwriting_required")
        touch = profile.get("lcd_touch_required")
        category = str(profile.get("lcd_category") or "")
    else:
        display_type = str(getattr(profile, "display_type", "") or "").upper()
        handwriting = getattr(profile, "lcd_handwriting_required", None)
        touch = getattr(profile, "lcd_touch_required", None)
        category = str(getattr(profile, "lcd_category", "") or "")
    if display_type == "LED":
        return "led"
    if display_type in ("LCD", "IFP"):
        try:
            from src.dialogue.lcd_decision import is_ifp_requirement

            if display_type == "IFP" or is_ifp_requirement(profile):
                return "ifp"
        except Exception:  # pragma: no cover - 防御式
            if display_type == "IFP":
                return "ifp"
        return "lcd"
    # 类型还没定：有 LCD 侧的手写/触控痕迹就当 LCD 口径（宁可说 LCD，也不要说点间距）
    if handwriting is True or (touch is True and category):
        return "ifp" if handwriting is True else "lcd"
    return ""


__all__ = ["product_family_of"]
