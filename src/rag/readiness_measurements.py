"""Pure formatting helpers for size hints shown by readiness questions."""
from __future__ import annotations

from typing import Any, Optional


def format_measurement(mm: Optional[float]) -> str:
    """Format a millimetre clue as a customer-facing measurement."""
    try:
        value = float(mm or 0)
    except (TypeError, ValueError):
        return ""
    if value <= 0:
        return ""
    if value >= 100 and abs(value % 10) < 1e-6:
        return f"{value / 10:g} cm"
    if value >= 100:
        return f"{value / 10:.1f} cm".replace(".0 cm", " cm")
    return f"{value:g} mm"


def size_hint_sentence(profile: Any, language: str = "en") -> str:
    """Turn an image-estimated size into a non-binding customer-facing hint."""
    hint = list(getattr(profile, "vision_size_hint_mm", None) or [])
    if len(hint) < 2:
        return ""
    try:
        width_m = float(hint[0]) / 1000
        height_m = float(hint[1]) / 1000
    except (TypeError, ValueError):
        return ""
    if width_m <= 0 or height_m <= 0:
        return ""
    if language == "zh":
        return f"图片上看大约是 {width_m:g} 米 × {height_m:g} 米。"
    return f"The image suggests roughly {width_m:g}m x {height_m:g}m."


__all__ = ["format_measurement", "size_hint_sentence"]
