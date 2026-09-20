"""
备选型号的"差别说明"（客户口径：备选要说成条件句，不罗列、不提价格）。

    "If you want <差别>, <型号>."

统一放在这里，供快路径（fast_path）与确定性推荐表达（recommend）共用。
"""
from __future__ import annotations

from typing import Any, Dict


def alternative_difference(top: Dict[str, Any], other: Dict[str, Any]) -> str:
    """备选款与首选款的差别（说成"你要什么就选它"，不提价格）。"""
    bits = []
    top_pitch = float(top.get("pixel_pitch_mm") or 0)
    other_pitch = float(other.get("pixel_pitch_mm") or 0)
    if top_pitch and other_pitch and abs(other_pitch - top_pitch) > 0.01:
        bits.append(
            f"a finer {other_pitch:g}mm pitch" if other_pitch < top_pitch
            else f"a wider {other_pitch:g}mm pitch"
        )

    top_brightness = int(top.get("brightness_nit") or 0)
    other_brightness = int(other.get("brightness_nit") or 0)
    if other_brightness and top_brightness and other_brightness > top_brightness:
        bits.append(f"higher brightness ({other_brightness}nit vs {top_brightness}nit)")
    elif other_brightness and top_brightness and other_brightness < top_brightness:
        bits.append(f"a lower brightness option ({other_brightness}nit)")

    top_features = set(top.get("features") or [])
    other_features = set(other.get("features") or [])
    for token, phrase in (
        ("cob", "COB packaging"),
        ("hdr", "HDR"),
        ("waterproof", "waterproofing"),
        ("gob", "GOB protection"),
        ("flexible", "a flexible / curved build"),
    ):
        if token in other_features and token not in top_features:
            bits.append(phrase)

    # 质保不主动提（客户口径）：不再用"更长质保"作为备选差异点
    if other.get("installation") and other.get("installation") != top.get("installation"):
        bits.append(f"a {other['installation']} version")
    if not bits:
        bits.append("a different cabinet / pitch combination")
    return ", ".join(bits[:2])


__all__ = ["alternative_difference"]
