"""Slot merging and normalized retrieval-query construction."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


def merge_slots(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Merge turn slots while maintaining inferred/default source markers."""
    markers = ("_inferred_slots", "_scenario_derived", "_default_slots")
    merged = dict(base or {})
    incoming_marker: Dict[str, str] = {}
    for marker in markers:
        for key in incoming.get(marker) or []:
            incoming_marker[str(key)] = marker

    for key, value in incoming.items():
        if key in markers:
            continue
        merged[key] = value
        source_marker = incoming_marker.get(key)
        for marker in markers:
            current = merged.get(marker)
            if marker == source_marker:
                if current is None:
                    merged[marker] = [key]
                elif key not in current:
                    current.append(key)
            elif current and key in current:
                current.remove(key)

    for marker in markers:
        if not merged.get(marker):
            merged.pop(marker, None)
    return merged


def build_retrieval_query(
    slots: Dict[str, Any],
    fallback: str = "",
    *,
    purpose_english: Callable[[Optional[str]], str],
) -> str:
    """Build the normalized English retrieval query from structured slots."""
    if not slots:
        return fallback or "LED display product"

    parts: List[str] = []
    environment = slots.get("environment")
    if environment == "indoor":
        parts.append("indoor")
    elif environment == "outdoor":
        parts.append("outdoor")
    elif environment == "semi_outdoor":
        parts.append("semi outdoor")

    installation = slots.get("installation")
    if installation == "rental":
        parts.append("rental")
    elif installation == "fixed":
        parts.append("fixed installation")

    display_type = slots.get("display_type")
    parts.append(f"{display_type} display" if display_type else "display")

    english_purpose = purpose_english(slots.get("purpose"))
    if english_purpose:
        parts.append(english_purpose)

    if slots.get("viewing_distance_m") is not None:
        parts.append(f"viewing distance {slots['viewing_distance_m']:g}m")
    if slots.get("pixel_pitch_mm") is not None:
        parts.append(f"pixel pitch {slots['pixel_pitch_mm']:g}mm")
    if slots.get("brightness_min") is not None:
        parts.append(f"brightness above {slots['brightness_min']}nit")
    if slots.get("waterproof"):
        parts.append("waterproof IP65")
    if slots.get("cob"):
        parts.append("COB")
    if slots.get("hdr"):
        parts.append("HDR")
    if slots.get("interaction"):
        parts.append("touch whiteboard interactive")
    if slots.get("model"):
        parts.append(str(slots["model"]))
    elif slots.get("series_id"):
        parts.append(str(slots["series_id"]))
    if slots.get("budget_level"):
        parts.append({"low": "low price", "mid": "mid price", "high": "premium"}[slots["budget_level"]])

    return " ".join(part for part in parts if part)


__all__ = ["build_retrieval_query", "merge_slots"]
