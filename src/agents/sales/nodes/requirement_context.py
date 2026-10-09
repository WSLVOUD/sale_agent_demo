"""Scene-dependent requirement context helpers."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

LEGACY_TO_PROFILE_FIELDS: Dict[str, tuple[str, ...]] = {
    "display_type": ("display_type",),
    "size": ("target_width_m", "target_height_m", "screen_size_hint_mm"),
    "brightness": ("brightness_min_nit", "brightness_max_nit"),
}

SCENE_CATEGORIES = {
    "meeting": [
        "会议室", "会议", "教室", "培训", "教学", "学校", "课堂",
        "conference", "classroom", "office", "control_room", "hotel",
        "restaurant", "airport", "exhibition", "showroom", "museum", "hall",
        "retail", "hospital", "bank",
    ],
    "church": ["教堂", "礼拜", "宗教", "礼拜堂", "church"],
    "stage": ["舞台", "演唱会", "演出", "表演", "剧场", "stage", "concert"],
    "outdoor": ["室外", "户外", "露天", "外墙", "广场", "体育场", "advertising", "stadium"],
}

CONTEXT_DEPENDENT_FIELDS = {
    "meeting": ["display_type"],
    "church": ["display_type", "size"],
    "stage": ["display_type", "size"],
    "outdoor": ["brightness"],
}

FACT_FIELDS = (
    "display_type",
    "environment",
    "purpose",
    "installation",
    "viewing_distance_m",
    "target_width_m",
    "target_height_m",
    "pixel_pitch_mm",
    "brightness_min_nit",
    "brightness_max_nit",
    "budget_level",
)


def get_scene_category(usage: str) -> str:
    if not usage:
        return "unknown"
    for category, keywords in SCENE_CATEGORIES.items():
        if any(keyword in usage for keyword in keywords):
            return category
    return "unknown"


def should_clear_field_on_scene_change(
    old_category: str, new_category: str, field: str
) -> bool:
    if old_category == new_category:
        return False
    if new_category in ["church", "stage", "outdoor"] and field == "display_type":
        return True
    if old_category == "meeting" and new_category != "meeting" and field == "size":
        return True
    return False


def snapshot_facts(profile: Any) -> Dict[str, Any]:
    return {field: getattr(profile, field, None) for field in FACT_FIELDS}


def facts_added(before: Dict[str, Any], profile: Any) -> bool:
    for field in FACT_FIELDS:
        new_value = getattr(profile, field, None)
        if new_value in (None, "", [], {}):
            continue
        if new_value != before.get(field):
            return True
    return False


def clear_context_on_scene_change(profile: Any, previous_purpose: str) -> None:
    """Clear only the existing scene-dependent fields when the scene category changes."""
    old_category = get_scene_category(previous_purpose or "")
    new_category = get_scene_category(profile.purpose or "")
    if (
        not previous_purpose
        or not profile.purpose
        or old_category == new_category
        or old_category == "unknown"
    ):
        return

    fields_to_clear: List[str] = []
    for fields in CONTEXT_DEPENDENT_FIELDS.values():
        if should_clear_field_on_scene_change(
            old_category, new_category, fields[0] if fields else ""
        ):
            fields_to_clear.extend(fields)
    for legacy_field in set(fields_to_clear):
        for profile_field in LEGACY_TO_PROFILE_FIELDS.get(legacy_field, ()):
            if getattr(profile, profile_field, None) is not None:
                logger.info(
                    "Clearing stale profile field '%s' (scene %s → %s)",
                    profile_field,
                    old_category,
                    new_category,
                )
                setattr(profile, profile_field, None)
                profile.sources.pop(profile_field, None)


_LEGACY_TO_PROFILE_FIELDS = LEGACY_TO_PROFILE_FIELDS
_get_scene_category = get_scene_category
_should_clear_field_on_scene_change = should_clear_field_on_scene_change
_FACT_FIELDS = FACT_FIELDS
_snapshot_facts = snapshot_facts
_facts_added = facts_added

__all__ = [
    "clear_context_on_scene_change",
    "CONTEXT_DEPENDENT_FIELDS",
    "FACT_FIELDS",
    "LEGACY_TO_PROFILE_FIELDS",
    "SCENE_CATEGORIES",
    "_FACT_FIELDS",
    "_LEGACY_TO_PROFILE_FIELDS",
    "_facts_added",
    "_get_scene_category",
    "_should_clear_field_on_scene_change",
    "_snapshot_facts",
]
