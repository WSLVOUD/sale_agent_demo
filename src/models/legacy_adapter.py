"""
M1/M7：RequirementProfile 的**只读兼容层**（Legacy Adapter）。

规则（来自《三套需求状态统一改造计划 v2.0》第十二/十六条）：

    RequirementProfile
           ↓
      Legacy Adapter
           ↓
      旧模块（sales requirements / solution requirement）

旧字段只允许由 Profile 生成，**绝不允许反过来修改 Profile**，
这样 `requirements` 与 `requirement` 都只是 Profile 的投影，不会各自漂移。
"""
from __future__ import annotations

from typing import Any, Dict

# Adapter 负责的键（重建视图时先清掉这些键，防止旧值残留）
LEGACY_ADAPTER_KEYS: tuple[str, ...] = (
    "display_type", "usage", "purpose", "location_type", "indoor", "outdoor",
    "content_type", "price_preference",
    "is_rental", "distance", "viewing_distance", "viewing_distance_m", "size",
    "pixel_pitch", "brightness_min", "screen_size_hint_mm", "size_axis",
    "model", "series_id",
)


def profile_to_legacy(profile: Any) -> Dict[str, Any]:
    """Profile → Sales Agent 的旧 requirements 字典（只读兼容层）。"""
    if profile is None:
        return {}
    legacy: Dict[str, Any] = {}

    if getattr(profile, "display_type", None):
        legacy["display_type"] = profile.display_type
    if getattr(profile, "purpose", None):
        legacy["usage"] = profile.purpose
        legacy["purpose"] = profile.purpose
    # 客户口径：内容类型（视频/图片/两者都有）与"价格/质量取向"也要跨轮传递
    if getattr(profile, "content_type", None):
        legacy["content_type"] = profile.content_type
    if getattr(profile, "price_preference", None):
        legacy["price_preference"] = profile.price_preference
    environment = getattr(profile, "environment", None)
    if environment:
        legacy["location_type"] = "室外" if environment in ("outdoor", "semi_outdoor") else "室内"
        legacy["indoor"] = environment == "indoor"
        legacy["outdoor"] = environment in ("outdoor", "semi_outdoor")
    if getattr(profile, "installation", None):
        legacy["is_rental"] = profile.installation == "rental"
    if getattr(profile, "viewing_distance_m", None) is not None:
        legacy["distance"] = f"{profile.viewing_distance_m:g}米"
        legacy["viewing_distance"] = legacy["distance"]
    width = getattr(profile, "target_width_m", None)
    height = getattr(profile, "target_height_m", None)
    if width and height:
        legacy["size"] = f"{width:g}米x{height:g}米"
    elif width:
        legacy["size"] = f"{width:g}米宽"
    elif height:
        legacy["size"] = f"{height:g}米高"
    if getattr(profile, "pixel_pitch_mm", None) is not None:
        legacy["pixel_pitch"] = profile.pixel_pitch_mm
    if getattr(profile, "brightness_min_nit", None) is not None:
        legacy["brightness_min"] = profile.brightness_min_nit
    # 尺寸线索（"129,2cm"这类）与方向必须跨轮传递
    if getattr(profile, "screen_size_hint_mm", None) is not None:
        legacy["screen_size_hint_mm"] = profile.screen_size_hint_mm
    if getattr(profile, "size_axis", None):
        legacy["size_axis"] = profile.size_axis
    if getattr(profile, "model", None):
        legacy["model"] = profile.model
    if getattr(profile, "series_id", None):
        legacy["series_id"] = profile.series_id
    return legacy


def profile_to_solution_requirement(profile: Any) -> Dict[str, Any]:
    """Profile → Solution Agent 的旧 requirement 字典（只读兼容层）。

    Solution 的 retrieve / recommend / reflection 仍读这套键，
    但值必须来自 Sales 的 Profile，而不是 Solution 自己再解析一遍对话。
    """
    if profile is None:
        return {}
    requirement: Dict[str, Any] = {}
    environment = getattr(profile, "environment", None)
    if environment:
        requirement["indoor"] = environment == "indoor"
        requirement["outdoor"] = environment in ("outdoor", "semi_outdoor")
    if getattr(profile, "viewing_distance_m", None) is not None:
        requirement["distance"] = f"{profile.viewing_distance_m:g}米"
    if getattr(profile, "purpose", None):
        requirement["purpose"] = profile.purpose
    width = getattr(profile, "target_width_m", None)
    height = getattr(profile, "target_height_m", None)
    if width and height:
        requirement["size"] = f"{width:g}米x{height:g}米"
    elif width:
        requirement["size"] = f"{width:g}米宽"
    elif height:
        requirement["size"] = f"{height:g}米高"
    if getattr(profile, "display_type", None):
        requirement["display_type"] = profile.display_type
    if getattr(profile, "brightness_min_nit", None) is not None:
        requirement["brightness"] = f"{profile.brightness_min_nit}nit"
    if getattr(profile, "installation", None):
        requirement["is_rental"] = profile.installation == "rental"
    if getattr(profile, "pixel_pitch_mm", None) is not None:
        requirement["pixel_pitch"] = profile.pixel_pitch_mm
    if getattr(profile, "model", None):
        requirement["model"] = profile.model
    if getattr(profile, "series_id", None):
        requirement["series_id"] = profile.series_id
    return requirement


def rebuild_legacy_view(target: Dict[str, Any], profile: Any) -> Dict[str, Any]:
    """把 Profile 投影覆盖到 target（旧字典）上：先清掉 Adapter 负责的键，再写入。"""
    if not isinstance(target, dict):
        target = {}
    for key in LEGACY_ADAPTER_KEYS:
        target.pop(key, None)
    target.update(profile_to_legacy(profile))
    return target


__all__ = [
    "LEGACY_ADAPTER_KEYS",
    "profile_to_legacy",
    "profile_to_solution_requirement",
    "rebuild_legacy_view",
]
