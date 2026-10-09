from src.agents.sales.nodes.requirement_context import clear_context_on_scene_change
from src.models.requirement import RequirementProfile


def test_scene_change_clears_meeting_specific_size_and_type_facts():
    profile = RequirementProfile(
        purpose="教堂",
        display_type="LED",
        target_width_m=3.0,
        target_height_m=2.0,
        screen_size_hint_mm=1000,
        viewing_distance_m=5.0,
        sources={
            "display_type": "customer_explicit",
            "target_width_m": "explicit",
            "target_height_m": "explicit",
            "screen_size_hint_mm": "explicit",
            "viewing_distance_m": "explicit",
        },
    )

    clear_context_on_scene_change(profile, "会议室")

    assert profile.display_type is None
    assert profile.target_width_m is None
    assert profile.target_height_m is None
    assert profile.screen_size_hint_mm is None
    assert profile.viewing_distance_m == 5.0
    assert "display_type" not in profile.sources
    assert "target_width_m" not in profile.sources
    assert "target_height_m" not in profile.sources
    assert "screen_size_hint_mm" not in profile.sources
    assert profile.sources["viewing_distance_m"] == "explicit"


def test_scene_change_preserves_facts_when_existing_rules_do_not_clear_them():
    profile = RequirementProfile(
        purpose="会议室",
        display_type="LED",
        target_width_m=3.0,
        sources={
            "display_type": "customer_explicit",
            "target_width_m": "explicit",
        },
    )

    clear_context_on_scene_change(profile, "教堂")

    assert profile.display_type == "LED"
    assert profile.target_width_m == 3.0
    assert profile.sources == {
        "display_type": "customer_explicit",
        "target_width_m": "explicit",
    }
