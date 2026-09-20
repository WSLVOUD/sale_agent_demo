"""v2.5 Phase 3：Resolution Requirement（1080P/2K/1440P/4K/8K/WxH + INPUT/DISPLAY/UNKNOWN）。"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.engineering import parse_resolution  # noqa: E402
from src.engineering.resolution import DISPLAY, INPUT, UNKNOWN  # noqa: E402


class TestPresets:

    @pytest.mark.parametrize("text,expected", [
        ("we need 1080p", (1920, 1080)),
        ("2k is enough", (2048, 1080)),
        ("shoot for 1440p", (2560, 1440)),
        ("a 4K screen please", (3840, 2160)),
        ("8K resolution", (7680, 4320)),
        ("3840x2160", (3840, 2160)),
        ("3840 × 2160 pixels", (3840, 2160)),
        ("custom 3200*1800", (3200, 1800)),
    ])
    def test_parse(self, text, expected):
        requirement = parse_resolution(text)
        assert requirement is not None and requirement.target == expected

    def test_no_resolution_returns_none(self):
        assert parse_resolution("we need an indoor screen") is None


class TestModes:

    def test_input_mode(self):
        requirement = parse_resolution("Support 4K input")
        assert requirement.mode == INPUT
        assert requirement.target == (3840, 2160)
        assert requirement.is_display is False

    def test_display_mode(self):
        requirement = parse_resolution("The LED itself needs to be 4K")
        assert requirement.mode == DISPLAY
        assert requirement.is_display is True

    def test_ambiguous_needs_clarification(self):
        requirement = parse_resolution("We need 4K")
        assert requirement.mode == UNKNOWN
        assert requirement.needs_clarification is True
        assert requirement.is_display is False

    def test_profile_records_the_requirement(self):
        from src.core.requirement_extractor import RequirementExtractor

        profile = RequirementExtractor().extract(
            "indoor fixed screen, we need the LED itself to be 4K",
            use_llm=False, session_id="",
        )
        assert profile.resolution_requirement
        assert profile.resolution_requirement["mode"] == DISPLAY
