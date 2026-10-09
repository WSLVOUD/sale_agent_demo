from types import SimpleNamespace

import pytest

from src.rag.readiness import format_measurement, size_hint_sentence
from src.rag.readiness_measurements import (
    format_measurement as format_measurement_helper,
    size_hint_sentence as size_hint_sentence_helper,
)


@pytest.mark.parametrize(
    ("millimetres", "expected"),
    [
        (1292, "129.2 cm"),
        (1300, "130 cm"),
        (99.9, "99.9 mm"),
        (0, ""),
        (-1, ""),
        (None, ""),
    ],
)
def test_format_measurement(millimetres, expected):
    assert format_measurement(millimetres) == expected


def test_measurement_helpers_remain_available_from_readiness():
    assert format_measurement is format_measurement_helper
    assert size_hint_sentence is size_hint_sentence_helper


def test_size_hint_sentence_only_uses_positive_image_measurements():
    profile = SimpleNamespace(vision_size_hint_mm=[2400, 1350])

    assert size_hint_sentence(profile) == "The image suggests roughly 2.4m x 1.35m."
    assert size_hint_sentence(profile, "zh") == "图片上看大约是 2.4 米 × 1.35 米。"
    assert size_hint_sentence(SimpleNamespace(vision_size_hint_mm=[0, 1350])) == ""
    assert size_hint_sentence(SimpleNamespace(vision_size_hint_mm=[2400])) == ""
