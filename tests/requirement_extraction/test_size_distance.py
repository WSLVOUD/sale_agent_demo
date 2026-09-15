"""数字 / 尺寸 / 视距仍然由确定性解析器负责（计划 Phase 9、Phase 28 禁止 4）。"""
import pytest


def pytest_generate_tests(metafunc):
    import json
    from pathlib import Path

    data = json.loads(
        (Path(__file__).parent / "golden_cases.json").read_text(encoding="utf-8")
    )
    if "dist_case" in metafunc.fixturenames:
        cases = data["distance"]
        metafunc.parametrize("dist_case", cases, ids=[c["text"][:40] for c in cases])
    if "size_case" in metafunc.fixturenames:
        cases = data["size"]
        metafunc.parametrize("size_case", cases, ids=[c["text"][:40] for c in cases])


def test_distance_parsing(dist_case, extractor):
    profile = extractor.extract(dist_case["text"], use_llm=False)
    assert profile.viewing_distance_m == pytest.approx(dist_case["expect"], abs=0.01), (
        f"{dist_case['text']} → {profile.viewing_distance_m}"
    )


def test_size_parsing(size_case, extractor):
    profile = extractor.extract(size_case["text"], use_llm=False)
    if "hint_mm" in size_case:
        assert profile.screen_size_hint_mm == pytest.approx(size_case["hint_mm"], abs=0.1)
        return
    width = profile.target_width_mm
    height = profile.target_height_mm
    expected_width = size_case.get("width")
    expected_height = size_case.get("height")
    if expected_width is None:
        assert width is None
    else:
        assert width == pytest.approx(expected_width, abs=1.0)
    if expected_height is None:
        assert height is None
    else:
        assert height == pytest.approx(expected_height, abs=1.0)


class TestNoLlmInNumberParsing:
    """数值必须来自解析器，不允许 LLM 参与（计划 Phase 9）。"""

    def test_distance_is_explicit_source(self, extractor):
        profile = extractor.extract("about 5m viewing distance", use_llm=False)
        assert profile.viewing_distance_m == pytest.approx(5.0)
        assert profile.sources.get("viewing_distance_m") == "explicit"

    def test_size_is_explicit_source(self, extractor):
        profile = extractor.extract("The screen should be 5m x 3m.", use_llm=False)
        assert profile.target_width_mm == pytest.approx(5000.0)
        assert profile.sources.get("target_width_m") == "explicit"
