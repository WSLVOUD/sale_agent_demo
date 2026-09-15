"""Environment 统一解析（计划 Phase 7 / 20.4 / 27）。"""
import pytest


def pytest_generate_tests(metafunc):
    if "env_case" in metafunc.fixturenames:
        import json
        from pathlib import Path

        data = json.loads(
            (Path(__file__).parent / "golden_cases.json").read_text(encoding="utf-8")
        )
        cases = data["environment"]
        metafunc.parametrize("env_case", cases, ids=[c["text"][:40] for c in cases])


def test_environment_resolution(env_case, extractor):
    """明确室内/室外必须解析出来；室内外都可能（concert/stage/wedding/rental）
    必须保持 None，交给 Gate 继续追问。"""
    profile = extractor.extract(env_case["text"], use_llm=False)
    assert profile.environment == env_case["expect"], (
        f"{env_case['text']} → {profile.environment!r}，期望 {env_case['expect']!r}"
    )


class TestAmbiguousScenesStayAmbiguous:
    """计划 Phase 7.1：禁止把 concert/stage/wedding/rental 猜成室内或室外。"""

    @pytest.mark.parametrize("text", [
        "We need a screen for a concert.",
        "It is for a stage performance.",
        "We need screens for a wedding.",
        "It is for a rental event.",
    ])
    def test_ambiguous_keeps_asking(self, extractor, text):
        profile = extractor.extract(text, use_llm=False)
        assert profile.environment is None
