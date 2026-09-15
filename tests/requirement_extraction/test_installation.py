"""Installation 统一解析（计划 Phase 8）。"""
import pytest


def pytest_generate_tests(metafunc):
    if "inst_case" in metafunc.fixturenames:
        import json
        from pathlib import Path

        data = json.loads(
            (Path(__file__).parent / "golden_cases.json").read_text(encoding="utf-8")
        )
        cases = data["installation"]
        metafunc.parametrize("inst_case", cases, ids=[c["text"][:40] for c in cases])


def test_installation_resolution(inst_case, extractor):
    profile = extractor.extract(inst_case["text"], use_llm=False)
    assert profile.installation == inst_case["expect"], (
        f"{inst_case['text']} → {profile.installation!r}，期望 {inst_case['expect']!r}"
    )


def test_no_forced_guess_when_not_stated(extractor):
    """客户没说安装方式时不能硬猜（可以来自场景默认，但来源必须是 default）。"""
    profile = extractor.extract("We need a screen for a church.", use_llm=False)
    assert profile.installation == "fixed"
    assert profile.sources.get("installation") == "default"


def test_explicit_customer_wins_over_scene_default(extractor):
    """"给会议室租一块屏"是正常业务：客户明确 rental 时不能被场景默认覆盖。"""
    profile = extractor.extract("We need a rental screen for a conference room.", use_llm=False)
    assert profile.installation == "rental"
    assert profile.sources.get("installation") == "explicit"
