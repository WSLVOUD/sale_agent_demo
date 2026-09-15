"""多语言场景识别（计划 Phase 21）：中/英离线可跑，其它语言依赖 LLM 语义层。"""
import pytest

from .conftest import run_llm_tests


def pytest_generate_tests(metafunc):
    if "ml_case" in metafunc.fixturenames:
        import json
        from pathlib import Path

        data = json.loads(
            (Path(__file__).parent / "golden_cases.json").read_text(encoding="utf-8")
        )
        cases = data["multilingual"]
        metafunc.parametrize(
            "ml_case", cases, ids=[f"{c['lang']}:{c['text'][:30]}" for c in cases]
        )


def test_multilingual_purpose(ml_case, extractor):
    text, expected = ml_case["text"], ml_case["expect_purpose"]
    profile = extractor.extract(text, use_llm=False)
    if profile.purpose == expected:
        return
    if ml_case.get("requires_semantic"):
        if not run_llm_tests():
            pytest.skip("非中英文场景需要 LLM 语义层（RUN_LLM_EXTRACTION_TESTS=1）")
        assert extractor.extract(text, use_llm=True).purpose == expected, text
        return
    assert profile.purpose == expected, f"{text} → {profile.purpose!r}，期望 {expected!r}"


class TestLanguageDetection:

    @pytest.mark.parametrize("text,lang", [
        ("我们需要一块会议室的屏", "zh"),
        ("We need a display for a church.", "en"),
        ("Wir brauchen einen Bildschirm.", "de"),
        ("Нам нужен экран.", "ru"),
        ("会議室用のディスプレイ", "ja"),
    ])
    def test_detect_language(self, text, lang):
        from src.rag.query_understanding import detect_language

        assert detect_language(text) == lang
