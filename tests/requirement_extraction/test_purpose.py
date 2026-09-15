"""Purpose 归一化 + 关键词独立性（计划 Phase 5.2 / 20.1-20.3 / 27）。"""
import pytest

from .conftest import run_llm_tests


def _extract_purpose(extractor, text: str, use_llm: bool = False):
    return extractor.extract(text, use_llm=use_llm).purpose


def pytest_generate_tests(metafunc):
    """按 golden_cases.json 动态生成 purpose 用例。"""
    if "purpose_case" in metafunc.fixturenames:
        import json
        from pathlib import Path

        data = json.loads(
            (Path(__file__).parent / "golden_cases.json").read_text(encoding="utf-8")
        )
        metafunc.parametrize(
            "purpose_case",
            data["purpose"],
            ids=[c["text"][:40] for c in data["purpose"]],
        )


def test_purpose_normalization(purpose_case, extractor):
    """规则层能做到就必须做对；做不到的交给 LLM 语义层（离线时跳过）。"""
    text, expected = purpose_case["text"], purpose_case["expect"]
    got = _extract_purpose(extractor, text)
    if got == expected:
        return
    assert got is None, f"规则层给出了错误 purpose：{got!r} != {expected!r}（{text}）"
    if not run_llm_tests():
        pytest.skip("需要 LLM 语义层：RUN_LLM_EXTRACTION_TESTS=1")
    assert _extract_purpose(extractor, text, use_llm=True) == expected, text


class TestKeywordIndependence:
    """计划 Phase 27：至少 30% 的 purpose 用例不含关键词表里的原始关键词。"""

    def test_dataset_has_enough_keyword_free_cases(self, golden_cases):
        from src.rag.query_understanding import _PURPOSE_KEYWORDS

        tokens = [k.lower() for _, keywords, _ in _PURPOSE_KEYWORDS for k in keywords]
        cases = golden_cases["purpose"]
        free = [
            case for case in cases
            if not any(token in case["text"].lower() for token in tokens)
        ]
        ratio = len(free) / len(cases)
        assert ratio >= 0.30, f"关键词无关用例只占 {ratio:.0%}（要求 ≥30%）"

    def test_keyword_free_cases_resolve_without_llm(self, golden_cases, extractor):
        """关键词表没覆盖的说法里，短语表（canonical phrase fast path）必须能解掉一批，
        其余交给 LLM 语义层 —— 但不能猜错（猜错由 test_purpose_normalization 拦住）。"""
        from src.rag.query_understanding import _PURPOSE_KEYWORDS

        tokens = [k.lower() for _, keywords, _ in _PURPOSE_KEYWORDS for k in keywords]
        free = [
            case for case in golden_cases["purpose"]
            if not any(token in case["text"].lower() for token in tokens)
        ]
        resolved = sum(
            1 for case in free if _extract_purpose(extractor, case["text"]) == case["expect"]
        )
        assert resolved >= 8, (
            f"关键词无关用例离线只解出 {resolved}/{len(free)} 条，短语表覆盖不足"
        )
