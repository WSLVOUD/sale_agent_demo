"""客户口径：推荐不出来时不许说"找不到 / 没有匹配的产品"，
要改成"能不能放宽某个参数"的邀请。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rag.reply_composer import (  # noqa: E402
    has_no_product_phrase,
    relaxation_answer,
)
from src.rag.rerank import sanitize_customer_response  # noqa: E402


class TestBannedPhrasesDetected:

    @pytest.mark.parametrize("text", [
        "No matching products found.",
        "No suitable model found. Try adding more details.",
        "I couldn't find a model in our catalog that matches those requirements.",
        "no models available",
        "没有匹配的产品",
        "我们找不到合适的型号",
        "没有找到匹配的型号",
        "未找到合适型号",
        "无匹配产品",
        "抱歉，我找不到匹配的屏",
    ])
    def test_banned(self, text):
        assert has_no_product_phrase(text), text

    @pytest.mark.parametrize("text", [
        "The TW11-3216-P3.0 fits your conference room.",
        "这里有一款适合您的产品：TW11-3216-P3.0",
        "我们要不要放宽点间距？",
    ])
    def test_allowed(self, text):
        assert not has_no_product_phrase(text), text


class TestRelaxationAnswer:

    def test_variants_rotate(self):
        variants = {relaxation_answer("en", seed) for seed in range(8)}
        assert len(variants) >= 3
        # 每条都要点出"可放宽的具体参数"，措辞可以不同
        for text in variants:
            lowered = text.lower()
            assert any(
                word in lowered
                for word in ("pixel pitch", "screen size", "viewing distance", "brightness", "size")
            ), text

    def test_chinese(self):
        assert "放宽" in relaxation_answer("zh", 0)

    def test_mentions_concrete_parameters(self):
        text = relaxation_answer("en", 0).lower()
        assert any(word in text for word in ("pixel pitch", "screen size", "viewing distance"))


class TestSanitizerRewrites:

    @pytest.mark.parametrize("text", [
        "No matching products found.",
        "No suitable outdoor model found. Try confirming the screen size to continue.",
        "I couldn't find a model in our catalog that matches those requirements.",
        "没有匹配的产品",
        "我们找不到合适的型号",
    ])
    def test_rewritten_to_relaxation_request(self, text):
        cleaned = sanitize_customer_response(text)
        assert cleaned
        assert not has_no_product_phrase(cleaned), cleaned

    def test_normal_reply_untouched(self):
        text = "TW11-3216-P3.0 is a great fit for your conference room."
        assert sanitize_customer_response(text) == text


class TestFallbackRepliesUseRelaxation:
    """各处"取不到结果"的兜底文案都必须用放宽参数的话术。"""

    def test_api_fallback(self):
        import inspect
        import src.api as api_mod

        source = inspect.getsource(api_mod)
        assert "No suitable model found" not in source
        assert "relaxation_answer" in source

    def test_fast_path_summary(self):
        from src.rag.fast_path import _build_product_summary

        assert not has_no_product_phrase(_build_product_summary([]))

    def test_solution_runner_fallback(self):
        import inspect
        import src.agents.solution.runner as runner_mod

        source = inspect.getsource(runner_mod)
        assert "No suitable model found" not in source
        assert "relaxation_answer" in source

    def test_recommend_node_no_candidates_message(self):
        import inspect
        import src.agents.solution.nodes.recommend as recommend_mod

        source = inspect.getsource(recommend_mod)
        assert "I couldn't find a model in our catalog" not in source

    def test_recommend_prompt_forbids_no_product_claim(self):
        import inspect
        import src.agents.solution.nodes.recommend as recommend_mod

        source = inspect.getsource(recommend_mod)
        assert "No matching products found in the database" not in source
        assert "No-product rule" in source
