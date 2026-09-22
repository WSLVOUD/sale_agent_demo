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
from tests._cases import assert_all_cases  # noqa: E402


def _is_banned(text: str) -> None:
    assert has_no_product_phrase(text), text


def _is_allowed(text: str) -> None:
    assert not has_no_product_phrase(text), text


class TestBannedPhrasesDetected:

    # 用例表（2026-09-22 瘦身：一条测试跑整张表）
    BANNED_TEXTS = [
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
    ]
    ALLOWED_TEXTS = [
        "The TW11-3216-P3.0 fits your conference room.",
        "这里有一款适合您的产品：TW11-3216-P3.0",
        "我们要不要放宽点间距？",
    ]

    def test_banned(self):
        assert_all_cases(self.BANNED_TEXTS, _is_banned, label="text")

    def test_allowed(self):
        assert_all_cases(self.ALLOWED_TEXTS, _is_allowed, label="text")


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


class TestFastPathRespectsAccumulatedRequirements:
    """实测 bug：客户前面已说"室外 + 租赁"，后面只回一句 "P3" 时，
    fast path 只用本条消息抽到的点间距约束 → 给室外租赁推荐了室内型号
    （TW11-3216 / TW21-3216），随后被回复清洗器删光，最后退化成"放宽条件"兜底。
    """

    def _fast(self, constraints):
        from src.rag.fast_path import fast_path_handle

        return fast_path_handle(
            query="P3", constraints=constraints, template_type=None, data_dir="data"
        )

    def test_constraint_merge_pulls_in_accumulated_requirements(self):
        from src.agents.solution.runner import merge_fast_path_constraints

        merged = merge_fast_path_constraints(
            {"pixel_pitch": 3.0, "pixel_pitch_tolerance": 0.5},
            {
                "display_type": "LED",
                "location_type": "室外",
                "outdoor": True,
                "indoor": False,
                "is_rental": True,
            },
        )
        assert merged["pixel_pitch"] == 3.0
        assert merged["display_type"] == "LED"
        assert merged["outdoor"] is True and merged["indoor"] is False
        assert merged["is_rental"] is True

    def test_outdoor_rental_never_returns_indoor_models(self):
        result = self._fast({
            "pixel_pitch": 3.0, "pixel_pitch_tolerance": 0.5,
            "outdoor": True, "indoor": False, "is_rental": True, "display_type": "LED",
        })
        # 目录里没有"室外 + 租赁"系列 → 允许为空，但绝不能推室内型号
        assert all(p["installation"] == "rental" for p in result["products"]), result["products"]

    def test_outdoor_fixed_returns_only_outdoor_and_survives_sanitizer(self):
        from src.rag.rerank import sanitize_customer_response

        result = self._fast({
            "pixel_pitch": 3.0, "pixel_pitch_tolerance": 0.5,
            "outdoor": True, "indoor": False, "is_rental": False, "display_type": "LED",
        })
        assert result["products"], "室外固装 P3 应该能匹配到户外固装型号"
        assert all(("OD" in p["model"] or "HOD" in p["model"]) for p in result["products"]), result["products"]
        # 清洗器（outdoor=True）不会再把它整段删掉
        assert sanitize_customer_response(result["answer"], outdoor=True).strip()

    def test_no_match_gives_relaxation_not_no_product(self):
        # 2026-09-18：库里新增了室外租赁（TW11-OR 等，最大 P4.8），
        # 所以"室外租赁 + P6"才是真正无匹配的组合。
        result = self._fast({"pixel_pitch": 6.0, "outdoor": True, "is_rental": True})
        assert not result["products"]
        assert result["answer"].strip()
        assert not has_no_product_phrase(result["answer"])


class TestEnglishOnlyGuard:
    """客户口径：策略=en（默认）时，回复里**不能出现任何一句中文**。

    实测 bug：客户问"anything else?"，自由问答节点（others）用中文答了一段。
    """

    def _policy_en(self, monkeypatch):
        from src.config import config

        monkeypatch.setattr(config, "RESPONSE_LANGUAGE_POLICY", "en")

    def test_cjk_detection(self):
        from src.rag.reply_composer import contains_cjk

        assert contains_cjk("当然有，还有几点值得您一起考虑一下。")
        assert not contains_cjk("Sure, here is another option for you.")

    def test_english_reply_needs_no_rewrite_call(self, monkeypatch):
        import src.core.llm as llm_mod
        from src.rag.reply_composer import enforce_english

        self._policy_en(monkeypatch)

        def _boom(*args, **kwargs):
            raise AssertionError("英文回复不应该触发重写调用")

        monkeypatch.setattr(llm_mod, "get_llm", _boom)
        text = "TW11-3216-P3.0 is the right fit for your church screen."
        assert enforce_english(text, message="anything else?") == text

    def test_chinese_reply_is_rewritten_into_english(self, monkeypatch):
        import src.core.llm as llm_mod
        from src.rag.reply_composer import enforce_english

        self._policy_en(monkeypatch)
        calls = []

        class _Resp:
            content = "Of course, there are a few more things worth considering."

        class _LLM:
            def invoke(self, prompt, *args, **kwargs):
                calls.append(prompt)
                return _Resp()

        monkeypatch.setattr(llm_mod, "get_llm", lambda *a, **k: _LLM())
        out = enforce_english("当然有，还有几点值得您一起考虑一下。", message="anything else?")
        assert out == "Of course, there are a few more things worth considering."
        assert calls, "命中中文必须调用一次 LLM 重写"

    def test_unfixable_chinese_is_dropped_not_sent(self, monkeypatch):
        import src.core.llm as llm_mod
        from src.rag.reply_composer import enforce_english

        self._policy_en(monkeypatch)

        class _Bad:
            def invoke(self, *args, **kwargs):
                raise RuntimeError("offline")

        monkeypatch.setattr(llm_mod, "get_llm", lambda *a, **k: _Bad())
        assert enforce_english("这是中文回复。", message="hi") == ""

    def test_rewrite_still_chinese_is_dropped(self, monkeypatch):
        import src.core.llm as llm_mod
        from src.rag.reply_composer import enforce_english

        self._policy_en(monkeypatch)

        class _StillChinese:
            content = "还是中文"

            def invoke(self, *args, **kwargs):
                return self

        monkeypatch.setattr(llm_mod, "get_llm", lambda *a, **k: _StillChinese())
        assert enforce_english("这是中文回复。", message="hi") == ""

    def test_policy_auto_keeps_customer_language(self, monkeypatch):
        from src.config import config
        from src.rag.reply_composer import enforce_english

        monkeypatch.setattr(config, "RESPONSE_LANGUAGE_POLICY", "auto")
        assert enforce_english("这是中文回复。", message="你好，我要一块屏") == "这是中文回复。"

    def test_api_never_returns_chinese_under_english_policy(self, monkeypatch):
        """端到端：编排器返回中文时，/chat 出去的文本必须是英文（或英文兜底）。"""
        import asyncio

        import src.core.llm as llm_mod
        from src import api
        from src.config import config

        monkeypatch.setattr(config, "RESPONSE_LANGUAGE_POLICY", "en")
        monkeypatch.setattr(llm_mod, "get_llm", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))

        class _ChineseOrchestrator:
            def process_message(self, message, session_id, images=None):
                return {
                    "response": "当然有，还有几点值得您一起考虑一下。",
                    "requirements": {},
                    "products": [],
                    "route": "agent",
                    "complexity": "simple",
                    "_perf": {},
                }

        original = api.orchestrator
        api.orchestrator = _ChineseOrchestrator()
        try:
            result = asyncio.run(
                api._chat_sync(api.ChatRequest(session_id="lang-guard", question="anything else?"))
            )
        finally:
            api.orchestrator = original

        from src.rag.reply_composer import contains_cjk

        assert result.answer.strip(), "中文被丢弃后必须有英文兜底"
        assert not contains_cjk(result.answer), result.answer
