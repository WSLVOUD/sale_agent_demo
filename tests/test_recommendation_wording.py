"""客户口径（2026-09-28 第五次）：推荐话术要短 —— 不复述客户需求、不加"怎么选"评论。

客户可见原文（室内教堂 3x5、5m 视距、固装）：

    🤖 Hi, thanks for reaching out about your indoor church display. For a permanent
       installation at roughly 3m by 5m with a viewing distance around 5m, the model
       I recommend is TW11-3216-P3.0. …
       … Both options use the same TW11-3216-P3.0 cabinets and suit a permanent church
       install, so the choice mainly comes down to whether you prefer the slightly
       taller horizontal layout or the slightly wider vertical one. Would you like me
       to prepare the quotation …?

客户要求："最后推荐的话术太多了，不要重复客户需求，直接推荐就行了"。

分工：
  · 复述需求（"For a permanent installation at roughly 3m by 5m with a viewing
    distance around 5m, …"）—— 提示词层禁止（模型措辞，由 prompt 规则约束）；
  · "怎么选"的收尾评论（"Both options … so the choice comes down to …"）——
    确定性删除（`src/utils/text.strip_fact_free_commentary`，单屏与多屏共用一份）。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


CUSTOMER_TEXT = (
    "Hi there. For a permanent installation at roughly 3m by 5m with a viewing "
    "distance around 5m, the model I recommend is TW11-3216-P3.0. Its 3.076mm pixel "
    "pitch is a great match for that viewing distance, and each cabinet is 640mm by "
    "480mm with 6 modules per cabinet, which keeps the build clean and serviceable. "
    "Horizontal tiling: 5 columns x 11 rows = 55 cabinets, giving an actual screen "
    "size of 3.2m wide by 5.28m high (16.896 sqm), using 330 modules. Vertical tiling "
    "(cabinets rotated 90 degrees): 7 columns x 8 rows = 56 cabinets, giving an actual "
    "screen size of 3.36m wide by 5.12m high (17.203 sqm), using 336 modules. "
    "Both options use the same TW11-3216-P3.0 cabinets and suit a permanent church "
    "install, so the choice mainly comes down to whether you prefer the slightly "
    "taller horizontal layout or the slightly wider vertical one. Would you like me "
    "to prepare the quotation for both layouts so you can compare them side by side?"
)


class TestFactFreeCommentaryIsDropped:
    def test_closing_layout_commentary_is_removed_merged(self):
        """合并自 5 条同类测试（瘦身；断言全部保留）。"""

        # ── test_closing_layout_commentary_is_removed ──
        from src.utils.text import strip_fact_free_commentary

        cleaned = strip_fact_free_commentary(CUSTOMER_TEXT)

        assert "the choice mainly comes down to" not in cleaned, cleaned
        assert "Both options use the same" not in cleaned, cleaned

        # ── test_layout_figures_are_kept ──
        from src.utils.text import strip_fact_free_commentary

        cleaned = strip_fact_free_commentary(CUSTOMER_TEXT)

        assert "5 columns x 11 rows = 55 cabinets" in cleaned, cleaned
        assert "7 columns x 8 rows = 56 cabinets" in cleaned, cleaned
        assert "TW11-3216-P3.0" in cleaned, cleaned
        assert cleaned.rstrip().endswith("?"), cleaned[-80:]

        # ── test_question_and_request_sentences_are_never_dropped ──
        from src.utils.text import strip_fact_free_commentary

        ask = "Both options work well — shall I prepare the quotation?"
        request = "Both options work well. Please share the exact dimensions."
        assert "shall I prepare the quotation" in strip_fact_free_commentary(ask)
        assert "Please share the exact dimensions" in strip_fact_free_commentary(request)

        # ── test_sentence_with_layout_numbers_is_not_treated_as_commentary ──
        from src.utils.text import is_fact_free_commentary

        assert is_fact_free_commentary("Both options fit the wall, so you can choose either.") is True
        assert is_fact_free_commentary(
            "Both options use the same cabinet count: 6 x 10 = 60 cabinets."
        ) is False

        # ── test_all_commentary_input_keeps_the_original ──
        from src.utils.text import strip_fact_free_commentary

        only = "Both options work for this wall, so you can choose whichever you prefer."
        assert strip_fact_free_commentary(only) == only


class TestOrchestratorAppliesTheTrimOnDelivery:
    def test_postprocess_final_drops_commentary_when_delivering_merged(self):
        """合并自 2 条同类测试（瘦身；断言全部保留）。"""

        # ── test_postprocess_final_drops_commentary_when_delivering ──
        from src.dialogue.response_coordinator import ResponseCoordinator

        result = {"products": [{"model": "TW11-3216-P3.0"}]}
        out = ResponseCoordinator().postprocess_final(
            CUSTOMER_TEXT, result=result, density="DETAILED"
        )

        assert "the choice mainly comes down to" not in out, out
        assert "5 columns x 11 rows = 55 cabinets" in out, out

        # ── test_postprocess_final_drops_the_requirement_echo_prefix ──
        from src.dialogue.response_coordinator import ResponseCoordinator

        result = {"products": [{"model": "TW11-3216-P3.0"}]}
        text = (
            "For a permanent installation at roughly 3m by 5m with a viewing distance "
            "around 5m, the model I recommend is TW11-3216-P3.0."
        )
        out = ResponseCoordinator().postprocess_final(text, result=result)

        assert "permanent installation" not in out, out
        assert "viewing distance around 5m" not in out, out
        assert "TW11-3216-P3.0" in out, out
        assert out.startswith("The model I recommend"), out


class TestRequirementEchoPrefix:
    def test_prefix_is_dropped_when_the_model_follows_merged(self):
        """合并自 4 条同类测试（瘦身；断言全部保留）。"""

        # ── test_prefix_is_dropped_when_the_model_follows ──
        from src.utils.text import strip_requirement_echo_prefix

        text = "For your 3m x 5m indoor church screen, the TW11-3216-P3.0 is the best fit."
        out = strip_requirement_echo_prefix(text)

        assert out == "The TW11-3216-P3.0 is the best fit.", out

        # ── test_prefix_is_kept_when_the_rest_has_no_model ──
        from src.utils.text import strip_requirement_echo_prefix

        text = "Since your wall is 3m x 5m, we can lay it out two ways."
        assert strip_requirement_echo_prefix(text) == text

        # ── test_sentence_without_prefix_is_untouched ──
        from src.utils.text import strip_requirement_echo_prefix

        text = "TW11-3216-P3.0 is the closest fit for your wall."
        assert strip_requirement_echo_prefix(text) == text

        # ── test_clause_holding_the_model_is_never_cut ──
        from src.utils.text import strip_requirement_echo_prefix

        text = "For the TW11-3216 series, we can also quote the TW21-3216-P2.5."
        assert strip_requirement_echo_prefix(text) == text


class TestRecommendationPromptForbidsRestatingRequirements:
    def _prompt(self, monkeypatch) -> str:
        """直接抓推荐表达那次 LLM 调用的提示词（比扫源码可靠）。"""
        import importlib
        from types import SimpleNamespace

        recommend_mod = importlib.import_module("src.agents.solution.nodes.recommend")
        captured: dict = {}

        class _LLM:
            def invoke(self, prompt):
                captured["prompt"] = prompt
                return SimpleNamespace(content="TW11-3216-P3.0 is the closest match.")

        monkeypatch.setattr(recommend_mod, "get_llm", lambda **kwargs: _LLM())
        recommend_mod._express_recommendation(
            recommendations=[
                {
                    "model": "TW11-3216-P3.0",
                    "series_id": "TW11-3216",
                    "pixel_pitch_mm": 3.0,
                    "brightness_nit": 500,
                    "cabinet_size_mm": "640x480",
                    "modules_per_cabinet": 6,
                    "reasons": ["3.076mm pitch suits the 5m viewing distance"],
                }
            ],
            profile=None,
            calculation=None,
            additional_requirements=[],
            customer_text="church, indoor, permanent, 3m x 5m, 5m viewing distance",
        )
        return captured["prompt"]

    def test_prompt_tells_the_model_not_to_restate_requirements(self, monkeypatch):
        prompt = self._prompt(monkeypatch)

        assert "Do NOT restate the customer's requirements" in prompt
        assert "they already told us" in prompt

    def test_prompt_forbids_layout_comparison_commentary(self, monkeypatch):
        prompt = self._prompt(monkeypatch)

        assert "the layouts compare" in prompt or "how the two layouts compare" in prompt
        assert "the figures speak for themselves" in prompt

    def test_prompt_asks_for_conciseness_without_cutting_content(self, monkeypatch):
        prompt = self._prompt(monkeypatch)

        assert "do not pad, do not repeat" in prompt
        assert "present BOTH tiling options" in prompt
