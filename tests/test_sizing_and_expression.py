"""
回归测试：
  1. 客户用不同单位回答屏幕尺寸（m/cm/mm/英尺/英寸、只写一侧单位、不写单位）
  2. 缺尺寸时必须追问；给出尺寸后立即计算箱体与模组
  3. 推荐话术不是固定模板（措辞会变化）
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.readiness import check_calculation_ready, question_for  # noqa: E402
from src.rag.query_understanding import extract_slots  # noqa: E402


def _confirmed(slots: dict) -> RequirementProfile:
    return RequirementProfile.from_slots(slots, explicit_keys=set(slots))


RECOMMEND_READY = {
    "environment": "indoor", "purpose": "church",
    "installation": "fixed", "viewing_distance_m": 5,
}


class TestScreenSizeUnits:
    """尺寸单位换算（客户不一定会用米）"""

    @pytest.mark.parametrize("text,width,height", [
        ("5m x 3m", 5000.0, 3000.0),
        ("5米*3米", 5000.0, 3000.0),
        ("5000x3000mm", 5000.0, 3000.0),
        ("500cm x 300cm", 5000.0, 3000.0),
        ("5000 x 3000", 5000.0, 3000.0),          # 无单位 → 按量级推断为 mm
        ("3m x 2m", 3000.0, 2000.0),
        ("10 m by 6 m", 10000.0, 6000.0),          # by 分隔
    ])
    def test_common_units(self, text, width, height):
        slots = extract_slots(text)
        assert slots.get("target_width_mm") == pytest.approx(width)
        assert slots.get("target_height_mm") == pytest.approx(height)

    def test_imperial_units(self):
        slots = extract_slots("16ft x 9ft")
        assert slots.get("target_width_mm") == pytest.approx(4876.8, abs=0.5)
        assert slots.get("target_height_mm") == pytest.approx(2743.2, abs=0.5)

    def test_single_unit_side_is_inherited(self):
        """只写一侧单位时，另一侧沿用（5m x 3 / 5000 x 3000mm）"""
        a = extract_slots("5m x 3")
        assert a.get("target_width_mm") == 5000.0
        assert a.get("target_height_mm") == 3000.0

        b = extract_slots("5000 x 3000mm")
        assert b.get("target_width_mm") == 5000.0
        assert b.get("target_height_mm") == 3000.0

    def test_single_dimension(self):
        assert extract_slots("5米宽").get("target_width_mm") == 5000.0
        assert extract_slots("3米高").get("target_height_mm") == 3000.0

    def test_size_is_not_mistaken_for_distance(self):
        assert extract_slots("5m x 3m").get("viewing_distance_m") is None
        assert extract_slots("16ft x 9ft").get("viewing_distance_m") is None
        # 裸数值仍然是观看距离
        assert extract_slots("5m").get("viewing_distance_m") == 5.0
        assert extract_slots("about 5 meters").get("viewing_distance_m") == 5.0


class TestCalculationGate:

    def test_missing_size_blocks_calculation(self):
        decision = check_calculation_ready(_confirmed(RECOMMEND_READY))
        assert decision.ready is False
        assert set(decision.missing) == {"width", "height"}
        assert decision.next_question

    def test_size_question_has_variants(self):
        texts = {question_for("size", "en", seed) for seed in range(8)}
        assert len(texts) >= 2
        assert all(("width" in t.lower() and "height" in t.lower()) for t in texts)

    def test_ready_when_size_known(self):
        profile = _confirmed({**RECOMMEND_READY, "target_width_mm": 5000, "target_height_mm": 3000})
        assert check_calculation_ready(profile).ready is True

    def test_single_dimension_only_asks_for_the_other(self):
        profile = _confirmed({**RECOMMEND_READY, "target_width_mm": 5000})
        decision = check_calculation_ready(profile)
        assert decision.missing == ["height"]


class TestRecommendNodeSizing:
    """缺尺寸要追问；有了尺寸要立刻算箱体/模组"""

    def _state(self, extra_slots: dict):
        slots = {**RECOMMEND_READY, **extra_slots}
        return {
            "requirement": {},
            "requirement_profile": _confirmed(slots),
            "products": [],
            "messages": [{"role": "user", "content": "church"}],
            "current_message": "church",
            "additional_requirements": [],
        }

    def test_asks_for_size_when_missing(self, monkeypatch):
        import src.agents.solution.nodes.recommend as recommend

        class _Failing:
            def invoke(self, *args, **kwargs):
                raise RuntimeError("offline")

        monkeypatch.setattr(recommend, "get_llm", lambda *a, **k: _Failing())
        result = recommend.recommend_node(self._state({}))

        assert result["screen_calculation"] is None
        answer = result["recommendation"].lower()
        assert "width" in answer and "height" in answer, answer

    def test_calculates_once_size_is_known(self, monkeypatch):
        import src.agents.solution.nodes.recommend as recommend

        class _Failing:
            def invoke(self, *args, **kwargs):
                raise RuntimeError("offline")

        monkeypatch.setattr(recommend, "get_llm", lambda *a, **k: _Failing())
        result = recommend.recommend_node(
            self._state({"target_width_mm": 5000, "target_height_mm": 3000})
        )

        calc = result["screen_calculation"]
        assert calc is not None
        assert calc["cabinet_count"] == 56
        assert calc["total_modules"] == 336
        assert "56" in result["recommendation"]

    def test_calculates_for_centimetre_answer(self, monkeypatch):
        """客户用厘米回答（500cm x 300cm）也要能直接算"""
        import src.agents.solution.nodes.recommend as recommend

        class _Failing:
            def invoke(self, *args, **kwargs):
                raise RuntimeError("offline")

        monkeypatch.setattr(recommend, "get_llm", lambda *a, **k: _Failing())

        slots = extract_slots("500cm x 300cm")
        result = recommend.recommend_node(self._state(slots))
        assert result["screen_calculation"]["cabinet_count"] == 56


class TestRecommendationWordingVaries:

    def test_fallback_openers_vary(self, monkeypatch):
        """LLM 不可用时的模板话术也要有多种说法，而不是每次都一样"""
        import src.agents.solution.nodes.recommend as recommend

        class _Failing:
            def invoke(self, *args, **kwargs):
                raise RuntimeError("offline")

        monkeypatch.setattr(recommend, "get_llm", lambda *a, **k: _Failing())
        slots = {**RECOMMEND_READY, "target_width_mm": 5000, "target_height_mm": 3000}
        profile = _confirmed(slots)

        answers = set()
        for _ in range(12):
            answers.add(recommend._express_recommendation(
                recommendations=[{
                    "model": "TW11-3216-P2.5", "pixel_pitch_mm": 2.5, "brightness_nit": 500,
                    "cabinet_size_mm": "640mm*480mm", "modules_per_cabinet": 6,
                    "price_tier": "low", "warranty_years": 1, "reasons": ["pitch fits"],
                }],
                profile=profile,
                calculation={"columns": 8, "rows": 7, "cabinet_count": 56,
                             "actual_width_m": 5.12, "actual_height_m": 3.36,
                             "area_sqm": 17.2, "total_modules": 336},
                additional_requirements=[],
                customer_text="church",
            ))
        assert len(answers) >= 2, "模板话术应该有不同的开头"

    def test_prompt_asks_for_varied_wording(self):
        """给 LLM 的指令里必须要求变化措辞"""
        import inspect

        import src.agents.solution.nodes.recommend as recommend

        source = inspect.getsource(recommend._express_recommendation)
        assert "Vary your wording" in source


_TOP = {
    "model": "TW11-3216-P3.0", "pixel_pitch_mm": 3.0, "brightness_nit": 500,
    "cabinet_size_mm": "640mm*480mm", "modules_per_cabinet": 6,
    "price_tier": "low", "warranty_years": 1, "series_id": "TW11",
    "features": [], "installation": "fixed", "reasons": ["pitch fits the viewing distance"],
}
_ALT_BRIGHT = {
    "model": "TW21-3216-P3.0", "pixel_pitch_mm": 3.0, "brightness_nit": 6000,
    "cabinet_size_mm": "640mm*480mm", "modules_per_cabinet": 6,
    "price_tier": "medium", "warranty_years": 2, "series_id": "TW21",
    "features": ["cob"], "installation": "fixed", "reasons": [],
}
_ALT_PITCH = {
    "model": "TW11-3216-P4.0", "pixel_pitch_mm": 4.0, "brightness_nit": 500,
    "cabinet_size_mm": "640mm*480mm", "modules_per_cabinet": 6,
    "price_tier": "low", "warranty_years": 1, "series_id": "TW11",
    "features": [], "installation": "fixed", "reasons": [],
}


class TestAlternativesReplyFormat:
    """客户问"还有其他推荐吗"时的回答格式（客户口径）：

        "If you want higher brightness, TW21-3216-P3.0." + 邀请补充需求；
        不重讲首选、不催尺寸、**绝不提价格**。
    """

    def _profile(self):
        return _confirmed({**RECOMMEND_READY, "viewing_distance_m": 10})

    def test_request_is_detected(self):
        import src.agents.solution.nodes.recommend as recommend

        for message in ("你还有其他的推荐吗？", "还有其他推荐吗", "有没有别的型号",
                        "还有什么方案", "any other options?"):
            assert recommend._ALTERNATIVES_RE.search(message), message
        assert not recommend._ALTERNATIVES_RE.search("我需要便宜质量好的屏幕")

    def test_fallback_offers_conditional_alternatives_with_invitation(self, monkeypatch):
        import src.agents.solution.nodes.recommend as recommend

        class _Failing:
            def invoke(self, *args, **kwargs):
                raise RuntimeError("offline")

        monkeypatch.setattr(recommend, "get_llm", lambda *a, **k: _Failing())

        answer = recommend._express_recommendation(
            recommendations=[_TOP, _ALT_BRIGHT, _ALT_PITCH],
            profile=self._profile(),
            calculation=None,
            additional_requirements=[],
            customer_text="你还有其他的推荐吗？",
            need_size_question=True,      # 缺尺寸也不该在本轮催尺寸
            follow_up=True,
        )
        lowered = answer.lower()

        assert "if you want" in lowered, answer
        assert "tw21-3216-p3.0" in lowered, answer
        assert "other requirements" in lowered, answer          # 邀请补充需求
        assert "width and height" not in lowered, answer        # 不在这一轮催尺寸
        for word in ("price", "cost", "budget", "tier", "cheap"):
            assert word not in lowered, (word, answer)

    def test_prompt_forbids_price_and_asks_conditional_alternatives(self, monkeypatch):
        """给 LLM 的提示词里：备选要说成条件句、且禁止提价格。"""
        import src.agents.solution.nodes.recommend as recommend

        captured = {}

        class _Capture:
            def invoke(self, prompt, *args, **kwargs):
                captured["prompt"] = prompt if isinstance(prompt, str) else str(prompt)

                class _R:
                    content = "ok"
                return _R()

        monkeypatch.setattr(recommend, "get_llm", lambda *a, **k: _Capture())
        recommend._express_recommendation(
            recommendations=[_TOP, _ALT_BRIGHT],
            profile=self._profile(),
            calculation=None,
            additional_requirements=[],
            customer_text="any other options?",
            follow_up=True,
        )
        prompt = captured["prompt"]
        # 产品数据里不能再带价格档位（规则里提到"price tier"是为了禁止它）
        data_section = prompt.split("Rules:")[0]
        assert "price tier" not in data_section, data_section
        assert "NEVER mention price" in prompt
        assert "If you want <that difference>" in prompt
        assert "higher brightness" in prompt      # 备选差异说明
