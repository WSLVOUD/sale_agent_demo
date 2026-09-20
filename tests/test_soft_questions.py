"""v2.2.4：软问题（使用场景 / 价位取向）恢复提问，且永不阻塞推荐。

客户口径：

    · 硬性条件 = 室内外 / 固装租赁 / P值 /（反推 P 值用的）观看距离 / 尺寸
    · 场景、价位取向是销售话术里的软问题：**只在硬性条件还没问完时顺带问**，
      硬性条件一齐就立刻推荐，不再问别的；
    · 客户的回答要落到字段上：场景 → purpose（并可能定室内外），
      价位取向 → price_preference（price / both → 默认档 low，quality → medium）。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.field_policy import is_hard_condition  # noqa: E402
from src.rag.readiness import check_recommendation_ready  # noqa: E402


def _profile(**slots) -> RequirementProfile:
    return RequirementProfile.from_slots(slots, explicit_keys=set(slots))


class TestHardConditionClassification:

    @pytest.mark.parametrize("slot", [
        "environment", "installation", "pixel_pitch", "viewing_distance", "size",
    ])
    def test_hard(self, slot):
        assert is_hard_condition(slot) is True

    @pytest.mark.parametrize("slot", ["purpose", "price_preference", "content_type"])
    def test_soft(self, slot):
        assert is_hard_condition(slot) is False


class TestScenarioQuestionIsBack:

    def test_purpose_is_asked_first(self):
        """环境之后就是场景（销售话术顺序）。"""
        decision = check_recommendation_ready(_profile(environment="indoor"))
        assert decision.next_slot == "purpose"
        question = (decision.next_question or "").lower()
        assert any(
            phrase in question
            for phrase in ("use it for", "used for", "application", "use case", "used")
        ), question

    def test_purpose_answer_sets_environment_too(self):
        from src.core.requirement_extractor import RequirementExtractor

        profile = RequirementExtractor().extract(
            "it's for a church", use_llm=False, session_id=""
        )
        assert profile.purpose == "church"
        assert profile.environment == "indoor"

    def test_scene_answer_is_used_for_scoring(self):
        """场景落到 purpose 后参与打分（不是只存着）。"""
        from src.rag.recommendation_engine import RecommendationEngine

        slots = {
            "display_type": "LED", "environment": "indoor", "installation": "fixed",
            "purpose": "church", "viewing_distance_m": 8,
            "target_width_mm": 5000, "target_height_mm": 3000,
        }
        result = RecommendationEngine().recommend(
            profile=_profile(**slots), top_k=3
        )
        assert result["recommendations"]
        assert result["recommendations"][0]["breakdown"]["scene"] is not None


class TestPricePreferenceQuestionIsBack:

    def test_asked_while_hard_conditions_are_pending(self):
        profile = _profile(environment="indoor", purpose="church", installation="fixed")
        decision = check_recommendation_ready(profile)
        assert decision.next_slot == "price_preference"
        question = (decision.next_question or "").lower()
        assert "price" in question and "quality" in question
        # 软问题不是"拦住推荐"的字段
        assert "price_preference" not in decision.missing
        assert "pixel_pitch" in decision.missing

    @pytest.mark.parametrize("answer,expected_preference,expected_tier", [
        ("price first", "price", "low"),
        ("quality first", "quality", "medium"),
        ("price and quality both matter", "both", "low"),
        ("both are fine", "both", "low"),
    ])
    def test_answer_maps_to_budget_tier(self, answer, expected_preference, expected_tier):
        from src.core.requirement_extractor import RequirementExtractor

        profile = RequirementExtractor().extract(answer, use_llm=False, session_id="")
        assert profile.price_preference == expected_preference
        assert profile.budget_level == expected_tier

    def test_bare_both_lands_on_the_asked_slot(self):
        """客户只回一个 "both"：按"上一轮问的就是价位取向"落地。"""
        from src.core.requirement_extractor import RequirementExtractor

        previous = _profile(
            display_type="LED", environment="indoor", installation="fixed",
            pixel_pitch_mm=3.0, target_width_mm=5000, target_height_mm=3000,
        )
        previous.last_asked_slot = "price_preference"
        profile = RequirementExtractor().extract(
            "both", previous_profile=previous, use_llm=False, session_id=""
        )
        assert profile.price_preference == "both"
        assert profile.budget_level == "low"

    def test_bare_both_is_not_used_for_other_questions(self):
        from src.core.requirement_extractor import RequirementExtractor

        previous = RequirementProfile()
        previous.last_asked_slot = "pixel_pitch"
        profile = RequirementExtractor().extract(
            "both", previous_profile=previous, use_llm=False, session_id=""
        )
        assert profile.price_preference is None


class TestSoftQuestionsNeverBlock:

    def test_all_hard_conditions_ready_without_soft_answers(self):
        profile = _profile(
            environment="indoor", installation="fixed", pixel_pitch_mm=3.0,
            target_width_mm=5000, target_height_mm=3000,
        )
        decision = check_recommendation_ready(profile)
        assert decision.ready is True
        assert decision.status == "READY"
        assert "purpose" not in (decision.missing or [])
        assert "price_preference" not in (decision.missing or [])

    def test_ready_means_no_more_questions(self):
        """硬性条件齐了 → 连软问题都不问（客户口径：不要再问别的）。"""
        profile = _profile(
            environment="indoor", installation="fixed", pixel_pitch_mm=2.5,
            target_width_mm=5000, target_height_mm=3000,
        )
        decision = check_recommendation_ready(profile)
        assert decision.next_question in (None, "")
        assert decision.next_slot == ""


class TestQuestionOrderThroughTheSalesNode:
    """走真实节点：问题顺序必须是 环境 → 场景 → 安装 → 价位取向 → P值 → 尺寸。"""

    @pytest.fixture
    def sales_node(self, monkeypatch):
        import importlib

        sales_req = importlib.import_module("src.agents.sales.nodes.requirement")
        extractor_mod = importlib.import_module("src.core.requirement_extractor")

        class _Response:
            content = '{"usage": null, "additional_requirements": [], "ack": ""}'

        class _FakeChat:
            def __init__(self, *args, **kwargs):
                pass

            def invoke(self, *args, **kwargs):
                return _Response()

        monkeypatch.setattr(sales_req, "ChatOpenAI", _FakeChat)
        monkeypatch.setattr(
            extractor_mod.RequirementExtractor,
            "_llm_semantic_extract",
            lambda self, message, rule_slots, session_id="": {},
        )
        extractor_mod.RequirementExtractor._semantic_cache.clear()
        return sales_req

    def _turn(self, module, message, profile=None):
        state = {
            "messages": [{"role": "user", "content": message}],
            "current_message": message, "session_id": "soft-questions",
            "requirements": {}, "additional_requirements": [],
            "intent": "need_query", "next_action": "ask",
            "should_generate_solution": False, "response": "",
            "pending_question": "", "pending_slot": "",
        }
        if profile is not None:
            state["requirement_profile"] = profile
        return module.requirement_mining(state)

    def test_question_sequence(self, sales_node):
        answers = {
            "environment": "indoor",
            "purpose": "it's for a church",
            "installation": "permanent installation",
            "price_preference": "both are fine",
            "pixel_pitch": "P3",
            "size": "5m x 3m",
        }
        out = self._turn(sales_node, "I need an LED screen")
        asked = []
        for _ in range(8):
            slot = out["pending_slot"]
            if out["should_generate_solution"] or not slot:
                break
            asked.append(slot)
            out = self._turn(sales_node, answers.get(slot, "indoor"),
                             out["requirement_profile"])

        assert asked[:6] == [
            "environment", "purpose", "installation",
            "price_preference", "pixel_pitch", "size",
        ], asked
        assert out["should_generate_solution"] is True
        profile = out["requirement_profile"]
        assert profile.purpose == "church"
        assert profile.price_preference == "both"
        assert profile.budget_level == "low"
