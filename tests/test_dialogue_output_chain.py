"""v2.5+++（对话决策与输出链路优化）：Environment / 单问题 / Pitch / Grounded Facts / Numeric。

对应计划书《LED_RAG_对话决策与输出链路优化计划.md》§13 的五组测试。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import (  # noqa: E402
    FinalResponseGuard,
    ResponseCoordinator,
    build_context,
    build_grounded_facts,
    next_question_plan,
    validate_response,
)
from src.dialogue.grounded_facts import (  # noqa: E402
    SOURCE_CALCULATED_FROM_DIMENSIONS,
    SOURCE_CUSTOMER,
    SOURCE_INFERRED_FROM_SCENE,
    SOURCE_RECOMMENDED_BY_ENGINE,
    SOURCE_RETRIEVED_FROM_PRODUCT_KB,
    GroundedFact,
)
from src.models.requirement import RequirementProfile  # noqa: E402
from src.core.requirement_extractor import RequirementExtractor  # noqa: E402
from src.rag.pitch_resolution import (  # noqa: E402
    EXACT,
    NEAREST_AVAILABLE,
    OUTSIDE_VALID_WINDOW,
    resolve_pitch,
)


def _profile(**slots) -> RequirementProfile:
    return RequirementProfile.from_slots(slots, explicit_keys=set(slots))


class _FakeModel:
    def __init__(self, name: str, pitch: float):
        self.model = name
        self.pixel_pitch_mm = pitch


RENTAL_PITCHES = [1.95, 2.6, 2.976, 3.91, 4.81]


# ── Environment（计划 §8 / §13）───────────────────────────────────────────
class TestEnvironmentQuestion:

    def test_environment_is_first_missing_question(self):
        profile = _profile(display_type="LED")
        plan = next_question_plan(profile, session_id="env-1")
        assert plan is not None
        assert plan.slot == "environment"
        assert plan.reason == "environment_hard_gate"

    def test_inferred_environment_skips_environment_question(self):
        """明显场景（教堂）推断出室内 → 不再问室内外（业务规则保留）。"""
        profile = RequirementExtractor().extract("It's for a church", use_llm=False)
        assert profile.environment == "indoor"
        assert profile.sources.get("environment") == "scenario_derived"
        plan = next_question_plan(profile, session_id="env-2")
        assert plan is None or plan.slot != "environment"

    def test_confirmed_environment_skips_environment_question(self):
        profile = _profile(display_type="LED", environment="indoor")
        plan = next_question_plan(profile, session_id="env-3")
        assert plan is None or plan.slot != "environment"

    def test_environment_gate_beats_random_pool(self):
        """哪怕其它槽位都还空着，第一问也必须是室内外。"""
        profile = _profile(display_type="LED", target_width_mm=3000, target_height_mm=5000)
        for session in ("env-4a", "env-4b", "env-4c"):
            plan = next_question_plan(profile, session_id=session)
            assert plan is not None and plan.slot == "environment", session

    def test_environment_not_asked_twice_when_customer_answers_something_else(self):
        """实测 bug：客户回 "3*5"（没答室内外）→ 系统又把室内外问了一遍。

        正确行为（v2.4 既定规则）：答非所问 → 换下一个问题，
        没答的那一项留到"其它问题问完"的硬性条件复问。
        """
        profile = _profile(display_type="LED", target_width_mm=3000, target_height_mm=5000)
        first = next_question_plan(profile, session_id="env-5")
        assert first is not None and first.slot == "environment"
        profile.record_ask("environment")
        profile.last_asked_slot = "environment"

        second = next_question_plan(profile, session_id="env-5")
        assert second is not None
        assert second.slot != "environment", "刚问过没答 → 这一轮换别的问"
        # 而且换的是"更容易/降门槛"以外的正常问法
        assert second.easier is False

    def test_environment_returns_in_hard_recap_with_easier_wording(self):
        """其它问题都问完 → 回头用降门槛的问法再问一次室内外。"""
        from src.dialogue import ASK_POOL, HARD_SLOTS

        profile = _profile(display_type="LED")
        for slot in ASK_POOL:
            profile.record_ask(slot)
        profile.ask_counts["environment"] = 1
        profile.last_asked_slot = "size"
        plan = next_question_plan(profile, session_id="env-6")
        assert plan is not None
        assert plan.slot == "environment"
        assert plan.easier is True
        assert plan.slot in HARD_SLOTS


# ── Single Question（计划 §3 / §13）───────────────────────────────────────
class TestSingleQuestionGuard:

    def test_one_customer_turn_one_question(self):
        guard = FinalResponseGuard()
        result = guard.finalize(
            "Is this indoor or outdoor? What sort of application will it be used in?"
        )
        assert result.text.count("?") == 1
        assert "multiple_questions_collapsed" in result.issues
        assert len(result.questions_dropped) == 1

    def test_multiple_internal_questions_only_one_output(self):
        """内部两个节点各带一个问题 → 只保留优先级高的（硬性 Gate）。"""
        guard = FinalResponseGuard()
        result = guard.finalize(
            "Happy to help. What budget level are you thinking about? "
            "Will the screen be installed indoors or outdoors?",
            questions=[
                {"text": "What budget level are you thinking about?", "slot": "budget"},
                {"text": "Will the screen be installed indoors or outdoors?", "slot": "environment"},
            ],
        )
        assert result.text.count("?") == 1
        assert "indoors or outdoors" in result.text, "应保留硬性 Gate 的那一个问题"
        assert "budget" not in result.text

    def test_response_coordinator_enforces_single_question(self):
        coordinator = ResponseCoordinator()
        text = coordinator.finalize(
            "Is this indoor or outdoor? And what size do you need?",
            session_id="guard-1",
            message="i need a screen",
        )
        assert text.count("?") == 1
        assert coordinator.last_guard is not None

    def test_internal_terms_never_reach_the_customer(self):
        guard = FinalResponseGuard()
        result = guard.finalize(
            "Your RequirementProfile says the size is missing. What size do you need?"
        )
        assert "RequirementProfile" not in result.text
        assert "internal_term_sentence_dropped" in result.issues
        assert result.text.count("?") == 1

    def test_extra_bubbles_cannot_add_a_second_question(self):
        """实测 bug：主回复问"室内还是室外"，附加气泡又问"用在什么场景" —— 两个气泡两个问题。

        现在附加气泡也要过 Guard：主回复已经带问题了，附加气泡里的问题就不再发。
        """
        guard = FinalResponseGuard()
        response = "Should I look at indoor or outdoor displays for you?"
        extras = [
            "What sort of application will this screen be used in?",
            "By the way, we ship from Shenzhen.",
        ]
        kept = guard.guard_extras(response, extras)
        assert len(kept) == 1
        assert kept[0] == "By the way, we ship from Shenzhen."

    def test_coordinator_guards_extras(self):
        coordinator = ResponseCoordinator()
        kept = coordinator.guard_extras(
            "Is this indoors or outdoors?",
            ["And what size do you need?"],
        )
        assert kept == []


# ── Pitch Resolution（计划 §4 / §13）──────────────────────────────────────
class TestPitchResolution:

    def test_exact_pitch(self):
        profile = _profile(environment="indoor", pixel_pitch_mm=3.0)
        resolution = resolve_pitch(
            profile, _FakeModel("TW11-3216-P3.0", 3.076), available_pitches=[2.5, 3.076, 4.0]
        )
        assert resolution.match_type == EXACT
        assert resolution.requested_text == "P3" and resolution.resolved_text == "P3"
        assert resolution.needs_explanation is False

    def test_nearest_available_pitch(self):
        profile = _profile(environment="indoor", installation="rental", pixel_pitch_mm=3.0)
        resolution = resolve_pitch(
            profile, _FakeModel("TW11-IR-P2.9", 2.976), available_pitches=RENTAL_PITCHES
        )
        assert resolution.match_type == NEAREST_AVAILABLE
        assert resolution.needs_explanation is True

    def test_pitch_mismatch_has_reason(self):
        profile = _profile(environment="indoor", installation="rental", pixel_pitch_mm=3.0)
        resolution = resolve_pitch(
            profile, _FakeModel("TW11-IR-P2.9", 2.976), available_pitches=RENTAL_PITCHES
        )
        assert resolution.reason
        explanation = resolution.explain()
        assert "P2.9" in explanation and "P3" in explanation

    def test_pitch_far_from_request_is_flagged(self):
        profile = _profile(environment="indoor", installation="rental", pixel_pitch_mm=3.0)
        resolution = resolve_pitch(
            profile, _FakeModel("TW11-IR-P4.8", 4.81), available_pitches=RENTAL_PITCHES
        )
        assert resolution.match_type == OUTSIDE_VALID_WINDOW
        assert resolution.needs_explanation is True

    def test_requested_pitch_not_equal_resolved_pitch_without_explanation(self):
        """最终话术里：requested 与 resolved 不一致时必须给出解释。"""
        pitch_resolution = {
            "requested_pitch_text": "P3",
            "resolved_pitch_text": "P2.9",
            "pitch_match_type": NEAREST_AVAILABLE,
            "needs_explanation": True,
        }
        bad = validate_response(
            "P3 it is. We'd use TW11-IR-P2.9 for this one.",
            pitch_resolution=pitch_resolution,
        )
        assert bad.pitch_mismatch_unexplained is True
        assert "pitch_mismatch_not_explained" in bad.issues

        good = validate_response(
            "P3 isn't an exact option in this range, but P2.9 is the closest available match.",
            pitch_resolution=pitch_resolution,
        )
        assert good.pitch_mismatch_unexplained is False
        assert "pitch_mismatch_not_explained" not in good.issues


# ── Grounded Facts（计划 §5 / §11.2 / §13）────────────────────────────────
class TestGroundedFacts:

    def _facts(self):
        return [
            GroundedFact(field="environment", value="indoor", source=SOURCE_CUSTOMER),
            GroundedFact(field="purpose", value="church", source=SOURCE_INFERRED_FROM_SCENE),
            GroundedFact(field="cabinet_size", value="640x480mm", source=SOURCE_RETRIEVED_FROM_PRODUCT_KB),
            GroundedFact(field="cabinet_count", value=24, source=SOURCE_CALCULATED_FROM_DIMENSIONS),
            GroundedFact(
                field="recommended_model", value="TW11-3216-P3.0",
                source=SOURCE_RECOMMENDED_BY_ENGINE,
            ),
        ]

    @pytest.mark.parametrize("source", [
        SOURCE_CUSTOMER,
        SOURCE_INFERRED_FROM_SCENE,
        SOURCE_CALCULATED_FROM_DIMENSIONS,
        SOURCE_RETRIEVED_FROM_PRODUCT_KB,
        SOURCE_RECOMMENDED_BY_ENGINE,
    ])
    def test_all_allowed_sources_are_accepted(self, source):
        """客户说的 / 推断的 / 算出来的 / 产品库的 / 引擎推荐的 —— 全部都允许。"""
        fact = GroundedFact(field="environment", value="indoor", source=source)
        assert fact.grounded is True

    def test_unknown_business_fact_blocked(self):
        """Context 里没有"安装方式"事实 → LLM 不能说"这是租赁款、安装快"。"""
        result = validate_response(
            "This rental model is quick to install and ideal for church events.",
            grounded_facts=self._facts(),
        )
        assert "rental" in result.ungrounded_facts
        assert "ungrounded_business_fact" in result.issues

    def test_grounded_installation_is_allowed(self):
        facts = self._facts() + [
            GroundedFact(field="installation", value="fixed", source=SOURCE_CUSTOMER)
        ]
        result = validate_response(
            "Since this is a fixed installation, we'd anchor it to the wall.",
            grounded_facts=facts,
        )
        assert "installation" not in result.ungrounded_facts

    def test_build_grounded_facts_marks_sources(self):
        # 用真实抽取链路：客户只说了场景（church）→ 环境由场景推断
        profile = RequirementExtractor().extract(
            "It's for a church, about 5m away, the wall is 4m x 2m",
            use_llm=False,
        )
        facts = build_grounded_facts(
            profile=profile,
            calculations={"cabinet_count": 24, "actual_width_mm": 3840},
        )
        by_field = {fact.field: fact.source for fact in facts}
        assert by_field.get("environment") == SOURCE_INFERRED_FROM_SCENE
        assert by_field.get("cabinet_count") == SOURCE_CALCULATED_FROM_DIMENSIONS

    def test_pitch_inferred_from_distance_is_marked(self):
        """视距 → 点间距的推断结果要标来源（计划 §9.2：推断也要进结构化状态）。"""
        profile = _profile(display_type="LED", viewing_distance_m=5)
        profile.pixel_pitch_mm = 3.0
        profile.sources["pixel_pitch_mm"] = "inferred"
        by_field = {fact.field: fact.source for fact in build_grounded_facts(profile=profile)}
        assert by_field.get("viewing_distance") == SOURCE_CUSTOMER
        assert by_field.get("pixel_pitch") == "inferred_from_distance"

    def test_grounded_facts_are_visible_to_the_llm_context(self):
        context = build_context(
            action="ASK",
            customer_message="indoor church 4x2m",
            question="How far will the audience be from the screen?",
            grounded_facts=build_grounded_facts(
                profile=RequirementExtractor().extract("It's for a church", use_llm=False)
            ),
        )
        block = context.prompt_block()
        assert "Grounded facts" in block
        assert "inferred_from_scene" in block


# ── Numeric Consistency（计划 §11.5 / §13）────────────────────────────────
class TestNumericConsistency:

    FACTS = [
        GroundedFact(field="cabinet_size", value="640x480mm", source=SOURCE_RETRIEVED_FROM_PRODUCT_KB),
        GroundedFact(field="cabinet_count", value=24, source=SOURCE_CALCULATED_FROM_DIMENSIONS),
        GroundedFact(field="module_count", value=144, source=SOURCE_CALCULATED_FROM_DIMENSIONS),
        GroundedFact(field="actual_screen_width_mm", value=3840, source=SOURCE_CALCULATED_FROM_DIMENSIONS),
    ]

    def test_screen_dimensions_consistent(self):
        result = validate_response(
            "That's 24 cabinets, giving an actual width of 3840 mm.",
            grounded_facts=self.FACTS,
            customer_message="we need 4x2m",
        )
        assert result.ungrounded_numbers == []

    def test_screen_dimension_invented_is_blocked(self):
        result = validate_response(
            "The actual width would be 5120 mm.",
            grounded_facts=self.FACTS,
            customer_message="we need 4x2m",
        )
        assert result.ungrounded_numbers
        assert "ungrounded_numeric" in result.issues

    def test_cabinet_and_module_counts_consistent(self):
        result = validate_response(
            "24 cabinets and 144 modules in total.",
            grounded_facts=self.FACTS,
        )
        assert "ungrounded_numeric" not in result.issues

    def test_pitch_consistent(self):
        facts = self.FACTS + [
            GroundedFact(
                field="pixel_pitch", value="P3", source=SOURCE_RECOMMENDED_BY_ENGINE
            )
        ]
        ok = validate_response("We'd go with the P3 model.", grounded_facts=facts)
        assert ok.unsupported_parameters == []
        bad = validate_response(
            "We'd go with the P5 model.", grounded_facts=facts,
            supported_parameters=["P3"],
        )
        assert bad.unsupported_parameters == ["P5"]
