"""
《全量需求捕获与 Unknown 容错优化》回归测试（Phase 1~23）。

覆盖计划第 20 节的 7 个用例，以及 DEGRADED_READY 推荐 / 计算不被 unknown 阻塞 /
Session Switch 不继承 unknown 状态。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.unknown_detector import detect_no_answer  # noqa: E402
from src.models.requirement import (  # noqa: E402
    MAX_ASKS_PER_SLOT,
    RequirementProfile,
)
from src.rag.readiness import (  # noqa: E402
    check_calculation_ready,
    check_recommendation_ready,
)


@pytest.fixture
def sales_llm(monkeypatch):
    """把需求抽取链路上的 LLM 全部换成空实现（本轮只验证规则 + 状态机）。"""
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
    # 语义补充抽取不再走网络：规则解析已经能覆盖本文件的全部场景
    monkeypatch.setattr(
        extractor_mod.RequirementExtractor,
        "_llm_semantic_extract",
        lambda self, message, rule_slots, session_id="": {},
    )
    extractor_mod.RequirementExtractor._semantic_cache.clear()
    return sales_req


def _turn(sales_req, message, profile=None, requirements=None, last_asked=""):
    """驱动一轮 requirement_mining。"""
    if profile is not None:
        profile.last_asked_slot = last_asked or profile.last_asked_slot
    state = {
        "messages": [{"role": "user", "content": message}],
        "current_message": message,
        "session_id": "unknown-tolerance-session",
        "intent": "need_query",
        "next_action": "ask",
        "requirements": dict(requirements or {}),
        "additional_requirements": [],
        "should_generate_solution": False,
        "response": "",
        "pending_question": "",
        "pending_slot": "",
    }
    if profile is not None:
        state["requirement_profile"] = profile
    return sales_req.requirement_mining(state)


def _reach_viewing_distance_question(sales_req, first_message="we need an indoor led screen for a church"):
    """推进到"AI 正在问观看距离"的那一轮。

    环境 + 场景已经由客户原话确定后，安装方式（HIGH 优先级）会被先问到；
    本助手把这类前置问题用"给答案"的方式走完，再开始观察观看距离的行为。
    """
    turn = _turn(sales_req, first_message, None, {})
    guard = 0
    while turn["pending_slot"] != "viewing_distance" and guard < 4:
        guard += 1
        turn = _turn(
            sales_req,
            "it's a fixed installation",
            turn["requirement_profile"],
            {},
        )
    assert turn["pending_slot"] == "viewing_distance", turn["pending_slot"]
    return turn


class TestUnknownDetection:

    @pytest.mark.parametrize("text", [
        "I don't know", "I don't know yet", "I have no idea", "I'm not sure",
        "Not sure", "No idea", "I don't have that information", "I can't tell",
        "I can't estimate", "hard to say",
        "不知道", "不清楚", "不确定", "没了解", "没有这个信息", "暂时不知道",
        "不太清楚", "无法确定", "没法估计",
    ])
    def test_dont_know(self, text):
        assert detect_no_answer(text) == "customer_does_not_know", text

    @pytest.mark.parametrize("text", [
        "let's skip it", "skip the viewing distance", "don't ask", "not available",
        "这个不用了", "这个先跳过", "这个没有",
    ])
    def test_skip(self, text):
        assert detect_no_answer(text) == "customer_skip", text

    def test_normal_answer_is_not_unknown(self):
        assert detect_no_answer("about 8 meters") is None
        assert detect_no_answer("indoor conference room") is None


class TestSlotStateMachine:
    """Phase 1：missing / unknown_pending / unknown / confirmed。"""

    def test_missing_to_confirmed(self):
        profile = RequirementProfile.from_slots({"viewing_distance_m": 8}, explicit_keys={"viewing_distance_m"})
        assert profile.slot_status("viewing_distance") == "confirmed"

    def test_missing_to_unknown(self):
        profile = RequirementProfile()
        assert profile.slot_status("viewing_distance") == "missing"
        profile.record_ask("viewing_distance")
        assert profile.slot_status("viewing_distance") == "unknown_pending"
        profile.record_ask("viewing_distance")
        assert profile.slot_status("viewing_distance") == "unknown"
        assert "viewing_distance" in profile.unknown_slots()

    def test_unknown_to_confirmed_later(self):
        profile = RequirementProfile()
        profile.record_ask("viewing_distance")
        profile.record_ask("viewing_distance")
        profile.mark_unknown("viewing_distance")
        assert profile.slot_status("viewing_distance") == "unknown"

        profile = profile.merge(
            RequirementProfile.from_slots({"viewing_distance_m": 10}, explicit_keys={"viewing_distance_m"})
        )
        assert profile.slot_status("viewing_distance") == "confirmed"
        assert profile.unknown_slots() == []
        assert profile.viewing_distance_m == pytest.approx(10.0)

    def test_skip_is_immediately_unknown(self):
        profile = RequirementProfile()
        profile.record_ask("pixel_pitch")
        profile.mark_unknown("pixel_pitch", "customer_skip")
        assert profile.slot_status("pixel_pitch") == "unknown"
        assert profile.ask_count("pixel_pitch") >= 1
        assert profile.unknown_reasons["pixel_pitch"] == "customer_skip"

    def test_first_dont_know_is_still_askable(self):
        """Phase 6：客户第一次说不知道 → unknown_pending（还可以降门槛再问一次）。"""
        profile = RequirementProfile()
        profile.record_ask("viewing_distance")
        profile.mark_unknown("viewing_distance")
        assert profile.ask_count("viewing_distance") == 1
        assert profile.slot_status("viewing_distance") == "unknown_pending"
        assert profile.is_unknown("viewing_distance") is False
        # Phase 10：客户后来主动补上 → unknown 自动解除
        profile = profile.merge(
            RequirementProfile.from_slots(
                {"viewing_distance_m": 9}, explicit_keys={"viewing_distance_m"}
            )
        )
        assert profile.slot_status("viewing_distance") == "confirmed"
        assert profile.unknown_reasons.get("viewing_distance") is None


class TestPlanTestCases:
    """计划第 20 节的 7 个用例。"""

    def test_1_normal_answer(self, sales_llm):
        asked = _reach_viewing_distance_question(sales_llm)

        answered = _turn(
            sales_llm, "about 8 meters", asked["requirement_profile"], {},
            last_asked="viewing_distance",
        )
        profile = answered["requirement_profile"]
        assert profile.viewing_distance_m == pytest.approx(8.0)
        assert profile.slot_status("viewing_distance") == "confirmed"

    def test_2_unknown_once(self, sales_llm):
        asked = _reach_viewing_distance_question(sales_llm)
        turn = _turn(
            sales_llm, "I don't know", asked["requirement_profile"], {},
            last_asked="viewing_distance",
        )
        profile = turn["requirement_profile"]
        # 客户第一次说不知道：记下原因，而且**不放弃**这一项
        assert profile.unknown_reasons["viewing_distance"] == "customer_does_not_know"
        assert profile.ask_count("viewing_distance") <= MAX_ASKS_PER_SLOT
        # 第二次问同一个槽位 → 降低门槛的问法（给区间 / 二选一），不是机械重复
        assert turn["pending_slot"] == "viewing_distance"
        question = turn["pending_question"].lower()
        assert any(word in question for word in ("close", "metres", "meters", "10"))
        assert question != asked["pending_question"].lower()

    def test_3_unknown_twice_then_stop_asking(self, sales_llm):
        first = _reach_viewing_distance_question(sales_llm)
        second = _turn(sales_llm, "I don't know", first["requirement_profile"], {}, last_asked="viewing_distance")
        third = _turn(sales_llm, "still don't know", second["requirement_profile"], {}, last_asked="viewing_distance")

        profile = third["requirement_profile"]
        assert profile.slot_status("viewing_distance") == "unknown"
        assert profile.ask_count("viewing_distance") == MAX_ASKS_PER_SLOT
        # 不再追问同一个字段
        assert third["pending_slot"] != "viewing_distance"
        # 问满两次后按已有信息继续（DEGRADED_READY），不再卡在需求收集
        assert third["recommendation_gate"]["status"] == "DEGRADED_READY"
        assert "viewing_distance" in third["recommendation_gate"]["unknown_slots"]

    def test_4_answer_other_information_while_not_knowing(self, sales_llm):
        first = _reach_viewing_distance_question(sales_llm)
        turn = _turn(
            sales_llm,
            "I don't know the viewing distance, but the screen is 5m x 3m",
            first["requirement_profile"],
            {},
            last_asked="viewing_distance",
        )
        profile = turn["requirement_profile"]
        # 不知道的字段 → unknown；同时说出的尺寸必须被记录（信息不丢）
        assert profile.unknown_reasons["viewing_distance"] == "customer_does_not_know"
        assert profile.target_width_mm == pytest.approx(5000.0)
        assert profile.target_height_mm == pytest.approx(3000.0)
        # 已经拿到值的字段不会被再次追问
        assert profile.slot_status("size") == "confirmed"

    def test_5_one_message_multiple_fields(self, sales_llm):
        turn = _turn(
            sales_llm,
            "Outdoor stadium screen, 8m x 5m, viewing distance around 15m",
            None,
            {},
        )
        profile = turn["requirement_profile"]
        assert profile.environment == "outdoor"
        assert profile.purpose == "stadium"
        assert profile.target_width_mm == pytest.approx(8000.0)
        assert profile.target_height_mm == pytest.approx(5000.0)
        assert profile.viewing_distance_m == pytest.approx(15.0)

    def test_6_later_addition_flips_unknown_to_confirmed(self, sales_llm):
        profile = RequirementProfile()
        profile.record_ask("viewing_distance")
        profile.record_ask("viewing_distance")
        profile.mark_unknown("viewing_distance")

        turn = _turn(
            sales_llm, "Actually, it's about 10 meters", profile, {},
            last_asked="viewing_distance",
        )
        result = turn["requirement_profile"]
        assert result.viewing_distance_m == pytest.approx(10.0)
        assert result.slot_status("viewing_distance") == "confirmed"

    def test_7_customer_skips(self, sales_llm):
        first = _reach_viewing_distance_question(sales_llm)
        turn = _turn(
            sales_llm, "let's skip the viewing distance",
            first["requirement_profile"], {}, last_asked="viewing_distance",
        )
        profile = turn["requirement_profile"]
        assert profile.slot_status("viewing_distance") == "unknown"
        assert profile.unknown_reasons["viewing_distance"] == "customer_skip"
        assert profile.ask_count("viewing_distance") <= MAX_ASKS_PER_SLOT
        assert turn["pending_slot"] != "viewing_distance"


class TestDegradedRecommendation:
    """Phase 11~16：unknown 不再阻塞推荐 / 计算。"""

    def _degraded_profile(self):
        slots = {
            "display_type": "LED",
            "environment": "indoor",
            "purpose": "conference",
            "installation": "fixed",
            "target_width_mm": 5000,
            "target_height_mm": 3000,
        }
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        profile.record_ask("viewing_distance")
        profile.record_ask("viewing_distance")
        profile.mark_unknown("viewing_distance")
        return profile

    def test_gate_returns_degraded_ready(self):
        decision = check_recommendation_ready(self._degraded_profile())
        assert decision.ready is True
        assert decision.status == "DEGRADED_READY"
        assert "viewing_distance" in decision.unknown_slots

    def test_recommendation_still_produced(self):
        from src.rag.recommendation_service import RecommendationService

        result = RecommendationService().recommend(self._degraded_profile())
        assert result["recommendation_status"] in ("RECOMMENDED", "DEGRADED")
        assert result["recommendations"], "unknown 不应阻止推荐"

    def test_calculation_not_blocked_by_unknown_distance(self):
        decision = check_calculation_ready(self._degraded_profile())
        assert decision.ready is True, decision.missing

    def test_continue_asking_when_not_yet_asked_twice(self):
        slots = {"display_type": "LED", "environment": "indoor", "purpose": "conference"}
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        decision = check_recommendation_ready(profile)
        assert decision.ready is False
        assert decision.status == "CONTINUE_ASKING"


class TestSessionSwitchClearsUnknownState:
    """Phase 18：新项目不能继承旧项目的 ask_count / unknown 状态。"""

    def test_ask_bookkeeping_survives_one_turn(self):
        """提问记账必须跨轮保留 —— 否则"同一字段最多问两次"形同虚设。"""
        from src.memory.store import memory

        session_id = "unknown-ask-bookkeeping"
        memory.clear(session_id)
        try:
            profile = RequirementProfile()
            profile.record_ask("environment")
            memory.set_requirement_profile(session_id, profile)

            restored = memory.get_requirement_profile(session_id)
            assert restored is not None
            assert restored["ask_counts"] == {"environment": 1}
        finally:
            memory.clear(session_id)

    def test_reset_requirement_state_clears_unknown(self):
        from src.memory.store import memory

        session_id = "unknown-session-switch"
        memory.clear(session_id)
        try:
            profile = RequirementProfile.from_slots(
                {"environment": "indoor", "purpose": "conference"},
                explicit_keys={"environment", "purpose"},
            )
            profile.record_ask("viewing_distance")
            profile.record_ask("viewing_distance")
            profile.mark_unknown("viewing_distance")
            memory.set_requirement_profile(session_id, profile)
            memory.set_requirements(session_id, {"usage": "conference"})

            memory.reset_requirement_state(session_id)

            assert memory.get_requirement_profile(session_id) is None
            assert memory.get_requirements(session_id) == {}
        finally:
            memory.clear(session_id)


def _degraded_profile():
    """室内 + 教堂 + 固装 + 5m x 3m，观看距离客户不知道。"""
    slots = {
        "display_type": "LED",
        "environment": "indoor",
        "purpose": "church",
        "installation": "fixed",
        "target_width_mm": 5000,
        "target_height_mm": 3000,
    }
    profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
    profile.record_ask("viewing_distance")
    profile.record_ask("viewing_distance")
    profile.mark_unknown("viewing_distance")
    return profile


class TestPhase13EngineUnknownSkip:
    """Phase 13：unknown != 0，未知维度直接跳过评分，不拉低所有候选。"""

    def test_unknown_viewing_distance_skips_pitch_scoring(self):
        from src.rag.recommendation_engine import RecommendationEngine

        result = RecommendationEngine().recommend(
            profile=_degraded_profile(), top_k=3, require_ready=True
        )
        assert result["recommendation_status"] == "DEGRADED"
        assert result["recommendations"], "客户不知道视距不代表不能推荐"
        for rec in result["recommendations"]:
            # 未知维度必须是 None（跳过），不能是 0.0（记零分）
            assert rec["breakdown"]["pitch"] is None, rec
            assert rec["score"] > 0

    def test_known_viewing_distance_scores_pitch(self):
        from src.rag.recommendation_engine import RecommendationEngine

        profile = _degraded_profile()
        profile.viewing_distance_m = 5.0
        profile.sources["viewing_distance_m"] = "explicit"
        result = RecommendationEngine().recommend(profile=profile, top_k=3, require_ready=True)
        assert result["recommendation_status"] == "RECOMMENDED"
        assert any(rec["breakdown"]["pitch"] is not None for rec in result["recommendations"])


class TestPhase14RecommendationBasis:
    """Phase 14：推荐结果要能说清"凭什么是它"。"""

    def test_service_reports_degraded_basis(self):
        from src.rag.recommendation_service import RecommendationService

        result = RecommendationService().recommend(_degraded_profile())
        assert result["recommendation_status"] == "DEGRADED"
        basis = result["recommendation_basis"]
        assert basis["recommendation_status"] == "DEGRADED"
        assert "environment" in basis["confirmed_requirements"]
        assert "purpose" in basis["confirmed_requirements"]
        assert "width" in basis["confirmed_requirements"]
        assert basis["unknown_requirements"] == ["viewing_distance"]
        assert result["unknown_requirements"] == ["viewing_distance"]
        assert result["missing_fields"] == []

    def test_normal_recommendation_is_not_degraded(self):
        from src.rag.recommendation_service import RecommendationService

        slots = {
            "display_type": "LED",
            "environment": "indoor",
            "purpose": "church",
            "installation": "fixed",
            "viewing_distance_m": 5,
        }
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        result = RecommendationService().recommend(profile)
        assert result["recommendation_status"] == "RECOMMENDED"
        assert result["recommendation_basis"]["unknown_requirements"] == []


class TestPhase15DegradedWording:
    """Phase 15：缺什么 + 影响什么，且绝不说"无法推荐"。"""

    def test_note_mentions_missing_and_impact(self):
        from src.rag.reply_composer import degraded_note, has_no_product_phrase

        note = degraded_note(["viewing_distance"], "en")
        assert "viewing distance" in note.lower()
        assert "pixel pitch" in note.lower()
        assert not has_no_product_phrase(note)
        zh_note = degraded_note(["viewing_distance"], "zh")
        assert "观看距离" in zh_note

    def test_note_is_appended_when_llm_forgets(self):
        from src.agents.solution.nodes.recommend import _ensure_degraded_note

        answer = "TW11-3216-P2.5 is the best fit for your church."
        patched = _ensure_degraded_note(answer, ["viewing_distance"], "en")
        assert "viewing distance" in patched.lower()
        # 已经说明过就不重复追加
        assert _ensure_degraded_note(patched, ["viewing_distance"], "en") == patched


class TestPhase16DegradedRecommendationEndToEnd:
    """Phase 16：推荐链路端到端 —— 仍出产品、仍算箱体、话术说明缺失项。"""

    class _BoomLLM:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("offline")

    class _TerseLLM:
        class _R:
            content = "TW11-3216-P2.5 is the best fit for your project."

        def invoke(self, *args, **kwargs):
            return self._R()

    def _state(self, profile):
        return {
            "requirement": {},
            "products": [],
            "messages": [{"role": "user", "content": "indoor LED screen for a church"}],
            "requirement_profile": profile,
            "understood_language": "en",
            "additional_requirements": [],
        }

    def test_node_recommends_with_unknown_distance(self, monkeypatch):
        import importlib

        rec_mod = importlib.import_module("src.agents.solution.nodes.recommend")
        monkeypatch.setattr(rec_mod, "get_llm", lambda *a, **k: self._TerseLLM())

        out = rec_mod.recommend_node(self._state(_degraded_profile()))

        assert out["products"], "客户不知道视距时仍应给出产品"
        answer = out["recommendation"]
        # Phase 15：话术必须点出缺失项及其影响
        assert "viewing distance" in answer.lower()
        # Phase 16：尺寸齐备 → 箱体/模组必须照常计算
        assert out["screen_calculation"], "视距 unknown 不应阻塞箱体计算"
        assert out["screen_calculation"]["cabinet_count"] > 0

    def test_fallback_template_also_states_the_gap(self, monkeypatch):
        import importlib

        rec_mod = importlib.import_module("src.agents.solution.nodes.recommend")
        monkeypatch.setattr(rec_mod, "get_llm", lambda *a, **k: self._BoomLLM())

        out = rec_mod.recommend_node(self._state(_degraded_profile()))

        answer = out["recommendation"].lower()
        assert "tw11" in answer
        assert "viewing distance" in answer
        assert not any(
            phrase in answer
            for phrase in ("cannot recommend", "insufficient", "not enough information")
        )


class TestPhase2GlobalExtraction:
    """Phase 2/3：每条客户消息都做全量提取，一条消息可更新多个字段。"""

    def _extract(self, message, previous=None):
        from src.core.requirement_extractor import RequirementExtractor

        return RequirementExtractor().extract(message, previous_profile=previous, use_llm=False)

    def test_one_message_updates_every_field(self):
        profile = self._extract(
            "It's an outdoor stadium screen, 8m x 5m, viewing distance is about 15 meters"
        )
        assert profile.environment == "outdoor"
        assert profile.purpose == "stadium"
        assert profile.target_width_mm == pytest.approx(8000.0)
        assert profile.target_height_mm == pytest.approx(5000.0)
        assert profile.viewing_distance_m == pytest.approx(15.0)

    def test_answer_other_field_is_not_dropped(self):
        """AI 问的是视距，客户答的是尺寸 —— 尺寸必须照常入库。"""
        profile = self._extract("the screen is 6m x 3m, it's for outdoor advertising")
        assert profile.target_width_mm == pytest.approx(6000.0)
        assert profile.target_height_mm == pytest.approx(3000.0)
        assert profile.environment == "outdoor"
        assert profile.purpose == "advertising"


class TestPhase17PlannerPriority:
    """Phase 17：Question Planner 不得再问已经 unknown 的字段。"""

    def _profile(self):
        slots = {
            "display_type": "LED",
            "environment": "indoor",
            "purpose": "church",
            "installation": "fixed",
        }
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        profile.record_ask("viewing_distance")
        profile.record_ask("viewing_distance")
        profile.mark_unknown("viewing_distance")
        return profile

    def test_unknown_slot_is_skipped(self):
        from src.agents.sales.question_planner import plan_next_question

        plan = plan_next_question(self._profile())
        if plan is not None:
            assert plan["slot"] != "viewing_distance_m"

    def test_plan_reports_field_priority(self):
        from src.agents.sales.question_planner import plan_next_question

        plan = plan_next_question(self._profile())
        if plan is not None:
            assert plan["priority"] in ("HIGH", "MEDIUM", "LOW")

    def test_missing_slots_excludes_unknown(self):
        from src.agents.sales.question_planner import missing_slots

        assert "viewing_distance_m" not in missing_slots(self._profile())


class TestGoldenConversation:
    """计划第 23 节：金标对话 —— 超过 2 次询问率必须为 0，且最终仍要推荐。"""

    def test_never_asks_more_than_twice_and_still_recommends(self, sales_llm):
        from collections import Counter

        turn = _turn(sales_llm, "we need an indoor led screen for a church", None, {})
        asked_slots = []
        for _ in range(12):
            if turn["should_generate_solution"]:
                break
            slot = turn["pending_slot"]
            assert slot, "未就绪时必须继续追问（但不能死循环）"
            asked_slots.append(slot)
            turn = _turn(
                sales_llm, "I don't know", turn["requirement_profile"], {},
                last_asked=slot,
            )

        counts = Counter(asked_slots)
        assert max(counts.values()) <= MAX_ASKS_PER_SLOT, counts
        profile = turn["requirement_profile"]
        assert profile.ask_count("installation") <= MAX_ASKS_PER_SLOT
        assert profile.ask_count("viewing_distance") <= MAX_ASKS_PER_SLOT
        # 客户两次都不知道 → 仍然基于已确认信息继续推荐
        assert turn["should_generate_solution"] is True
        assert turn["recommendation_gate"]["status"] == "DEGRADED_READY"
