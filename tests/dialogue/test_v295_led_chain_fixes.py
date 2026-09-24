"""LED 链路两个实测问题（2026-09-24 客户会话）。

问题 1（答"led"却走了自由问答）

    客户: led
    AI  : LED it is, that's our core line… Let's take a slightly different angle —
          if one of the requirements can be relaxed (for example the pixel pitch,
          the screen size, or the viewing distance), I can match a model for you
          right away. That said, what pixel pitch do you have in mind…?

    "relax 一个条件我就能给你配型号" 是**匹配不到产品时**的话术
    （`rag/reply_composer.relaxation_answer`）。它出现在这里，是因为
    classify 把"led"这个**回答**误判成 `product_question`，整轮被送进 Solution 的
    自由问答（RAG）链路，那边给不出答案就用了放宽话术。

问题 2（给了 P4 之后又问观看距离）

    客户: p4 → … → 客户: permanent → AI: Roughly how far will viewers be from the screen?

    点间距与观看距离是**一对**（计划：先问 P 值，P 值不知道才用视距反推）。
    Gate 的 ActionPlanner 已经把它标成 skip（日志可见 `'viewing_distance': 'skip'`），
    但 Dialogue Policy 的候选列表只调了 `field_action`，**没有跑跨槽位规则**
    （`apply_cross_slot_rules`），于是把 viewing_distance 当候选排到了 size 前面，
    收口层的"以 Policy 为准"再把它强制问出去。
"""
import os
import sys
from types import SimpleNamespace

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _led_profile_with_pitch(pitch_mm: float = 4.0):
    from src.models.requirement import RequirementProfile

    slots = {"display_type": "LED", "pixel_pitch_mm": pitch_mm}
    return RequirementProfile.from_slots(slots, explicit_keys=set(slots))


class TestPitchAndViewingDistanceAreOnePair:
    """问题 2：有点间距就不要再问观看距离（Gate 与 Policy 必须同一套判断）。"""

    def test_policy_candidates_exclude_viewing_distance_when_pitch_is_known(self):
        from src.dialogue.policy import question_candidates

        profile = _led_profile_with_pitch(4.0)
        slots = [slot for slot, _score in question_candidates(profile, session_id="pitch-pair")]

        assert "pixel_pitch" not in slots, "点间距已经给了，不该再问"
        assert "viewing_distance" not in slots, (
            "客户给了 P 值 → 观看距离不再问（跨槽位规则）"
        )

    def test_policy_candidates_do_offer_the_distance_when_pitch_is_unknown(self):
        """对照：点间距还不知道时，视距是可以问的（用于反推点间距）。"""
        from src.dialogue.policy import question_candidates
        from src.models.requirement import RequirementProfile

        profile = RequirementProfile.from_slots(
            {"display_type": "LED"}, explicit_keys={"display_type"}
        )
        slots = [slot for slot, _score in question_candidates(profile, session_id="pitch-open")]

        assert "pixel_pitch" in slots

    def test_next_question_is_not_the_distance_after_pitch(self):
        from src.dialogue.policy import select_next_question

        picked = select_next_question(_led_profile_with_pitch(4.0), session_id="pitch-next")

        assert picked is not None
        assert picked[0] != "viewing_distance", picked

    def test_orchestrator_policy_candidates_match_the_gate(self, monkeypatch):
        """收口层用的那份候选（ActionConsistency 的依据）也要排除视距。"""
        from src.memory.store import memory
        from src.orchestrator import DualAgentOrchestrator

        session_id = "pitch-pair-orch"
        memory.clear(session_id)
        try:
            memory.set_requirement_profile(session_id, _led_profile_with_pitch(4.0))

            class _Sales:
                def run(self, *a, **k):
                    return {}

            class _Solution:
                def run(self, *a, **k):
                    return {}

            orchestrator = DualAgentOrchestrator(
                sales_agent=_Sales(), solution_agent=_Solution()
            )
            monkeypatch.setattr(orchestrator, "memory_store", memory)

            slots = orchestrator._ranked_question_candidates(session_id)
            assert "viewing_distance" not in slots, slots
        finally:
            memory.clear(session_id)

    def test_parked_distance_is_skipped_once_pitch_is_known(self):
        """客户说过"不知道"的视距（ASK_LATER）在点间距确定后也必须停问。

        实测链路（2026-09-24 第二份日志）：

            AI 问视距 → 客户答 "p4"（答的是点间距）→ 视距被 park 成 ASK_LATER
            → 客户又给了尺寸 → 点间距已知，可视距还是候选 → Policy 又把它问了一遍

        原因：``apply_cross_slot_rules`` 只处理 ASK / ASK_EASIER / DEFER…，
        **漏了 ASK_LATER**，于是"点间距已知就不再问视距"这条规则没生效。
        """
        from src.models.requirement import RequirementProfile
        from src.rag.field_policy import (
            ASK_LATER,
            RECOMMENDATION_SLOTS,
            SKIP,
            apply_cross_slot_rules,
            field_action,
        )

        slots = {"display_type": "LED", "environment": "indoor", "pixel_pitch_mm": 4.0}
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        profile.record_ask("viewing_distance")     # 问过一次
        profile.mark_unknown("viewing_distance")   # 客户没答上 → parked（ASK_LATER）

        actions = {slot: field_action(profile, slot) for slot in RECOMMENDATION_SLOTS}
        assert actions["pixel_pitch"] == "use", actions["pixel_pitch"]
        assert actions["viewing_distance"] == ASK_LATER, actions["viewing_distance"]

        apply_cross_slot_rules(profile, actions)

        assert actions["viewing_distance"] == SKIP, (
            "点间距已经确定 → 视距必须变成 skip（不管它之前是 ask 还是 ask_later）"
        )

    def test_policy_candidates_exclude_a_parked_distance_once_pitch_is_known(self):
        from src.dialogue.policy import question_candidates
        from src.models.requirement import RequirementProfile

        slots = {"display_type": "LED", "environment": "indoor", "pixel_pitch_mm": 4.0}
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        profile.record_ask("viewing_distance")
        profile.mark_unknown("viewing_distance")

        listed = [slot for slot, _ in question_candidates(profile, session_id="parked")]

        assert "viewing_distance" not in listed, listed

    def test_gate_does_not_keep_the_distance_as_missing(self):
        """Gate 的 missing 里也不该再出现"视距"（否则收口层还会用它当理由）。"""
        from src.models.requirement import RequirementProfile
        from src.rag.readiness import check_recommendation_ready

        slots = {"display_type": "LED", "environment": "indoor", "pixel_pitch_mm": 4.0}
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        profile.record_ask("viewing_distance")
        profile.mark_unknown("viewing_distance")

        decision = check_recommendation_ready(profile)

        assert "viewing_distance" not in (decision.missing or []), decision.missing


class TestTypeAnswerIsNotAProductQuestion:
    """问题 1：客户只是回答"要哪种屏"，不能当成产品提问走自由问答。"""

    def _classify(self, monkeypatch, message: str, llm_intent: str = "product_question"):
        import importlib

        classify_mod = importlib.import_module("src.agents.sales.nodes.classify")
        understanding_mod = importlib.import_module(
            "src.dialogue.product_type_understanding"
        )
        from src.dialogue import get_conversation_state
        from src.memory.store import memory

        session_id = f"type-answer-{abs(hash(message)) % 9999}"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        # 上一轮我们问的就是"要哪种屏"→ 语境判断才会被触发
        get_conversation_state(session_id).note_ai_turn(
            action="ask_only",
            question="Are you looking for an LED display, or an LCD?",
            slot="display_type",
        )

        class _FakeLLM:
            def __init__(self, *args, **kwargs):
                pass

            def invoke(self, *args, **kwargs):
                return SimpleNamespace(content=llm_intent)

        monkeypatch.setattr(classify_mod, "ChatOpenAI", _FakeLLM)
        monkeypatch.setattr(
            understanding_mod,
            "understand_product_type_reply",
            lambda *a, **k: {"reply": "chose", "display_type": "LED"},
        )

        state = {
            "current_message": message,
            "messages": [],
            "session_id": session_id,
            "requirements": {},
        }
        out = classify_mod.classify(state)
        memory.clear(session_id)
        return out

    def test_short_type_answer_is_treated_as_an_answer(self, monkeypatch):
        out = self._classify(monkeypatch, "led")

        assert out["intent"] == "need_query", (
            "客户在回答「要哪种屏」，这条不是产品提问，不能走自由问答（会冒出放宽话术）"
        )
        assert out.get("display_type_decision", {}).get("display_type") == "LED"

    def test_answer_with_a_real_question_keeps_the_question_intent(self, monkeypatch):
        """客户一边选型一边提问（"led, 多少钱？"）→ 仍然要回答他的问题。"""
        out = self._classify(monkeypatch, "led, how much is it?")

        assert out["intent"] == "product_question", out["intent"]


class TestNoRelaxationTalkDuringCollection:
    """放宽条件的话术只属于"匹配不到产品"，不该出现在正常采集轮。"""

    def test_relaxation_answer_is_not_used_by_the_sales_layer(self):
        for rel in (
            "src/agents/sales/nodes/script_generator.py",
            "src/agents/sales/nodes/requirement.py",
            "src/orchestrator.py",
        ):
            text = open(os.path.join(project_root, rel), encoding="utf-8").read()
            assert "relaxation_answer" not in text, rel
