"""v2.5++++：《消息聚合与自然对话链路优化计划》§16 的测试。

覆盖：连续消息聚合 / 单条消息不额外等待 / 回答需求 / 客户主动询价 / 主动问交期 /
多意图 / P3→P2.9 解释 / Unsupported Fact / LLM 调用统计。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import (  # noqa: E402
    ANSWER_ONLY,
    ANSWER_REQUIREMENT,
    ANSWER_THEN_ASK,
    ASK_ONLY,
    DELIVERY_QUESTION,
    MULTI_INTENT,
    PRICE_QUESTION,
    PRODUCT_QUESTION,
    decide_action,
    detect_speech_act,
    should_append_requirement_question,
)
from src.dialogue.response_validator import validate_response  # noqa: E402
from src.input import MessageAggregator  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _profile(**slots) -> RequirementProfile:
    return RequirementProfile.from_slots(slots, explicit_keys=set(slots))


# ── §16.1 / §16.2：消息聚合 ───────────────────────────────────────────────
class TestMessageAggregation:

    def _aggregator(self, debounce=0.6, window=1.8):
        clock = _Clock()
        return MessageAggregator(
            debounce_seconds=debounce, max_window_seconds=window, now=clock
        ), clock

    def test_consecutive_messages_become_one_turn(self):
        aggregator, clock = self._aggregator()
        session = "agg-1"
        assert aggregator.add(session, {"message_id": "m1", "text": "3*5"}) is None
        clock.advance(0.3)
        assert aggregator.add(session, {"message_id": "m2", "text": "indoor"}) is None
        clock.advance(0.3)
        assert aggregator.add(session, {"message_id": "m3", "text": "P3"}) is None
        clock.advance(0.6)   # 静默 600ms → 这一轮结束
        turn = aggregator.flush(session)
        assert turn is not None
        assert len(turn.messages) == 3
        assert turn.text == "3*5 indoor P3"
        assert aggregator.flush(session) is None    # 只发一次

    def test_max_window_forces_flush(self):
        """客户一直不停（间隔都小于 debounce）→ 到 1800ms 上限必须发出。"""
        aggregator, clock = self._aggregator()
        session = "agg-2"
        assert aggregator.add(session, {"message_id": "a", "text": "hello"}) is None
        clock.advance(0.5)
        assert aggregator.add(session, {"message_id": "b", "text": "there"}) is None
        clock.advance(0.5)
        assert aggregator.add(session, {"message_id": "c", "text": "again"}) is None
        clock.advance(0.9)    # 距第一条已 1.9s > 1800ms 上限
        turn = aggregator.add(session, {"message_id": "d", "text": "P3"})
        assert turn is not None
        assert turn.text == "hello there again"

    def test_single_message_only_waits_the_debounce(self):
        """§16.2：单条消息最多多等 debounce（600ms），不会等到 max window。"""
        aggregator, clock = self._aggregator()
        session = "agg-3"
        aggregator.add(session, {"message_id": "s1", "text": "I need an indoor LED screen."})
        clock.advance(0.61)
        turn = aggregator.flush(session)
        assert turn is not None and turn.text.startswith("I need an indoor LED screen")

    def test_defaults_match_the_plan(self):
        from src.input.message_aggregator import (
            DEFAULT_DEBOUNCE_SECONDS,
            DEFAULT_MAX_WINDOW_SECONDS,
        )

        assert DEFAULT_DEBOUNCE_SECONDS == pytest.approx(0.6)
        assert DEFAULT_MAX_WINDOW_SECONDS == pytest.approx(1.8)


# ── §16.3 / §16.4 / §16.5：SpeechAct + Dialogue Policy ────────────────────
class TestSpeechActAndPolicy:

    def test_answer_requirement_is_recognized(self):
        profile = _profile(display_type="LED", environment="indoor")
        profile.last_asked_slot = "pixel_pitch"
        speech = detect_speech_act("P3", profile=profile, last_asked_slot="pixel_pitch")
        assert speech.speech_act == ANSWER_REQUIREMENT
        assert speech.field == "pixel_pitch"
        assert not speech.is_customer_question

        action = decide_action(speech, profile=profile, session_id="sa-1")
        assert action.action == ASK_ONLY
        assert action.target_slot and action.target_slot != "environment"

    def test_price_question_is_answered_only(self):
        profile = _profile(display_type="LED")
        speech = detect_speech_act("How much is it?", profile=profile)
        assert speech.speech_act == PRICE_QUESTION
        action = decide_action(speech, profile=profile, session_id="sa-2")
        assert action.action == ANSWER_ONLY
        assert should_append_requirement_question("How much is it?", profile) is False

    def test_delivery_question_is_answered_only(self):
        profile = _profile(display_type="LED")
        for message in ("How long is delivery?", "你的交付日期是多久？"):
            speech = detect_speech_act(message, profile=profile)
            assert speech.speech_act == DELIVERY_QUESTION, message
            assert decide_action(speech, profile=profile).action == ANSWER_ONLY, message
            assert should_append_requirement_question(message, profile) is False, message

    def test_multi_intent(self):
        profile = _profile(display_type="LED")
        profile.last_asked_slot = "pixel_pitch"
        speech = detect_speech_act(
            "P3. Also, how long is delivery?", profile=profile,
            last_asked_slot="pixel_pitch",
        )
        assert speech.speech_act == MULTI_INTENT
        assert speech.requirements.get("pixel_pitch_mm") == 3.0
        assert speech.question_kind() == DELIVERY_QUESTION
        # 先回答客户（delivery），这一轮不追加需求问题
        assert should_append_requirement_question(
            "P3. Also, how long is delivery?", profile
        ) is False

    def test_product_question_can_still_ask_a_needed_field(self):
        """问产品规格、而回答它确实缺硬性条件（尺寸）→ 允许有理由地追问。"""
        profile = _profile(display_type="LED", environment="indoor", installation="fixed")
        profile.last_asked_slot = "environment"
        speech = detect_speech_act("Can it do 4K?", profile=profile)
        assert speech.speech_act == PRODUCT_QUESTION
        assert should_append_requirement_question("Can it do 4K?", profile) is True


# ── §16.7：P3 → P2.9 ─────────────────────────────────────────────────────
class TestPitchExplanationEndToEnd:

    def test_recommendation_carries_pitch_resolution_and_explains(self):
        from src.rag.recommendation_engine import RecommendationEngine

        engine = RecommendationEngine()
        profile = RequirementProfile.from_slots(
            {
                "environment": "indoor",
                "installation": "rental",
                "viewing_distance_m": 5,
                "pixel_pitch_mm": 3.0,
            },
            explicit_keys={"environment", "installation", "viewing_distance_m", "pixel_pitch_mm"},
        )
        result = engine.recommend(profile=profile, top_k=3)
        assert result["recommendations"], "P3 租赁型号应该能选出来"
        top = result["recommendations"][0]
        resolution = top["pitch_resolution"]
        assert resolution["requested_pitch_text"] == "P3"
        assert resolution["resolved_pitch"] == pytest.approx(2.976, abs=0.01)
        assert resolution["needs_explanation"] is True
        # 解释句已经进入推荐理由（LLM 不会只丢一个 P2.9 出来）
        assert any("P2.9" in reason and "P3" in reason for reason in top["reasons"])


# ── §16.8：Unsupported Fact ───────────────────────────────────────────────
class TestUnsupportedFact:

    def test_invented_viewing_distance_is_blocked(self):
        from src.dialogue import build_grounded_facts

        facts = build_grounded_facts(profile=_profile(display_type="LED", environment="indoor"))
        bad = validate_response(
            "This is suitable for your 5m viewing distance.",
            grounded_facts=facts,
        )
        assert "ungrounded_numeric" in bad.issues

    def test_inferred_distance_is_allowed(self):
        from src.dialogue import build_grounded_facts

        profile = _profile(display_type="LED")
        profile.pixel_pitch_mm = 2.976
        profile.sources["pixel_pitch_mm"] = "inferred"
        facts = build_grounded_facts(profile=profile)
        ok = validate_response(
            "The pixel pitch of 2.976 works well here.",
            grounded_facts=facts,
        )
        assert "ungrounded_numeric" not in ok.issues


# ── §15：LLM 调用统计 ────────────────────────────────────────────────────
class TestLLMCallTracking:

    def test_tracker_counts_calls_in_a_turn(self):
        from src.observability.llm_tracker import LLMCallTracker

        tracker = LLMCallTracker()
        tracker.begin_turn("s-track", message_count=3, aggregated=True)
        with tracker.track(model="deepseek-chat") as record:
            record.total_tokens = 42
            record.prompt_tokens = 30
            record.completion_tokens = 12
        stats = tracker.end_turn()
        assert stats is not None
        assert stats.calls == 1
        assert stats.total_tokens == 42
        assert stats.session_id == "s-track"
        assert stats.records[0].model == "deepseek-chat"

    def test_llm_entry_point_is_tracked(self):
        """真正的调用入口（get_llm 返回的实例）也要被统计到。"""
        from src.core.llm import get_llm
        from src.observability.llm_tracker import get_llm_tracker

        tracker = get_llm_tracker()
        tracker.reset()
        tracker.begin_turn("s-llm", message_count=1, aggregated=False)
        try:
            llm = get_llm(use_retry=False)
        except Exception as exc:  # pragma: no cover - 没配 key 的环境
            pytest.skip(f"no LLM credentials: {exc}")
        try:
            llm.invoke("ping")
        except Exception:
            pass          # 离线环境必然失败 —— 但**失败也要记账**
        stats = tracker.current_turn().stats()
        tracker.end_turn()
        assert stats.calls == 1
        assert stats.records[0].model

    def test_finish_turn_attaches_stats_and_logs(self):
        from src.observability.llm_tracker import get_llm_tracker
        from src.orchestrator import DualAgentOrchestrator

        orchestrator = DualAgentOrchestrator.__new__(DualAgentOrchestrator)
        tracker = get_llm_tracker()
        tracker.reset()
        tracker.begin_turn("s-finish", message_count=2, aggregated=True)
        with tracker.track(model="deepseek-chat"):
            pass
        result = {}
        orchestrator._finish_llm_turn(result, "s-finish")
        assert result["_llm_stats"]["llm_calls"] == 1
        assert result["_turn"]["message_count"] == 2
        assert result["_turn"]["aggregated"] is True

    def test_perf_summary_reports_tracked_calls(self):
        from src.observability.llm_tracker import get_llm_tracker
        from src.observability.perf import PerfTracker

        tracker = get_llm_tracker()
        tracker.reset()
        tracker.begin_turn("s-perf", message_count=1)
        with tracker.track(model="deepseek-chat"):
            pass
        perf = PerfTracker("s-perf", "hi")
        stats = tracker.end_turn()
        assert stats.calls == 1
        summary = perf.summary()
        assert summary["llm_calls"] == 1        # 不再是 0
        assert "llm_latency_ms" in summary and "llm_tokens" in summary
