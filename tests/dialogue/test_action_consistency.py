"""Phase 1（《ResponseGenerator 架构收口改造计划》）：一个 Turn = 一个最终 DialogueAction。

计划 §16.1/§16.2 要求：

    QuestionPlanner = A，DialoguePolicy = B → 最终只能执行 B
    DialogueAction.question_slot == FinalResponse 实际询问的 slot
    ASK / ANSWER_AND_ASK → 最多一个问题；DIRECT_ANSWER / RECOMMEND → 0 个需求问题

实测 bug（2026-09-23，本文件锁住）：Policy 定的是 ASK(viewing_distance)，
销售层准备的问句却是 pixel_pitch → 收口后**最终问的还是 pixel_pitch**，
日志里 `selected_action.question_slot` 与实际问句不一致（两个决策中心）。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.memory.store import memory  # noqa: E402
from src.orchestrator import DualAgentOrchestrator  # noqa: E402


class _StubSales:
    def run(self, session_id, message, has_vision=False, **_kwargs):
        return {
            "intent": "need_query",
            "next_action": "ask",
            "response": "ok",
            "requirements": {},
            "products": [],
            "pending_question": "",
            "pending_slot": "",
        }


class _StubSolution:
    def run(self, *args, **kwargs):
        return {"answer": "Sure.", "products": []}


def _orchestrator() -> DualAgentOrchestrator:
    return DualAgentOrchestrator(sales_agent=_StubSales(), solution_agent=_StubSolution())


def _finalize(result, session_id: str, message: str = "indoor church 3x5"):
    memory.clear(session_id)
    memory.mark_first_contact_done(session_id)
    try:
        orch = _orchestrator()
        return orch._finalize_turn_response(result, session_id, message)
    finally:
        memory.clear(session_id)


class TestSingleActionPerTurn:

    def test_selected_action_is_single_and_others_are_discarded(self):
        result = {
            "response": "How far will people be sitting from the screen?",
            "pending_question": "How far will people be sitting from the screen?",
            "pending_slot": "viewing_distance",
            "next_action": "ask",
            "dialogue_action": {
                "action": "ask_only",
                "target_slot": "viewing_distance",
                "question": True,
                "priority": 3,
            },
            "products": [],
            "requirements": {},
        }
        out = _finalize(result, "consistency-1")
        assert out.get("selected_action"), "必须有唯一的最终 Action"
        assert isinstance(out["selected_action"], dict)
        assert out["selected_action"]["action"] in ("ask_only", "ASK", "ANSWER_AND_ASK")
        # 候选可以有多个，但最终只执行一个（其余进 discarded_actions 留痕）
        assert all(
            item != out["selected_action"] for item in (out.get("discarded_actions") or [])
        )
        assert out.get("action_candidates")


class TestPolicySlotWins:

    def test_final_question_follows_the_policy_slot(self):
        """Policy 说 viewing_distance，销售层准备了 pixel_pitch → 以 Policy 为准。"""
        result = {
            "response": "Your 3x5 screen is a good size. What pixel pitch do you need?",
            "pending_question": "What pixel pitch do you need?",
            "pending_slot": "pixel_pitch",
            "next_action": "ask",
            "dialogue_action": {
                "action": "ask_only",
                "target_slot": "viewing_distance",
                "question": True,
                "priority": 1,
            },
            "products": [],
            "requirements": {},
        }
        out = _finalize(result, "consistency-2")
        assert out["question_slot"] == "viewing_distance"
        assert out["selected_action"]["question_slot"] == "viewing_distance"
        # 承诺 §16.2：日志里的槽位 = 客户实际看到的问句
        assert out["question_count"] == 1
        assert "how far" in out["response"].lower(), out["response"]
        assert "pixel pitch" not in out["response"].lower()
        assert out["action_consistency"]["overridden_slot"] == "pixel_pitch"

    def test_agreement_keeps_the_prepared_question(self):
        result = {
            "response": "What pixel pitch do you need?",
            "pending_question": "What pixel pitch do you need?",
            "pending_slot": "pixel_pitch",
            "next_action": "ask",
            "dialogue_action": {
                "action": "ask_only",
                "target_slot": "pixel_pitch",
                "question": True,
                "priority": 1,
            },
            "products": [],
            "requirements": {},
        }
        out = _finalize(result, "consistency-3")
        assert out["question_slot"] == "pixel_pitch"
        assert out.get("action_consistency") in (None, {})
        assert "pixel pitch" in out["response"].lower()


class TestNoQuestionWhenNotAsking:

    def test_recommend_action_asks_nothing(self):
        result = {
            "response": "For your 3x5 indoor church screen, TW11-3216-P3.0 fits well.",
            "pending_question": "How far will people be sitting from the screen?",
            "pending_slot": "viewing_distance",
            "next_action": "ask",
            "dialogue_action": {"action": "RECOMMEND", "question": False},
            "products": [{"model": "TW11-3216-P3.0"}],
            "requirements": {},
        }
        out = _finalize(result, "consistency-4")
        assert out["question_count"] == 0, out["response"]
        assert out["question_slot"] == ""


class TestDuplicateFirewallStillWins:

    """重复提问闸门是同一决策层的防重放行 —— Policy 的对齐不能反过来覆盖它。"""

    def test_rerouted_question_is_not_realigned_back(self, monkeypatch):
        session_id = "consistency-5"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        try:
            orch = _orchestrator()
            # 模拟闸门"刚改过问"：verify 时直接返回 rerouted（真实链路见 duplicate_firewall 测试）
            monkeypatch.setattr(
                orch,
                "_apply_duplicate_firewall",
                lambda *a, **k: ("installation", "duplicate_question_rerouted"),
            )
            result = {
                "response": "Should it be a fixed install or a rental setup?",
                "pending_question": "Should it be a fixed install or a rental setup?",
                "pending_slot": "installation",
                "next_action": "ask",
                "turn_id": "turn-now",
                "dialogue_action": {
                    "action": "ask_only",
                    "target_slot": "viewing_distance",
                    "question": True,
                    "priority": 1,
                },
                "products": [],
                "requirements": {},
            }
            out = orch._finalize_turn_response(result, session_id, "indoor")
            # 闸门已经改问 installation → 不能被 Policy 对齐回 viewing_distance
            assert out["question_slot"] == "installation", out
            assert out.get("action_consistency") in (None, {})
        finally:
            memory.clear(session_id)
