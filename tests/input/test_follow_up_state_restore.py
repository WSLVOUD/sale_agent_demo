"""客户插话 → 作废重跑时，必须把"作废那一版"的状态一起退回。

真实日志（2026-09-21）：客户发完 "i need a display"，AI 正在生成时又发 "3 * 5"。
作废的那一版把 environment 记成了"已经问过"，重跑时它就被跳过 → AI 改问 P 值，
客户看到的是**两个不同的问题连续抛出**。
"""
import copy
import os
import sys
import threading
import time

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.input.message import CustomerMessage  # noqa: E402
from src.input.message_deduplicator import MessageDeduplicator  # noqa: E402
from src.input.message_inbox import MessageInbox  # noqa: E402
from src.input.turn_executor import TurnExecutor  # noqa: E402
from src.input.turn_manager import TurnManager  # noqa: E402
from src.input.turn_store import TurnStore  # noqa: E402


class _FakeSessionState:
    """模拟真实的需求档案 + 问题登记（asked_slot / ask_count / questions）。"""

    def __init__(self):
        self.asked_slot = ""
        self.ask_count = 0
        self.questions = []
        self.texts = []

    def snapshot(self):
        return copy.deepcopy(self.__dict__)

    def restore(self, snap):
        self.__dict__.update(copy.deepcopy(snap))


STATE = _FakeSessionState()


class _PlanningRunner:
    """像真实链路一样：先记"这一项问过了"，再返回那句话。"""

    def __init__(self):
        self.calls = []
        self.delay = 0.3

    def __call__(self, payload):
        text = str(payload.get("text") or "")
        self.calls.append(text)
        STATE.texts.append(text)
        # 环境没问过 → 问环境（硬性 Gate）；问过 → 问下一项
        if STATE.asked_slot != "environment":
            question, slot = "Is the installation going to be indoors or outdoors?", "environment"
        else:
            question, slot = "What pixel pitch are you thinking?", "pixel_pitch"
        STATE.asked_slot = slot
        STATE.ask_count += 1
        STATE.questions.append(slot)
        time.sleep(self.delay)
        return {
            "response": question,
            "action": "ask_only",
            "question_slot": slot,
            "response_count": 1,
            "_turn": {"llm_calls": 1},
        }


def _executor(runner, *, grace_ms=60):
    store = TurnStore()
    manager = TurnManager(store, grace_seconds=grace_ms / 1000.0, max_window_seconds=1.2)
    return TurnExecutor(
        runner,
        store=store,
        builder=manager,
        inbox=MessageInbox(),
        deduplicator=MessageDeduplicator(),
        state_snapshot=lambda session_id: STATE.snapshot(),
        state_restore=lambda session_id, snap: STATE.restore(snap),
    )


class TestFollowUpStateRestore:

    def setup_method(self):
        global STATE
        STATE = _FakeSessionState()

    def test_superseded_run_does_not_pollute_the_question_state(self):
        runner = _PlanningRunner()
        executor = _executor(runner)
        results = {}

        def send(key, text, delay):
            time.sleep(delay)
            results[key] = executor.submit(
                session_id="restore-1", text=text, message_ids=[f"r-{key}"]
            )

        first = threading.Thread(target=send, args=("a", "i need a display", 0.0))
        second = threading.Thread(target=send, args=("b", "3 * 5", 0.15))
        first.start()
        second.start()
        first.join(timeout=20)
        second.join(timeout=20)

        assert results["a"].turn_id == results["b"].turn_id
        assert results["a"].response == results["b"].response, "客户只收到一条回复"
        # 关键：最终问的仍然是"还没被回答的那一项"，而不是因为记录被污染而换题
        assert results["a"].question_slot == "environment", results["a"].question_slot
        assert "indoors or outdoors" in results["a"].response
        # 作废的那一版不算"问过"：最终只记了一次
        assert STATE.ask_count == 1, STATE.ask_count
        assert STATE.questions == ["environment"], STATE.questions
        # 两条消息都进了这一轮的输入
        assert "3 * 5" in runner.calls[-1] and "i need a display" in runner.calls[-1]

    def test_two_separate_turns_can_move_on(self):
        """客户没答出来 → 问题可以跳转（硬性条件也一样）。

        "不连续提问"由 continuation_budget 控制：客户没答时先承接（最多 3 条），
        第 4 条才拉回需求 —— 见 tests/dialogue/test_ack_streak.py。
        """
        from src.dialogue import (
            get_conversation_state,
            next_question_plan,
            reset_conversation_state,
        )
        from src.models.requirement import RequirementProfile
        from src.rag.query_understanding import extract_slots

        def profile_from(*messages):
            merged = {}
            for message in messages:
                for key, value in (extract_slots(message) or {}).items():
                    if not str(key).startswith("_"):
                        merged[key] = value
            return RequirementProfile.from_slots(merged, explicit_keys=set(merged))

        session_id = "restore-2"
        reset_conversation_state(session_id)
        profile = profile_from("i need a display")
        first = next_question_plan(
            profile, session_id=session_id, conversation=get_conversation_state(session_id)
        )
        assert first.slot == "environment"
        profile.last_asked_slot = "environment"
        profile.record_ask("environment")

        # 客户答的是尺寸 → 环境仍然未知，但可以跳到别的候选（不必反复问环境）
        answered = profile_from("i need a display", "3 * 5")
        answered.last_asked_slot = "environment"
        answered.record_ask("environment")
        second = next_question_plan(
            answered, session_id=session_id, conversation=get_conversation_state(session_id)
        )
        assert getattr(second, "slot", "") != "environment", "允许跳转，不重复问"

    def test_soft_question_still_steps_aside(self):
        """软问题（使用场景 / 价位取向）不追问，避免问卷感。"""
        from src.dialogue import (
            get_conversation_state,
            next_question_plan,
            reset_conversation_state,
        )
        from src.models.requirement import RequirementProfile

        session_id = "restore-3"
        reset_conversation_state(session_id)
        slots = {"display_type": "LED", "environment": "indoor", "target_width_mm": 3000,
                 "target_height_mm": 5000, "pixel_pitch_mm": 3.0}
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        profile.last_asked_slot = "price_preference"
        profile.record_ask("price_preference")
        plan = next_question_plan(
            profile, session_id=session_id, conversation=get_conversation_state(session_id)
        )
        assert getattr(plan, "slot", "") != "price_preference"
