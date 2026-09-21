"""v2.7 Phase 20（§36）：把真实日志 Replay 一遍。

要求（计划 §36/§45）：

    · 每条客户消息 = 一个 Turn = 一次 Agent Run = 一条 Response；
    · 客户答非所问（问 indoor/outdoor、答 3×5）→ 信息必须进入 RequirementProfile；
    · 同一项问题在没有新信息时不得连续重复；
    · 每个 Turn 只有一个问题。

Replay 用真实的对话层函数（question flow / registry / firewall）跑，不依赖 LLM。
"""
import json
import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import (  # noqa: E402
    get_conversation_state,
    next_question_plan,
    previous_slot_blocked,
    reset_conversation_state,
)
from src.input.message import CustomerMessage  # noqa: E402
from src.input.message_deduplicator import MessageDeduplicator  # noqa: E402
from src.input.message_inbox import MessageInbox  # noqa: E402
from src.input.turn_executor import TurnExecutor  # noqa: E402
from src.input.turn_manager import TurnManager  # noqa: E402
from src.input.turn_store import TurnStore  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.query_understanding import extract_slots  # noqa: E402

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_session_log.json")


class _ScriptedSales:
    """Replay 用的"离线销售 Agent"：真实跑需求提取 + 问题选择，不调 LLM。"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.profile = RequirementProfile.from_slots(
            {"display_type": "LED"}, explicit_keys={"display_type"}
        )
        self.calls = []
        self.questions = []

    def __call__(self, payload):
        text = str(payload.get("text") or "")
        self.calls.append(text)

        slots = {
            key: value
            for key, value in (extract_slots(text) or {}).items()
            if not str(key).startswith("_")
        }
        if slots:
            merged = dict(self.profile.to_facts())
            merged.update(slots)
            self.profile = RequirementProfile.from_slots(
                merged, explicit_keys=set(merged)
            )

        plan = next_question_plan(
            self.profile,
            session_id=self.session_id,
            conversation=get_conversation_state(self.session_id),
        )
        slot = str(getattr(plan, "slot", "") or "")
        question = str(getattr(plan, "question", "") or "")
        if slot:
            self.questions.append(slot)

        return {
            "response": question or "Understood.",
            "action": "ask_only" if slot else "answer_only",
            "question_slot": slot,
            "response_count": 1,
            "requirements": dict(self.profile.to_facts()),
            "response_mode": "NATURAL",
            "final_response": {"text": question, "validation_result": "PASS"},
            "_turn": {"llm_calls": 1, "speech_act": "NEW_REQUIREMENT"},
        }


def _executor(runner, *, grace_ms=80):
    store = TurnStore()
    manager = TurnManager(
        store, grace_seconds=grace_ms / 1000.0, max_window_seconds=1.2
    )
    return TurnExecutor(
        runner,
        store=store,
        builder=manager,
        inbox=MessageInbox(),
        deduplicator=MessageDeduplicator(),
    )


def _load_turns():
    with open(LOG_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


class TestRealLogReplay:

    def test_one_turn_one_agent_run_one_question(self):
        replay = _load_turns()
        session_id = replay["session_id"]
        reset_conversation_state(session_id)
        sales = _ScriptedSales(session_id)
        executor = _executor(sales)

        outcomes = []
        for index, turn in enumerate(replay["turns"]):
            outcomes.append(
                executor.submit(
                    session_id=session_id,
                    messages=[
                        CustomerMessage(
                            session_id=session_id,
                            text=turn["customer"],
                            message_id=f"log-{index}",
                        )
                    ],
                )
            )
            time.sleep(0.15)  # 模拟客户隔了一会儿才发下一条

        assert len(sales.calls) == len(replay["turns"]), "每条客户消息 = 一次 Agent"
        assert all(item.response_count == 1 for item in outcomes)
        assert all(item.trace["response_count"] == 1 for item in outcomes)
        assert all(item.trace["commit_status"] == "COMMITTED" for item in outcomes)

    def test_no_question_is_repeated_without_new_information(self):
        replay = _load_turns()
        session_id = replay["session_id"] + "-repeat"
        reset_conversation_state(session_id)
        profile = RequirementProfile.from_slots(
            {"display_type": "LED"}, explicit_keys={"display_type"}
        )
        asked = []
        for turn in replay["turns"]:
            plan = next_question_plan(
                profile,
                session_id=session_id,
                conversation=get_conversation_state(session_id),
            )
            slot = str(getattr(plan, "slot", "") or "")
            if not slot:
                continue
            if asked and asked[-1] == slot:
                assert previous_slot_blocked(profile, slot) is False, (
                    f"{slot} 在没有任何新信息的情况下被连续问了两次"
                )
            asked.append(slot)
            profile.last_asked_slot = slot
            profile.record_ask(slot)

    def test_wrong_slot_answer_keeps_the_information(self):
        """§41：问 indoor/outdoor、客户答 3×5 → 尺寸必须进档案。"""
        session_id = "replay-wrong-slot"
        reset_conversation_state(session_id)
        sales = _ScriptedSales(session_id)
        executor = _executor(sales)

        executor.submit(
            session_id=session_id,
            messages=[CustomerMessage(session_id=session_id, text="i need a led display", message_id="w1")],
        )
        time.sleep(0.15)
        executor.submit(
            session_id=session_id,
            messages=[CustomerMessage(session_id=session_id, text="3*5", message_id="w2")],
        )

        facts = sales.profile.to_facts()
        assert facts.get("target_width_mm") or facts.get("target_height_mm"), facts
