"""v2.7 §42（Phase 23）：连续信息必须被完整吸收，而不是每条重问一次。"""
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
from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.query_understanding import extract_slots  # noqa: E402

MESSAGES = [
    "I need an LED screen.",
    "Indoor.",
    "3x5 meters.",
    "P3.",
    "8 meters.",
]


class _AbsorbingRunner:
    """把一轮里所有消息的槽位合并进档案（模拟真实需求提取的吸收能力）。"""

    def __init__(self):
        self.calls = []
        self.profile = RequirementProfile.from_slots(
            {"display_type": "LED"}, explicit_keys={"display_type"}
        )

    def __call__(self, payload):
        text = str(payload.get("text") or "")
        self.calls.append(text)
        merged = dict(self.profile.to_facts())
        for part in text.splitlines():
            for key, value in (extract_slots(part) or {}).items():
                if not str(key).startswith("_"):
                    merged[key] = value
        self.profile = RequirementProfile.from_slots(merged, explicit_keys=set(merged))
        return {
            "response": "Will the screen be installed indoors or outdoors?",
            "action": "ask_only",
            "question_slot": "environment",
            "response_count": 1,
            "_turn": {"llm_calls": 1},
        }


class TestContinuousInformation:

    def _executor(self, runner, session_id, *, grace_ms=250):
        store = TurnStore()
        manager = TurnManager(
            store, grace_seconds=grace_ms / 1000.0, max_window_seconds=2.0
        )
        return TurnExecutor(
            runner,
            store=store,
            builder=manager,
            inbox=MessageInbox(),
            deduplicator=MessageDeduplicator(),
        )

    def test_five_messages_are_absorbed_in_one_turn(self):
        session_id = "continuous-1"
        runner = _AbsorbingRunner()
        executor = self._executor(runner, session_id)
        outcomes = []

        def send(index, text):
            time.sleep(0.04 * index)
            outcomes.append(
                executor.submit(
                    session_id=session_id,
                    messages=[
                        CustomerMessage(
                            session_id=session_id,
                            text=text,
                            message_id=f"c-{index}",
                        )
                    ],
                )
            )

        threads = [
            threading.Thread(target=send, args=(index, text))
            for index, text in enumerate(MESSAGES)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)

        assert len(runner.calls) == 1, f"5 条连续消息 = 1 次 Agent，实际 {len(runner.calls)}"
        assert len({item.turn_id for item in outcomes}) == 1
        assert all(item.response_count == 1 for item in outcomes)

        facts = runner.profile.to_facts()
        assert facts.get("environment") == "indoor"
        assert facts.get("pixel_pitch_mm") == 3.0
        assert facts.get("viewing_distance_m") == 8.0
        assert facts.get("target_width_mm") or facts.get("target_height_mm")
