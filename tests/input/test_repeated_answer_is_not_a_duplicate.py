"""同一句话回答**不同**的问题，不是"重发"（客户口径 2026-09-30）。

实测（客户原文）：

    AI: Is this a tender project?
    客户: no
    AI: Do you need an OPS slot (a built-in PC module)?
    客户: no / no / no / no / no / no      ← 全都没有反应，终端也没收到消息

根因：消息去重的**指纹**是"同 session + 同文字"（TTL 10 分钟）。客户两次都用 "no"
回答不同的 yes/no 问题 → 第二次被当成"重发"直接丢弃（`duplicate_payload_fingerprint`），
连同一条 turn 的结果一起返回，客户看不到任何新内容。

判据改成"上下文"：只有**上一轮还在处理**（并入同一轮）或**刚提交、还在重发窗口内**
才算重发；过了窗口，同一句话就是新消息。
"""
import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.input.message_deduplicator import MessageDeduplicator  # noqa: E402
from src.input.message_inbox import MessageInbox  # noqa: E402
from src.input.turn_builder import TurnBuilder  # noqa: E402
from src.input.turn_executor import TurnExecutor  # noqa: E402
from src.input.turn_store import TurnStore  # noqa: E402


class _Runner:
    """每次执行都记一笔；按顺序返回"这一轮问了哪一项"。"""

    def __init__(self, question_slots=("lcd_ops",)):
        self.calls = []
        self.question_slots = list(question_slots)

    def __call__(self, payload):
        self.calls.append(dict(payload))
        index = min(len(self.calls) - 1, len(self.question_slots) - 1)
        slot = self.question_slots[index]
        text = str(payload.get("text") or "")
        reply = f"reply #{len(self.calls)} asking {slot} (to {text!r})"
        return {
            "response": reply,
            "action": "ask_only",
            "question_slot": slot,
            "response_count": 1,
            "final_response": {"text": reply, "validation_result": "PASS"},
            "_turn": {"llm_calls": 1},
        }


def _executor(runner, *, grace_seconds: float):
    store = TurnStore()
    builder = TurnBuilder(store, grace_seconds=0.08, max_window_seconds=1.2)
    return TurnExecutor(
        runner,
        store=store,
        builder=builder,
        inbox=MessageInbox(),
        deduplicator=MessageDeduplicator(),
        duplicate_grace_seconds=grace_seconds,
    )


def _submit(executor, text, message_id):
    return executor.submit(
        session_id="repeat-no", text=text, message_ids=[message_id], source="api"
    )


class TestRepeatedAnswerAcrossQuestions:
    def test_same_no_answering_the_next_question_is_processed_again(self):
        """客户用同一个 "no" 回答**另一个** yes/no 问题 → 必须再跑一次。"""
        runner = _Runner(question_slots=("lcd_size", "lcd_ops", "lcd_camera"))
        executor = _executor(runner, grace_seconds=0.1)

        _submit(executor, "meeting", "m-0")      # → 问 lcd_size
        first_no = _submit(executor, "no", "m-1")  # 答 lcd_size → 下一问问 lcd_ops
        time.sleep(0.15)                          # 超过重发窗口，客户已看到下一问
        second_no = _submit(executor, "no", "m-2")  # 答 lcd_ops

        assert len(runner.calls) == 3, "第二句 no 必须真的再跑一次 Agent"
        assert not second_no.duplicate, second_no
        assert first_no.turn_id != second_no.turn_id
        assert "reply #3" in str(second_no.response), second_no.response

    def test_a_quick_retry_is_still_deduped(self):
        """窗口内的同内容重发（客户端重试 / 双击）照旧只跑一次。"""
        runner = _Runner(question_slots=("lcd_ops",))
        executor = _executor(runner, grace_seconds=30.0)

        _submit(executor, "meeting", "m-0")
        first = _submit(executor, "no", "m-1")
        second = _submit(executor, "no", "m-2")

        assert len(runner.calls) == 2, "刚回完就重发 → 复用上一轮（不重复跑 Agent）"
        assert second.duplicate, second

    def test_a_retry_while_the_turn_is_still_running_joins_it(self):
        runner = _Runner(question_slots=("lcd_ops",))
        executor = _executor(runner, grace_seconds=0.0)

        outcomes = []

        def send():
            outcomes.append(_submit(executor, "no", "m-1"))

        import threading

        thread = threading.Thread(target=send)
        thread.start()
        time.sleep(0.03)  # 还在聚合/生成中
        joined = _submit(executor, "no", "m-2")
        thread.join(timeout=10)

        assert len(runner.calls) == 1, "同一条消息在生成期间重发 → 并入同一轮"
        assert joined.duplicate or outcomes, (joined, outcomes)
