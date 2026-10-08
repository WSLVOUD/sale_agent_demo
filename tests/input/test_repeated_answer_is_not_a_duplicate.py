"""同一句话回答**不同**的问题，不是"重发"（客户口径 2026-09-30）。

实测（客户原文）：

    AI: Is this a tender project?
    客户: no
    AI: Do you need an OPS slot (a built-in PC module)?
    客户: no / no / no / no / no / no      ← 全都没有反应，终端也没收到消息

根因：消息去重的**指纹**是"同 session + 同文字"（TTL 10 分钟）。客户两次都用 "no"
回答不同的 yes/no 问题 → 第二次被当成"重发"直接丢弃（`duplicate_payload_fingerprint`），
连同一条 turn 的结果一起返回，客户看不到任何新内容。

判据改成"上下文优先"：**回答的是另一个问题 → 新消息**（多快都算）；
只有"上一轮还在处理（并入同一轮）"或"同一个问题、刚提交还在重发窗口内"
才算重发。

> 2026-10 补：上一版把"5 秒内刚提交"放在语境判定**之前**，于是客户**很快**
> 回答下一问时（实测 3 秒内又答一个 "no"）仍被当成重发丢弃 —— 客户体感就是
> "发一个 no 没反应，得再发一个才动"。现在语境判定排在最前面。
"""
import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.input.message_deduplicator import MessageDeduplicator  # noqa: E402
from src.input.message_inbox import MessageInbox  # noqa: E402
from src.input.message_store import InMemoryMessageStore  # noqa: E402
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
        # 去重状态用**独立**的 store：默认 get_message_store() 是进程级单例，
        # 会让同一个测试里的多个场景互相污染（前一个场景登记过的 "no" 会让
        # 后一个场景一上来就被判成重复）。
        deduplicator=MessageDeduplicator(store=InMemoryMessageStore()),
        duplicate_grace_seconds=grace_seconds,
    )


def _submit(executor, text, message_id):
    return executor.submit(
        session_id="repeat-no", text=text, message_ids=[message_id], source="api"
    )


class TestFingerprintCarriesTheQuestionContext:
    """指纹本身要带语境 —— 不能只靠"判成重复之后再复核"。"""

    def test_same_text_answering_different_questions_has_different_fingerprints(self):
        from src.input.message_deduplicator import payload_fingerprint

        tender = payload_fingerprint("s", "no", None, "lcd_tender")
        ops = payload_fingerprint("s", "no", None, "lcd_ops")
        assert tender != ops, "同一个 no 回答不同问题必须是两条不同的消息"

    def test_same_text_and_same_context_is_still_a_retry(self):
        """客户端双击 / 网络重试：文字与语境都一样 → 仍然要能去重。"""
        from src.input.message_deduplicator import payload_fingerprint

        first = payload_fingerprint("s", "no", None, "lcd_ops")
        retry = payload_fingerprint("s", "no", None, "lcd_ops")
        assert first == retry

    def test_context_is_optional_for_old_callers(self):
        """老调用（不传语境）行为不变。"""
        from src.input.message_deduplicator import payload_fingerprint

        assert payload_fingerprint("s", "no") == payload_fingerprint("s", "no", None, "")

    def test_deduplicator_uses_the_context(self):
        from src.input.message_deduplicator import MessageDeduplicator
        from src.input.message_store import InMemoryMessageStore

        dedup = MessageDeduplicator(store=InMemoryMessageStore())
        assert dedup.check("s", text="no", context="lcd_tender").is_duplicate is False
        dedup.remember("s", text="no", context="lcd_tender")
        # 换个问题再答一次 "no" → 不是重复
        assert dedup.check("s", text="no", context="lcd_ops").is_duplicate is False
        # 同一个问题重发 → 是重复
        assert dedup.check("s", text="no", context="lcd_tender").is_duplicate is True


class TestRepeatedAnswerAcrossQuestions:
    def test_same_no_answering_the_next_question_is_processed_again_merged(self):
        """合并自 3 条同类测试（瘦身；断言全部保留）。"""

        # ── test_same_no_answering_the_next_question_is_processed_again ──
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

        # ── test_same_no_answering_the_next_question_quickly_is_processed_too ──
        """客户**很快**回答下一问（还在重发窗口内）也必须处理。

        实测（客户 2026-10）：AI 问招标 → 客户 "no" → AI 同一轮里接着问 OPS →
        客户几秒内又答 "no" → 被判成"重发"丢弃 → 前端收到 duplicate=true 不渲染 →
        客户体感"发一个 no 没反应，得再发一个才动"。
        时间窗口不能抢在"回答的是不是同一个问题"前面。
        """
        runner = _Runner(question_slots=("lcd_tender", "lcd_ops", "lcd_camera"))
        # 窗口故意开得很大：模拟"客户答得比窗口还快"
        executor = _executor(runner, grace_seconds=30.0)

        _submit(executor, "meeting room", "m-0")    # → 问 lcd_tender
        quick_1 = _submit(executor, "no", "m-1")    # 答 lcd_tender → 下一问问 lcd_ops
        # 等这一轮**提交完**（超过聚合 grace），但远小于 30s 的重发窗口 ——
        # 这正是真实时序：AI 一问出来，客户几秒内就答了。
        time.sleep(0.15)
        quick_2 = _submit(executor, "no", "m-2")    # 立刻答 lcd_ops（仍在窗口内）

        assert len(runner.calls) == 3, "同一句 no 回答另一个问题，多快都必须再跑一次"
        assert not quick_2.duplicate, quick_2
        assert quick_1.turn_id != quick_2.turn_id
        assert "reply #3" in str(quick_2.response), quick_2.response

        # ── test_a_quick_retry_is_still_deduped ──
        """窗口内的同内容重发（客户端重试 / 双击）照旧只跑一次。"""
        runner = _Runner(question_slots=("lcd_ops",))
        executor = _executor(runner, grace_seconds=30.0)

        _submit(executor, "meeting", "m-0")
        first = _submit(executor, "no", "m-1")
        second = _submit(executor, "no", "m-2")

        assert len(runner.calls) == 2, "刚回完就重发 → 复用上一轮（不重复跑 Agent）"
        assert second.duplicate, second

        # ── test_a_retry_while_the_turn_is_still_running_joins_it ──
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
