"""v2.7 Phase 16 + §26：真并发下的"一个 Turn 只执行一次"。

计划点名的 6 个 Case：

    Case 1 连续消息        M1 M2 M3 → 1 个 Turn、1 次 Agent、1 条回复
    Case 2 快速连续消息    50~200ms 间隔 → 聚合成 1 个 Turn
    Case 3 同 Message 重试 同一 message_id 再发 → Agent 只跑 1 次
    Case 4 同 Turn 重试    同一 turn_id 再发 → Agent 只跑 1 次，返回同一回复
    Case 5 并发请求        两个请求同时到达 → Session Lock 串行化
    Case 6 n8n Retry       Webhook 重放若干次 → 只有一个业务结果
"""
import os
import sys
import threading
import time

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.input.message_deduplicator import MessageDeduplicator  # noqa: E402
from src.input.message_inbox import MessageInbox  # noqa: E402
from src.input.turn_builder import TurnBuilder  # noqa: E402
from src.input.turn_executor import TurnExecutor  # noqa: E402
from src.input.turn_store import RESPONDED, TurnStore  # noqa: E402


class _RecordingRunner:
    """记录每次执行（用于断言 Agent 跑了几次）。"""

    def __init__(self, delay: float = 0.0):
        self.calls = []
        self.delay = delay
        self._lock = threading.Lock()

    def __call__(self, payload):
        with self._lock:
            self.calls.append(dict(payload))
        if self.delay:
            time.sleep(self.delay)
        text = str(payload.get("text") or "")
        return {
            "response": f"reply to {text!r}",
            "action": "ask_only",
            "question_slot": "environment",
            "response_count": 1,
            "final_response": {"text": f"reply to {text!r}", "validation_result": "PASS"},
            "_turn": {"llm_calls": 1, "speech_act": "NEW_REQUIREMENT"},
        }


def _executor(runner, *, grace_ms=150, max_window_ms=1200):
    store = TurnStore()
    builder = TurnBuilder(
        store, grace_seconds=grace_ms / 1000.0, max_window_seconds=max_window_ms / 1000.0
    )
    return TurnExecutor(
        runner,
        store=store,
        builder=builder,
        inbox=MessageInbox(),
        deduplicator=MessageDeduplicator(),
    )


class TestCase1ContinuousMessages:

    def test_three_messages_make_one_turn_and_one_agent_run(self):
        runner = _RecordingRunner()
        executor = _executor(runner, grace_ms=400)
        outcomes = []

        def send(index, text):
            time.sleep(0.03 * index)
            outcomes.append(
                executor.submit(session_id="case1", text=text, message_ids=[f"m{index}"])
            )

        threads = [
            threading.Thread(target=send, args=(1, "3*5")),
            threading.Thread(target=send, args=(2, "indoor")),
            threading.Thread(target=send, args=(3, "P3")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)

        assert len(runner.calls) == 1, f"Agent 只能跑一次，实际 {len(runner.calls)}"
        assert {item.turn_id for item in outcomes} == {outcomes[0].turn_id}
        assert outcomes[0].aggregated is True
        assert outcomes[0].response_count == 1
        payload_text = runner.calls[0]["text"]
        assert payload_text.index("3*5") < payload_text.index("indoor") < payload_text.index("P3")

    def test_frontend_batched_messages_are_one_turn(self):
        """前端已聚合（一次请求带 3 条）→ 也是 1 个 Turn、1 次 Agent。"""
        runner = _RecordingRunner()
        executor = _executor(runner)
        outcome = executor.submit(
            session_id="case1b",
            text="3*5\nindoor\nP3",
            message_ids=["b1", "b2", "b3"],
        )
        assert len(runner.calls) == 1
        assert outcome.message_ids == ["b1", "b2", "b3"]
        assert outcome.aggregated is True


class TestCase2FastMessages:

    def test_50_to_200ms_spacing_is_aggregated(self):
        runner = _RecordingRunner()
        executor = _executor(runner, grace_ms=300)
        outcomes = []

        def send(index):
            time.sleep(0.05 * (index + 1))
            outcomes.append(
                executor.submit(session_id="case2", text=f"m{index}", message_ids=[f"f{index}"])
            )

        threads = [threading.Thread(target=send, args=(i,)) for i in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)

        assert len(runner.calls) == 1, f"应聚合成 1 个 Turn，实际 {len(runner.calls)} 次"
        assert len({item.turn_id for item in outcomes}) == 1


class TestCase3MessageRetry:

    def test_same_message_id_runs_agent_once(self):
        runner = _RecordingRunner()
        executor = _executor(runner)
        first = executor.submit(session_id="case3", text="hello", message_ids=["same-1"])
        second = executor.submit(session_id="case3", text="hello", message_ids=["same-1"])
        assert len(runner.calls) == 1
        assert second.duplicate is True
        assert second.response == first.response
        assert second.turn_id == first.turn_id


class TestCase4TurnRetry:

    def test_same_turn_id_returns_committed_response(self):
        runner = _RecordingRunner()
        executor = _executor(runner)
        first = executor.submit(
            session_id="case4", text="hello", message_ids=["t4-a"], turn_id="T100"
        )
        second = executor.submit(
            session_id="case4", text="hello", message_ids=["t4-b"], turn_id="T100"
        )
        assert len(runner.calls) == 1
        assert second.duplicate is True
        assert second.turn_id == "T100"
        assert second.response == first.response


class TestCase5ConcurrentRequests:

    def test_two_requests_are_serialized_by_session_lock(self):
        runner = _RecordingRunner(delay=0.05)
        executor = _executor(runner, grace_ms=120)
        results = []

        def send(index, message_id):
            results.append(
                executor.submit(session_id="case5", text=f"msg{index}", message_ids=[message_id])
            )

        threads = [
            threading.Thread(target=send, args=(1, "c5-1")),
            threading.Thread(target=send, args=(2, "c5-2")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)

        assert len({item.turn_id for item in results}) == 1, "同一时间只能有一个 Turn"
        assert len(runner.calls) == 1

    def test_message_after_seal_starts_next_turn(self):
        """已封口的 Turn 不再吞新消息 → 新消息进入下一轮（计划 §5.3）。"""
        runner = _RecordingRunner()
        executor = _executor(runner, grace_ms=80)
        first = executor.submit(session_id="case5b", text="one", message_ids=["c5b-1"])
        time.sleep(0.2)
        second = executor.submit(session_id="case5b", text="two", message_ids=["c5b-2"])
        assert first.turn_id != second.turn_id
        assert len(runner.calls) == 2
        assert "one" in runner.calls[0]["text"]
        assert "two" in runner.calls[1]["text"]


class TestCase6WebhookRetry:

    def test_webhook_replays_produce_one_business_result(self):
        runner = _RecordingRunner()
        executor = _executor(runner)
        payload_id = "n8n-msg-1"
        outcomes = [
            executor.submit(session_id="case6", text="retry", message_ids=[payload_id])
            for _ in range(3)
        ]
        assert len(runner.calls) == 1
        assert len({item.response for item in outcomes}) == 1
        assert sum(1 for item in outcomes if item.duplicate) == 2

    def test_failed_turn_can_be_retried(self):
        """执行失败 → 去重登记撤销，客户重试要能重新跑。"""
        calls = {"n": 0}

        def flaky(payload):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return {"response": "ok", "action": "ask_only", "response_count": 1}

        executor = _executor(flaky)
        first = executor.submit(session_id="case6b", text="x", message_ids=["fail-1"])
        second = executor.submit(session_id="case6b", text="x", message_ids=["fail-1"])
        assert first.status == "FAILED"
        assert second.status == "COMMITTED"
        assert calls["n"] == 2


class TestInboxAndTrace:

    def test_message_walks_the_status_machine(self):
        runner = _RecordingRunner()
        executor = _executor(runner)
        executor.submit(session_id="inbox-1", text="hi", message_ids=["inbox-m1"])
        message = executor.inbox.get("inbox-m1")
        assert message is not None
        assert message.status == "COMMITTED"
        assert message.turn_id

    def test_turn_trace_is_recorded(self):
        runner = _RecordingRunner()
        executor = _executor(runner)
        outcome = executor.submit(session_id="trace-1", text="hi", message_ids=["trace-m1"])
        assert outcome.trace["turn_id"] == outcome.turn_id
        assert outcome.trace["commit_status"] == "COMMITTED"
        assert outcome.trace["response_count"] == 1
        assert outcome.trace["llm_call_count"] == 1

    def test_turn_store_is_idempotent(self):
        store = TurnStore()
        record = store.create("s1", message_ids=["m1"], text="hi")
        store.commit(record.turn_id, response="first", action="ask_only", question_slot="size")
        store.commit(record.turn_id, response="second", action="ANSWER")
        assert store.get(record.turn_id).final_response == "first"
        assert store.previous_question("s1")["question_slot"] == "size"


class TestFollowUpWhileGenerating:
    """真实日志（2026-09-21）的"连续询问"场景：

    客户发完 "3 * 5" 后 2.3s 又发了 "indoor"，而此时第一条还在生成（8.4s）。
    旧行为：第二条开了新 Turn → 客户连着收到两条回复。
    现在：生成期间到达的消息并入同一个 Turn，客户只收到**一条**回复。
    """

    def test_message_arriving_during_generation_is_merged(self):
        runner = _RecordingRunner(delay=0.6)
        executor = _executor(runner, grace_ms=80)
        results = {}

        def send(key, text, delay):
            time.sleep(delay)
            results[key] = executor.submit(
                session_id="follow-up-1", text=text, message_ids=[f"fu-{key}"]
            )

        first = threading.Thread(target=send, args=("a", "3 * 5", 0.0))
        second = threading.Thread(target=send, args=("b", "indoor", 0.3))
        first.start()
        second.start()
        first.join(timeout=20)
        second.join(timeout=20)

        assert results["a"].turn_id == results["b"].turn_id, "两条消息属于同一个 Turn"
        assert results["a"].response == results["b"].response, "客户只看到一条回复"
        assert results["a"].response_count == 1
        # 契约：并入进来的那一条请求带 joined=True（API 会映射成 duplicate=true），
        # 前端据此**不再渲染第二个气泡**；只有领头那一条负责渲染。
        assert results["a"].joined is False
        assert results["b"].joined is True
        # 这一轮的最终输入必须包含两条消息（信息不丢）
        assert "3 * 5" in runner.calls[-1]["text"]
        assert "indoor" in runner.calls[-1]["text"]
        # 第一条被并入后重跑，因此内部跑了 2 次；但只提交了 1 条回复
        assert len({item.turn_id for item in results.values()}) == 1
        assert results["a"].trace["commit_status"] == "COMMITTED"

    def test_follow_up_does_not_create_a_second_committed_turn(self):
        runner = _RecordingRunner(delay=0.4)
        executor = _executor(runner, grace_ms=60)
        seen = []

        def send(text, delay, key):
            time.sleep(delay)
            outcome = executor.submit(
                session_id="follow-up-2", text=text, message_ids=[f"x-{key}"]
            )
            seen.append(outcome)

        threads = [
            threading.Thread(target=send, args=("first", 0.0, "1")),
            threading.Thread(target=send, args=("second", 0.2, "2")),
            threading.Thread(target=send, args=("third", 0.35, "3")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)

        committed = [item for item in seen if item.status == "COMMITTED"]
        assert committed, "至少有一次提交"
        assert len({item.turn_id for item in committed}) == 1
        assert len({item.response for item in committed}) == 1, "只有一个客户可见回复"

    def test_follow_up_missing_the_merge_window_is_never_dropped(self):
        """补发消息要是在"回复已经生成完、只差提交"的窗口到达 → 另起一轮回答它。

        实测风险：前端现在会把客户补发的消息**立刻**送出去（不再等上一轮回复），
        所以必须保证这一条不会被"并入失败"直接吞掉 —— 客户发了却没人回答。
        """
        from src.input.message import CustomerMessage

        runner = _RecordingRunner()
        executor = _executor(runner, grace_ms=60)
        first = executor.submit(session_id="late-1", text="first", message_ids=["late-1"])
        assert "first" in first.response

        # 手工构造那个窗口：Turn 已经是"回复已生成"（非终态）且仍标记为运行中，
        # 补发的消息进了 follow-ups，却没能被并进那一轮（真实链路的竞态）。
        executor.store.mark(first.turn_id, RESPONDED)
        executor._mark_running("late-1", first.turn_id)
        executor._add_follow_ups(
            first.turn_id,
            [
                CustomerMessage(
                    message_id="late-2", session_id="late-1", text="second"
                )
            ],
        )

        second = executor.submit(
            session_id="late-1", text="second", message_ids=["late-2"]
        )
        assert second.duplicate is False, "没并进去就不能拿旧回复糊弄客户"
        assert second.turn_id != first.turn_id, "应该另起一轮"
        assert "second" in second.response
        assert len(runner.calls) == 2
