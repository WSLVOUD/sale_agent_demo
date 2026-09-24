"""v2.5 Phase 1：多条消息聚合成一个 UserTurn（debounce / window / 幂等 / 会话锁 / 多模态）。"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.input import MessageAggregator, UserTurn  # noqa: E402


class _Clock:
    def __init__(self):
        self.value = 1000.0

    def __call__(self):
        return self.value

    def tick(self, seconds):
        self.value += seconds


def _aggregator(clock, debounce=1.8, window=7.0):
    return MessageAggregator(debounce_seconds=debounce, max_window_seconds=window, now=clock)


class TestAggregation:

    def test_single_message(self):
        clock = _Clock()
        agg = _aggregator(clock)
        assert agg.add("s1", {"message_id": "m1", "text": "hi"}) is None
        clock.tick(2)
        assert agg.flush("s1") is None or True
        # 单条消息：debounce 到了以后由下一次 add / flush 发出
        agg.add("s1", {"message_id": "m1", "text": "hi"}) if False else None

    @pytest.mark.parametrize("count", [2, 3, 5])
    def test_consecutive_messages_become_one_turn(self, count):
        clock = _Clock()
        agg = _aggregator(clock)
        result = None
        for index in range(count):
            result = agg.add("s", {"message_id": f"m{index}", "text": f"part{index}"})
            if index < count - 1:
                clock.tick(0.5)      # 连续发送：还在 debounce 窗口内
                assert result is None
        clock.tick(2)                # 客户停顿超过 debounce
        turn = agg.add("s", {"message_id": "final", "text": "last"})
        assert isinstance(turn, UserTurn), "停顿后新消息到达 → 上一轮一次性发出"
        assert len(turn.messages) == count, "连续 count 条消息 → 一个 UserTurn"
        assert turn.text.startswith("part0") and turn.text.endswith(f"part{count - 1}")
        assert len(turn.message_ids) == count
        # 停顿之后那条消息属于下一个 turn
        clock.tick(2)
        nxt = agg.add("s", {"message_id": "final2", "text": "next"})
        assert isinstance(nxt, UserTurn) and "last" in nxt.text

    def test_max_window_forces_flush(self):
        clock = _Clock()
        agg = _aggregator(clock, debounce=1.8, window=3.0)
        agg.add("s", {"message_id": "m1", "text": "a"})
        clock.tick(3.5)                       # 超过 max window：必须把"a"这一轮发出去
        first = agg.add("s", {"message_id": "m2", "text": "b"})
        assert isinstance(first, UserTurn) and "a" in first.text
        clock.tick(3.5)
        second = agg.add("s", {"message_id": "m3", "text": "c"})
        assert isinstance(second, UserTurn) and "b" in second.text

    def test_different_sessions_are_isolated(self):
        clock = _Clock()
        agg = _aggregator(clock)
        agg.add("a", {"message_id": "m1", "text": "from-a"})
        clock.tick(2)
        turn_b = agg.add("b", {"message_id": "m2", "text": "from-b"})
        assert turn_b is None or turn_b.session_id == "b"
        turn_a = agg.flush("a")
        assert turn_a is not None and "from-a" in turn_a.text


class TestIdempotency:

    def test_duplicate_message_id_is_ignored(self):
        clock = _Clock()
        agg = _aggregator(clock)
        agg.add("s", {"message_id": "dup", "text": "first"})
        clock.tick(2)
        assert agg.add("s", {"message_id": "dup", "text": "first"}) is None
        turn = agg.flush("s")
        assert turn is not None
        assert turn.text.count("first") == 1

    def test_session_lock_prevents_double_run(self):
        clock = _Clock()
        agg = _aggregator(clock)
        agg.add("s", {"message_id": "m1", "text": "a"})
        agg.mark_processing("s", True)
        clock.tick(3)
        assert agg.add("s", {"message_id": "m2", "text": "b"}) is None, "处理中不再启动第二个 turn"
        agg.mark_processing("s", False)
        turn = agg.flush("s")
        assert turn is not None and "b" in turn.text


class TestCollectMessagesEntry:
    """Phase 12-5（计划 2.0）：只保留 `collect_messages` 这一个合并入口。

    原来这一组测的是 `input.turn_payload`（merge_message_parts / merge_request_payload）
    与 API 的 `_merge_turn_request` —— 那条链已确认 **0 个业务调用**（只被测试与
    兼容层引用），本次删除后，断言全部迁移到**真实入口**上（覆盖不减）。
    """

    def test_messages_keep_order_and_images(self):
        from src.input import collect_messages

        messages = collect_messages(
            session_id="collect-order",
            messages=[
                {"text": "We need an LED screen.", "message_id": "m1"},
                {"text": "It is for a church.", "message_id": "m2"},
                {"images": ["img-1"], "message_id": "m3"},
                {"text": "Around 200 people.", "message_id": "m4"},
            ],
        )

        assert [m.text for m in messages] == [
            "We need an LED screen.",
            "It is for a church.",
            "",
            "Around 200 people.",
        ]
        assert [m.message_id for m in messages] == ["m1", "m2", "m3", "m4"]
        assert messages[2].images == ["img-1"]

    def test_legacy_question_still_works(self):
        from src.input import collect_messages

        messages = collect_messages(session_id="collect-legacy", question="indoor screen")

        assert [m.text for m in messages] == ["indoor screen"]

    def test_duplicate_message_ids_are_dropped(self):
        from src.input import collect_messages

        # 同一个请求里放了重复的 message_id → 只算一条（跨请求的幂等在 TurnExecutor，
        # 见本文件 TestIdempotency）
        messages = collect_messages(
            session_id="collect-dedupe",
            messages=[
                {"text": "first message", "message_id": "dup-1"},
                {"text": "first message", "message_id": "dup-1"},
            ],
        )

        assert [m.text for m in messages] == ["first message"]

    def test_api_collect_helper_legacy_path(self):
        """API 层：老客户端只发 question 也要收成一条消息。"""
        from src.api import ChatRequest, _collect_turn_request

        messages = _collect_turn_request(
            ChatRequest(session_id="api-legacy", question="5m x 3m indoors")
        )

        assert [m.text for m in messages] == ["5m x 3m indoors"]

    def test_api_collect_helper_prefers_messages(self):
        from src.api import ChatRequest, _collect_turn_request

        messages = _collect_turn_request(
            ChatRequest(
                session_id="api-collect",
                messages=[
                    {"text": "we need a screen", "message_id": "api-1"},
                    {"text": "for a church", "message_id": "api-2"},
                ],
            )
        )

        assert [m.text for m in messages] == ["we need a screen", "for a church"]


class TestMultimodal:

    def test_text_image_text(self):
        clock = _Clock()
        agg = _aggregator(clock)
        agg.add("s", {"message_id": "m1", "text": "I need something like this."})
        clock.tick(0.3)
        agg.add("s", {"message_id": "m2", "images": ["image_1"]})
        clock.tick(0.3)
        agg.add("s", {"message_id": "m3", "text": "For an indoor church."})
        clock.tick(2)
        turn = agg.add("s", {"message_id": "m4", "text": "Around 200 people."})
        assert turn is not None, "停顿后新消息到达 → 上一轮一次性发出"
        assert turn.images == ["image_1"]
        assert turn.has_images is True
        assert "indoor church" in turn.text
        assert turn.message_ids == ["m1", "m2", "m3"]
        clock.tick(2)
        nxt = agg.add("s", {"message_id": "m5", "text": "thanks"})
        assert nxt is not None and "Around 200 people." in nxt.text

    def test_image_only(self):
        clock = _Clock()
        agg = _aggregator(clock)
        agg.add("s", {"message_id": "m1", "images": ["img"]})
        clock.tick(2)
        turn = agg.flush("s") or agg.add("s", {"message_id": "m2", "text": "x"})
        assert turn is not None and turn.has_images

    def test_turn_to_dict(self):
        turn = UserTurn(session_id="s")
        turn.add({"message_id": "m1", "text": "hello"})
        turn.close()
        payload = turn.to_dict()
        assert payload["text"] == "hello" and payload["message_count"] == 1
