"""v2.7 §4.3：请求里的消息收集（真实日志 bug 回归）。

实测（2026-09-21 真实日志）：前端同时发 `messages[]` 和兼容字段 `question`
（内容相同），旧实现把两者都收下 → 一句话变成两条：

    "message_ids": ["m-1789978546098-0", "m-1789978546098-0"]
    "last_customer_message": "hi\\nhi"
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.input.message import collect_messages, join_text  # noqa: E402


class TestCollectMessages:

    def test_legacy_question_is_ignored_when_messages_carry_it(self):
        messages = collect_messages(
            session_id="s1",
            question="3 * 5",
            messages=[{"text": "3 * 5", "message_id": "m-1"}],
            message_ids=["m-1"],
        )
        assert len(messages) == 1
        assert join_text(messages) == "3 * 5"

    def test_duplicate_message_ids_are_collapsed(self):
        messages = collect_messages(
            session_id="s1",
            messages=[
                {"text": "hi", "message_id": "m-1"},
                {"text": "hi", "message_id": "m-1"},
            ],
        )
        assert len(messages) == 1

    def test_legacy_only_client_still_works(self):
        messages = collect_messages(
            session_id="s1", question="i need a led display", message_ids=["legacy-1"]
        )
        assert len(messages) == 1
        assert messages[0].text == "i need a led display"
        assert messages[0].message_id == "legacy-1"

    def test_images_are_kept(self):
        messages = collect_messages(
            session_id="s1",
            images=["data:image/png;base64,AAA"],
            messages=[{"text": "look at this", "message_id": "m-9"}],
        )
        assert len(messages) == 2 or messages[0].images == ["data:image/png;base64,AAA"]
        assert any(message.images for message in messages)

    def test_batch_of_messages_keeps_order(self):
        messages = collect_messages(
            session_id="s1",
            messages=[
                {"text": "i need a led display", "message_id": "b1"},
                {"text": "indoor", "message_id": "b2"},
                {"text": "3*5", "message_id": "b3"},
            ],
        )
        assert [message.message_id for message in messages] == ["b1", "b2", "b3"]
        assert join_text(messages) == "i need a led display\nindoor\n3*5"
