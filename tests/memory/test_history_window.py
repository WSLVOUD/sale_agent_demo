"""会话记忆窗口：最近 50 条以内、按长度预算裁剪、只保留真实对话。"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.memory.store import memory  # noqa: E402
from src.memory.history_window import (  # noqa: E402
    get_dialogue_window,
    render_dialogue_window,
)


class TestDialogueWindow:

    def setup_method(self):
        memory.clear("hw-test")

    def teardown_method(self):
        memory.clear("hw-test")

    def test_keeps_user_and_assistant_in_order(self):
        memory.add("hw-test", "user", "i need a led display")
        memory.add("hw-test", "assistant", "Will the screen be installed indoors or outdoors?")
        memory.add("hw-test", "user", "indoor")
        window = get_dialogue_window("hw-test")
        assert [item["role"] for item in window] == ["user", "assistant", "user"]
        assert window[-1]["content"] == "indoor"

    def test_filters_non_dialogue_entries(self):
        memory.add("hw-test", "system", "internal note")
        memory.add("hw-test", "user", "hello")
        window = get_dialogue_window("hw-test")
        assert len(window) == 1 and window[0]["content"] == "hello"

    def test_limits_to_the_configured_count(self):
        for index in range(80):
            memory.add("hw-test", "user", f"msg-{index}")
        window = get_dialogue_window("hw-test", limit=50)
        assert len(window) == 50
        assert window[-1]["content"] == "msg-79"
        assert window[0]["content"] == "msg-30"

    def test_respects_character_budget(self):
        for index in range(50):
            memory.add("hw-test", "user", "x" * 500)
        window = get_dialogue_window("hw-test", limit=50, per_message=300, total=1500)
        assert sum(len(item["content"]) for item in window) <= 1500 + 300
        assert all(len(item["content"]) <= 301 for item in window)

    def test_render_marks_roles(self):
        memory.add("hw-test", "user", "indoor")
        memory.add("hw-test", "assistant", "What size?")
        text = render_dialogue_window(get_dialogue_window("hw-test"))
        assert "客户: indoor" in text and "AI: What size?" in text

    def test_unknown_session_is_empty(self):
        assert get_dialogue_window("hw-missing") == []
