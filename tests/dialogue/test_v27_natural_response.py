"""v2.7 §16（Phase 14）：机械确认话术必须被清掉。"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import (  # noqa: E402
    find_mechanical_prefixes,
    is_mechanical,
    strip_mechanical_phrases,
)


class TestMechanicalPhrases:

    def test_detects_banned_openers(self):
        for text in (
            "Got it. What size do you need?",
            "Thanks. What size do you need?",
            "Understood. What size?",
            "Perfect, what pitch do you want?",
            "Based on that, I recommend P3.",
            "Thank you for providing that detail.",
        ):
            assert find_mechanical_prefixes(text), text

    def test_strips_openers(self):
        text, removed = strip_mechanical_phrases("Got it. What size do you need?")
        assert removed == ["Got it"]
        assert text == "What size do you need?"

    def test_strips_multiple_openers(self):
        text, removed = strip_mechanical_phrases("Got it, thanks. What size?")
        assert len(removed) >= 1
        assert text.endswith("What size?")

    def test_allow_keeps_them(self):
        text, removed = strip_mechanical_phrases("Got it. Thanks for the details.", allow=True)
        assert removed == []
        assert text.startswith("Got it")

    def test_plain_question_is_untouched(self):
        text, removed = strip_mechanical_phrases("Will the screen be installed indoors?")
        assert removed == []
        assert text == "Will the screen be installed indoors?"

    def test_pure_ack_is_dropped(self):
        text, removed = strip_mechanical_phrases("Got it.")
        assert text == ""
        assert removed == ["Got it"]

    def test_is_mechanical_helper(self):
        assert is_mechanical("Thanks. Anything else?") is True
        assert is_mechanical("Which pitch do you want?") is False
