"""闲聊要按语境回一句，不许一直念自己的模板（客户口径 2026-09-30）。

客户原话：

    「没有回答客户说的话，就一直说自己的话，而且就算僵硬的一种话术，
      这个需要优化，让 ai 自己根据语境上下文去回答，而且必须回答客户的话，
      也要回答有利于自己的话术。」

实测：推荐完 DS-O-75 之后——

    客户: 今天天气不从        → "I can help you look up product specs or recommend the
                                 right display based on your needs. What would you like
                                 to know or discuss?"
    客户: do u like watching tv → "Hello! I can help you look up display specs or
                                 recommend the right product based on your use case."

那两句是 Solution `conversation_node` 里**写死的模板**。现在改成模型按语境生成。
"""
import os
import sys
from types import SimpleNamespace

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue.chat_reply import generate_chat_reply  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402

_CANNED = "I can help you look up product specs or recommend the right display"


def _lcd() -> RequirementProfile:
    return RequirementProfile.from_slots(
        {"display_type": "LCD", "lcd_category": "advertising"},
        explicit_keys={"display_type", "lcd_category"},
    )


@pytest.fixture
def fake_llm(monkeypatch):
    captured = {}

    class _Fake:
        def invoke(self, prompt, *args, **kwargs):
            captured["prompt"] = str(prompt)
            return SimpleNamespace(
                content="Ha, it's a lovely one — I'm here whenever you want to pick "
                "the screen back up or go through the quotation."
            )

    monkeypatch.setattr("src.core.llm.get_llm", lambda *a, **k: _Fake())
    return captured


@pytest.fixture
def broken_llm(monkeypatch):
    class _Boom:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("no network")

    monkeypatch.setattr("src.core.llm.get_llm", lambda *a, **k: _Boom())


class TestLcdChatUsesTheModel:
    def test_chat_reply_is_generated_from_context(self, fake_llm):
        from src.agents.solution.nodes.intent import conversation_node

        state = {
            "current_message": "今天天气不从",
            "requirement_profile": _lcd(),
            "session_id": "chat-1",
            "messages": [],
        }
        out = conversation_node(state)

        assert _CANNED not in out["recommendation"], "不许再回写死的模板"
        assert "lovely" in out["recommendation"], out["recommendation"]
        # 客户这句话必须进 prompt，模型才有机会接住它
        assert "今天天气不从" in fake_llm["prompt"]
        # 规则要写死在 prompt 里：接住客户 + 不编造 + 对我方有利
        for rule in ("Answer what the customer actually said", "Never invent facts",
                     "favourable to us"):
            assert rule in fake_llm["prompt"], rule

    def test_tv_question_is_also_answered(self, fake_llm):
        from src.agents.solution.nodes.intent import conversation_node

        out = conversation_node(
            {
                "current_message": "do u like watching tv",
                "requirement_profile": _lcd(),
                "session_id": "chat-2",
                "messages": [],
            }
        )

        assert _CANNED not in out["recommendation"]
        assert "watching tv" in fake_llm["prompt"]

    def test_falls_back_to_the_template_when_the_model_is_unavailable(self, broken_llm):
        from src.agents.solution.nodes.intent import conversation_node

        out = conversation_node(
            {
                "current_message": "今天天气不从",
                "requirement_profile": _lcd(),
                "session_id": "chat-3",
                "messages": [],
            }
        )

        assert out["recommendation"].strip(), "模型不可用时也不能给空回复"
        assert _CANNED in out["recommendation"]


class TestLedChatIsUntouched:
    def test_led_keeps_its_own_replies(self, fake_llm):
        from src.agents.solution.nodes.intent import conversation_node

        led = RequirementProfile.from_slots(
            {"display_type": "LED"}, explicit_keys={"display_type"}
        )
        out = conversation_node(
            {
                "current_message": "hello there",
                "requirement_profile": led,
                "session_id": "chat-led",
                "messages": [],
            }
        )

        assert out["recommendation"].startswith("Hello! I can help you look up display specs")


class TestSalesSideOfftopicAck:
    def test_offtopic_ack_is_generated_too(self, fake_llm):
        from src.dialogue.response_coordinator import ResponseCoordinator

        coordinator = ResponseCoordinator(profile_lookup=lambda _s: _lcd())
        out = coordinator.finalize(
            "product pitch",
            session_id="chat-4",
            message="今天天气不从",
            product_family="lcd",
            offtopic_turn=True,
        )

        assert "product pitch" not in out
        assert "lovely" in out, out

    def test_offtopic_ack_has_a_fallback(self, broken_llm):
        from src.dialogue.response_coordinator import ResponseCoordinator

        coordinator = ResponseCoordinator(profile_lookup=lambda _s: _lcd())
        out = coordinator.finalize(
            "product pitch",
            session_id="chat-5",
            message="今天天气不从",
            product_family="lcd",
            offtopic_turn=True,
        )

        assert out.strip()
        assert "product pitch" not in out


class TestChatReplyHelper:
    def test_empty_message_is_not_sent_to_the_model(self, fake_llm):
        assert generate_chat_reply("", session_id="chat-6") == ""
