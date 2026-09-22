"""
交付时间问答（客户口径）回归测试：

  - 客户问交期 → 从下单付款开始计算，常规交付时间约 15–30 天；
  - 客户要求加快 → 可以走空运（能提前，但成本会增加）；
  - 客户说"想 11 月安装" → 先接住档期，再把交期说清楚（不承诺具体日期）。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rag.delivery_info import (  # noqa: E402
    delivery_answer,
    extract_timing_phrase,
    install_timing_note,
    is_delivery_question,
    wants_faster_delivery,
)
from tests._cases import assert_all_cases  # noqa: E402


def _is_delivery_question(message: str) -> None:
    assert is_delivery_question(message), message


def _wants_faster(message: str) -> None:
    assert wants_faster_delivery(message), message


class TestDeliveryQuestionDetection:

    # 用例表（2026-09-22 瘦身：一条测试跑整张表）
    DELIVERY_QUESTIONS = [
        "你们多久能发货？",
        "交期大概多久",
        "什么时候能交货",
        "delivery time?",
        "what is your lead time?",
        "how long does delivery take?",
        "when can you deliver?",
    ]
    FASTER_REQUESTS = [
        "can you ship faster?",
        "我们比较急，能加快吗",
        "需要尽快到货",
        "we need it asap",
    ]

    def test_delivery_questions(self):
        assert_all_cases(
            self.DELIVERY_QUESTIONS, _is_delivery_question, label="message"
        )

    def test_faster_requests(self):
        assert_all_cases(self.FASTER_REQUESTS, _wants_faster, label="message")

    def test_non_delivery_messages(self):
        assert not is_delivery_question("我需要室内会议室的屏")
        assert not is_delivery_question("do you have P1.2 COB LED?")


class TestDeliveryAnswers:

    def test_lead_time_is_15_to_30_days(self):
        for seed in range(4):
            answer = delivery_answer("你们多久能发货？", language="en", seed=seed)
            assert answer
            lowered = answer.lower()
            assert "15" in lowered and "30" in lowered, answer
            assert "order" in lowered, answer
            assert any(word in lowered for word in ("pay", "paid", "payment")), answer

    def test_chinese_lead_time_answer(self):
        answer = delivery_answer("交期多久？", language="zh", seed=0)
        assert "15" in answer and "30" in answer
        assert "下单付款" in answer

    def test_faster_means_air_freight_with_extra_cost(self):
        answer = delivery_answer("can you ship faster?", language="en", seed=0)
        lowered = answer.lower()
        assert "air" in lowered, answer
        assert any(word in lowered for word in ("cost", "price", "expensive")), answer

        zh = delivery_answer("能不能加急？", language="zh", seed=0)
        assert "空运" in zh and "成本" in zh

    def test_question_plus_faster_gives_both(self):
        answer = delivery_answer("交期多久？我们比较急，能加快吗", language="zh", seed=0)
        assert "15" in answer and "30" in answer
        assert "空运" in answer

    def test_unrelated_message_returns_none(self):
        assert delivery_answer("我需要室内会议室的屏") is None
        assert install_timing_note("我需要室内会议室的屏") is None


class TestInstallTiming:

    def test_extracts_month_phrase(self):
        assert extract_timing_phrase("i wanna buy a display, and wanna install 11月") == "11月"
        assert extract_timing_phrase("need it by December") == "December"

    def test_timing_note_mentions_lead_time(self):
        note = install_timing_note("i wanna buy a display, and wanna install 11月", language="en")
        assert "11月" in note
        assert "15" in note and "30" in note

    def test_chinese_timing_note(self):
        note = install_timing_note("我们想11月安装", language="zh")
        assert "11月" in note
        assert "15" in note and "30" in note


class TestScriptGeneratorDeliveryBranch:
    """节点级：客户问交期 / 提档期时，答复 + 继续问需求（不能只顾着追问）。"""

    def _state(self, message, pending="Is the installation going to be indoors or outdoors?"):
        return {
            "messages": [{"role": "user", "content": message}],
            "current_message": message,
            "session_id": "delivery-branch",
            "intent": "need_query",
            "next_action": "ask",
            "requirements": {},
            "additional_requirements": [],
            "should_generate_solution": False,
            "response": "",
            "pending_question": pending,
            "pending_slot": "environment",
        }

    def test_delivery_question_is_answered_without_forced_question(self):
        from src.agents.sales.nodes.script_generator import script_generator

        out = script_generator(self._state("你们多久能发货？"))
        reply = out["response"]
        assert "15" in reply and "30" in reply, reply
        assert "indoors or outdoors" not in reply, reply   # v2.5++++ 计划 §16.5：answer_only，不硬塞需求问题
        assert out["next_action"] == "ask"

    def test_install_month_is_acknowledged(self):
        """复刻客户日志：'i wanna buy a disaply ,and wanna install 11月'"""
        from src.agents.sales.nodes.script_generator import script_generator

        out = script_generator(
            self._state("i wanna  buy a disaply ,and wanna install 11月")
        )
        reply = out["response"]
        assert "11月" in reply, reply
        assert "15" in reply and "30" in reply, reply
        assert "indoors or outdoors" in reply, reply

    def test_faster_request_mentions_air_freight(self):
        from src.agents.sales.nodes.script_generator import script_generator

        out = script_generator(self._state("交期要多久？我们比较急，能不能加快"))
        reply = out["response"].lower()
        assert "air" in reply or "空运" in reply, reply
        assert "cost" in reply or "成本" in reply, reply

    def test_normal_question_unaffected(self):
        from src.agents.sales.nodes.script_generator import script_generator

        out = script_generator(self._state("it is for a conference room"))
        assert "15–30" not in out["response"]
