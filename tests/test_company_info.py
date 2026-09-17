"""
公司信息问答回归测试（客户实测反馈）。

实测 bug：客户问 "Do you have a representative in western India"，
系统回 "Yes — we do carry an LED."（答非所问 + 凭空说 Yes），
完全没有使用 data/company_profile.txt 里的公司信息（iSEMC / Shenzhen, China）。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rag.company_info import company_answer, is_company_question  # noqa: E402
from src.rag.reply_composer import availability_answer, compose_requirement_reply  # noqa: E402


class TestAgentDistributorAnswerContent:
    """客户问"有没有代理商/经销商"时的回答内容（客户口径）：

    ① 只有中国（深圳）这一个公司/工厂，当地没有代理或办事处；
    ② 工厂是我们自己的，没有中间环节 → 开销更低；
    ③ 海外客户由深圳团队直接对接。
    且不能每次都同一句话（按轮次换说法）。
    """

    def test_chinese_answer_mentions_factory_and_cost(self):
        answer = company_answer("你们在肯尼亚有代理商吗？", language="zh", seed=0)
        assert answer
        assert "工厂" in answer, answer
        assert any(word in answer for word in ("成本", "开销", "中间")), answer
        assert any(word in answer for word in ("只有", "没有")), answer
        assert "Shenzhen" in answer or "深圳" in answer

    def test_english_answer_mentions_factory_and_cost(self):
        answer = company_answer("Do you have a distributor in Kenya?", language="en", seed=0)
        assert answer
        assert "factory" in answer.lower()
        assert any(word in answer.lower() for word in ("cost", "price", "middleman")), answer
        assert "Shenzhen" in answer

    def test_wording_varies_by_seed(self):
        answers = {
            company_answer("你们在肯尼亚有代理商吗？", language="zh", seed=seed)
            for seed in range(4)
        }
        assert len(answers) == 4, "不能每次都同一句固定话术"
        for answer in answers:
            assert "工厂" in answer and ("成本" in answer or "开销" in answer)

    def test_non_company_question_returns_none(self):
        assert company_answer("do you have P1.2 COB LED", language="en") is None


class TestCompanyQuestionDetection:

    @pytest.mark.parametrize("message", [
        "Let me ask. Do you have a representative in western India",
        "do you have an office in Europe?",
        "are you a local distributor in Nigeria",
        "你们在肯尼亚有代理商吗？",
        "where are you located?",
        "what is your company address",
        "你们在深圳有工厂吗",
        "你们公司在哪里",
        "有没有当地的经销商",
    ])
    def test_company_questions(self, message):
        assert is_company_question(message), message

    @pytest.mark.parametrize("message", [
        "do you have P1.2 COB LED",
        "we need an indoor LED display for a church",
        "is it a fixed installation?",
        "how much is a P2.5 screen",
    ])
    def test_non_company_questions(self, message):
        assert not is_company_question(message), message


class TestCompanyAnswerIsGrounded:

    def test_representative_question_uses_company_profile(self):
        answer = company_answer("Do you have a representative in western India")
        assert answer
        assert "Shenzhen" in answer
        # 不允许凭空说"有"当地办事处：必须说明只有深圳这一个所在地
        lowered = answer.lower()
        assert any(
            phrase in lowered
            for phrase in ("only office", "single site", "only site", "only location")
        ), answer
        assert "distributor" in lowered or "branch" in lowered or "agent" in lowered

    def test_variants_rotate(self):
        texts = {company_answer("where are you located?", seed=seed) for seed in range(6)}
        assert len(texts) >= 2
        assert all("Shenzhen" in text for text in texts)

    def test_chinese_answer(self):
        answer = company_answer("你们公司在哪里", language="zh")
        assert answer and "深圳" in answer and ("只有" in answer or "唯一" in answer)

    def test_returns_none_for_non_company_question(self):
        assert company_answer("we need an indoor LED display") is None


class TestAvailabilityAnswerDoesNotHijackCompanyQuestions:
    """回归：公司类问题不能被"有没有某规格"的核实逻辑抢答。"""

    def test_representative_question_is_not_availability(self):
        assert availability_answer("Do you have a representative in western India") is None
        assert availability_answer("do you have an office in Europe") is None

    def test_generic_display_question_gives_no_false_yes(self):
        """只说 "LED display"（没有具体规格）时不能回 "Yes — we do carry an LED."。"""
        assert availability_answer("do you have an LED display") is None

    def test_specific_spec_still_answered(self):
        assert "1.2" in (availability_answer("do you have P1.2 COB LED") or "")
        assert "2.5" in (availability_answer("do you have a P2.5 LED screen") or "")


class TestComposedCompanyReply:

    def test_natural_bridge_before_next_question(self):
        """回答完公司/价格类问题后要有自然过渡，不能硬接问句。"""
        from src.rag.reply_composer import _BRIDGES

        text = compose_requirement_reply(
            question="Is it a permanent install, or is it for rental/events?",
            slot="installation",
            message="Do you have a representative in western India",
            language="en",
            seed=0,
        )
        assert any(bridge in text for bridge in _BRIDGES["en"]), text

    def test_echo_of_requirement_also_gets_a_bridge(self):
        """客户口径：接话（复述需求）和提问之间也必须有过渡，不能两句硬拼。"""
        from src.rag.reply_composer import _BRIDGES

        text = compose_requirement_reply(
            question="Is it a permanent install, or is it for rental/events?",
            slot="installation",
            message="we need it for a church",
            language="en",
            seed=0,
        )
        assert any(bridge in text for bridge in _BRIDGES["en"]), text

    def test_company_answer_plus_next_requirement_question(self):
        question = "Is this a permanent installation or a rental setup?"
        text = compose_requirement_reply(
            question=question,
            slot="installation",
            message="Do you have a representative in western India",
            language="en",
            seed=0,
        )
        assert "Shenzhen" in text
        assert "we do carry an LED" not in text
        assert text.rstrip().endswith("?")

    def test_script_generator_answers_company_question(self):
        from src.agents.sales.nodes.script_generator import script_generator

        state = {
            "messages": [
                {"role": "user", "content": "Do you have a representative in western India"}
            ],
            "current_message": "Do you have a representative in western India",
            "session_id": "company-question-session",
            "intent": "need_query",
            "next_action": "ask",
            "requirements": {"usage": "church", "location_type": "室内", "display_type": "LED"},
            "additional_requirements": [],
            "should_generate_solution": False,
            "response": "",
            "pending_question": "Is this a permanent installation or a rental setup?",
            "pending_slot": "installation",
            "suppress_greeting": True,
        }
        result = script_generator(state)

        assert result["next_action"] == "ask"
        assert "Shenzhen" in result["response"]
        assert "we do carry an LED" not in result["response"]
        assert result["response"].rstrip().endswith("?")

    def test_script_generator_others_branch_company_question(self):
        from src.agents.sales.nodes.script_generator import script_generator

        state = {
            "messages": [{"role": "user", "content": "where are you located?"}],
            "current_message": "where are you located?",
            "session_id": "company-question-others",
            "intent": "others",
            "next_action": "others",
            "requirements": {"usage": "church", "location_type": "室内"},
            "additional_requirements": [],
            "should_generate_solution": False,
            "response": "",
            "pending_question": "Is this a permanent installation or a rental setup?",
            "pending_slot": "installation",
            "suppress_greeting": False,
        }
        result = script_generator(state)

        # 不再走没有公司资料的自由问答
        assert result["next_action"] == "ask"
        assert "Shenzhen" in result["response"]
