"""客户口径（2026-09-21）：授权 LLM 自己组织接话和问法，但**不许说不存在的事**。

覆盖三件事：

    ① 问句库不再自带模板铺垫（"Quick one —"、"Just so I …"）
    ② 客户口径（2026-09-21 追加）：**不要破折号，统一用逗号**
    ③ LLM 自己组织说法（不再照抄模板）；校验失败先让 LLM 重写一次，
       仍然越界（编造型号 / 事实）才退回模板 —— 事实安全永远优先于"自然"
"""
import os
import sys
from types import SimpleNamespace

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import build_context, compose_from_context, generate_response  # noqa: E402
from src.dialogue.response_validator import validate_response  # noqa: E402
from src.rag.readiness import QUESTION_VARIANTS, question_for, question_intent  # noqa: E402
from src.rag.rerank import sanitize_customer_response  # noqa: E402

_CANNED_PREAMBLES = (
    "quick one",
    "quick check",
    "one quick question",
    "just so i",
    "while we're at it",
    "in the meantime",
    "to narrow it down",
)


@pytest.fixture(autouse=True)
def _reset_llm_breaker():
    """其它用例如果触发过真实 LLM 失败，熔断会冷却 60s —— 这里每例前重置，
    保证"提示词/授权"这类断言与执行顺序无关。"""
    from src.dialogue import reset_llm_breaker

    reset_llm_breaker()
    yield
    reset_llm_breaker()


class _FakeLLM:
    """按顺序返回预设回复，并记录收到的 prompt（用于断言是否重写）。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(str(prompt))
        text = self.replies.pop(0) if self.replies else ""
        return SimpleNamespace(content=text)


def _context(**overrides):
    base = dict(
        action="ASK",
        customer_message="3*5",
        question="Will the screen be installed indoors or outdoors?",
        question_slot="environment",
        question_intent=question_intent("environment"),
        required_question="environment",
        missing_fields=["environment"],
        language="en",
        restrictions=["ask_only_one_question"],
    )
    base.update(overrides)
    return build_context(**base)


class TestQuestionBankIsASeedNotATemplate:

    def test_no_canned_preamble_anywhere(self):
        for slot, variants in QUESTION_VARIANTS.items():
            for seed in range(12):
                question = question_for(slot, "en", seed)
                if not question:
                    continue
                lowered = question.lower()
                assert not lowered.startswith(_CANNED_PREAMBLES), (slot, seed, question)

    def test_intent_is_available_per_slot(self):
        assert "室内" in question_intent("environment")
        assert "租赁" in question_intent("installation")


class TestCleaningUsesCommasInsteadOfDashes:

    def test_em_dash_becomes_a_comma(self):
        text = sanitize_customer_response("Fixed install — or is it for rental events?")
        assert "—" not in text and "–" not in text, text
        assert text == "Fixed install, or is it for rental events?", text

    def test_no_double_commas(self):
        text = sanitize_customer_response("Pricing depends on the model —— then I quote.")
        assert ", ," not in text and ",," not in text
        assert "—" not in text


class TestValidatorKeepsIntentAndFactsHonest:

    def test_question_about_the_wrong_topic_is_flagged(self):
        result = validate_response(
            "Which pitch do you want?",
            question_slot="environment",
            required_question="environment",
        )
        assert "question_intent_mismatch" in result.issues

    def test_same_question_phrasing_is_flagged(self):
        result = validate_response(
            "Will the screen be installed indoors or outdoors?",
            question_slot="environment",
            recent_questions=["Will the screen be installed indoors or outdoors?"],
        )
        assert "repeated_question_phrasing" in result.issues

    def test_a_fresh_phrasing_passes(self):
        result = validate_response(
            "Is this one going indoors, or will it live outside?",
            question_slot="environment",
            recent_questions=["Will the screen be installed indoors or outdoors?"],
        )
        assert result.issues == [], result.issues


class TestLLMIsAuthorizedToPhraseItItself:

    def test_natural_variation_is_used_as_is(self):
        llm = _FakeLLM(["Indoors or outdoors? That decides which cabinet I'd quote."])
        text = generate_response(_context(), llm=llm)
        assert text == "Indoors or outdoors? That decides which cabinet I'd quote."
        assert len(llm.prompts) == 1, "合格就不该再重写"

    def test_wrong_topic_is_rewritten_once_then_used(self):
        llm = _FakeLLM([
            "Which pitch do you want?",              # 问错主题 → 要求重写
            "Indoors or outdoors — which one is it?",  # 重写后合格
        ])
        text = generate_response(_context(), llm=llm)
        assert text == "Indoors or outdoors — which one is it?"
        assert len(llm.prompts) == 2, "应触发一次重写"
        assert "Problems to fix" in llm.prompts[1]

    def test_invented_model_never_reaches_the_customer(self):
        """编造型号 / 编造事实 → 重写仍越界 → 退回模板（事实安全优先）。"""
        llm = _FakeLLM([
            "We'd use model TW99-X with a 5-year warranty.",
            "Model TW99-X is perfect and the price is 1000 USD.",
        ])
        context = _context()
        text = generate_response(context, llm=llm)
        assert "TW99" not in text
        assert text == compose_from_context(context, seed=0) or text in (
            context.question,
        ) or "indoors or outdoors" in text.lower()

    def test_prompt_tells_the_model_phrasing_is_its_own(self):
        llm = _FakeLLM(["Indoors or outdoors?"])
        generate_response(_context(), llm=llm)
        prompt = llm.prompts[0]
        assert "WORDING is entirely yours" in prompt
        assert "do NOT copy it verbatim" in prompt
        assert "explicitly allowed to" in prompt

    def test_prompt_carries_previous_phrasings(self):
        llm = _FakeLLM(["Is this indoors or outdoors?"])
        context = _context(recent_questions=["Will the screen be installed indoors or outdoors?"])
        generate_response(context, llm=llm)
        assert "do not repeat them" in llm.prompts[0]
