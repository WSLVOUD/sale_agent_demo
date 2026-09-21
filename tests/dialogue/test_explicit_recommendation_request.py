"""客户口径（2026-09-21）：客户明确说"给我推荐 / 帮我选 / 报价"时，必须走推荐链路。

实测 bug：客户说"给我推荐"，LLM 把它判成 `product_question` → 被当成"提问"走了
自由问答，只回一句"我这就给你准备"，**一个产品都没推荐**。
（旧的覆盖规则只覆盖 others / industry，漏掉了 product_question。）
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import importlib  # noqa: E402

classify_module = importlib.import_module("src.agents.sales.nodes.classify")
from src.agents.sales.nodes.classify import detect_intent  # noqa: E402


class _FakeLLM:
    """把 LLM 的意图分类固定成指定值（模拟真实 LLM 的误判）。"""

    def __init__(self, intent: str):
        self.intent = intent

    def invoke(self, *_args, **_kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(content=self.intent)


def _classify(message: str, llm_intent: str, monkeypatch) -> str:
    # classify 节点内部直接 new ChatOpenAI(...) → 换掉它即可注入固定的分类结果
    monkeypatch.setattr(classify_module, "ChatOpenAI", lambda **_kwargs: _FakeLLM(llm_intent))
    state = {
        "current_message": message,
        "messages": [{"role": "user", "content": message}],
        "session_id": "explicit-reco",
        "requirements": {},
    }
    return classify_module.classify(state)["intent"]


class TestExplicitRecommendationWins:

    def test_rule_detector_recognises_the_request(self):
        assert detect_intent("给我推荐") == "recommendation"
        assert detect_intent("帮我推荐一个") == "recommendation"
        assert detect_intent("recommend me one") == "recommendation"

    def test_product_question_misclassification_is_overridden(self, monkeypatch):
        assert _classify("给我推荐", "product_question", monkeypatch) == "need_query"

    def test_objection_misclassification_is_overridden(self, monkeypatch):
        assert _classify("给我推荐一款", "objection", monkeypatch) == "need_query"

    def test_others_is_still_overridden(self, monkeypatch):
        assert _classify("给我推荐", "others", monkeypatch) == "need_query"

    def test_plain_requirement_answer_stays_need_query(self, monkeypatch):
        assert _classify("indoor", "need_query", monkeypatch) == "need_query"

    def test_explicit_closing_is_not_turned_into_a_recommendation(self, monkeypatch):
        """客户明确结束对话时，不因为句子里带了"推荐"就强行推荐。"""
        intent = _classify("不用了，谢谢，再见", "closing", monkeypatch)
        assert intent == "closing"
