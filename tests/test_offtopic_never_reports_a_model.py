"""回归：跑题/闲聊**不许**被包装成"符合你需求的型号"。

客户口径（2026-10 实测）：

    客户: do u like watching TV          （需求已经推荐完）
    AI  : Based on your requirements, the closest match is TW21-3216-P2.5.
    AI  : （再发一次）还是同一句

日志里那条链路是确定的：

    next_action == "others" → 无条件送**产品 RAG**
    → 检索回来 6 条产品片段 → 模型只聊型号
    → 校验器判 'customer_question_not_answered' → 结构化拼装
    → ModelGuard 把型号全抹掉 → 正文变空
    → api 空回复兜底 product_fallback_answer → 报型号

两处修复：
  ① orchestrator 的 ``others`` 分支：不是业务/产品问题 → 按语境接话并拉回产品，
     且 ``products`` 置空（不拿型号糊弄客户）；
  ② api 空回复兜底：跑题轮绝不报型号，只给"接住 + 拉回"的过渡句。
"""
import os
import re
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestOffTopicIsNotABusinessQuestion:

    def test_chit_chat_is_not_a_business_question(self):
        """这条判定是分流依据：跑题 → 不进产品 RAG。"""
        from src.agents.sales.nodes.requirement import _is_product_or_business_question

        for message in ("do u like watching TV", "how's it going?",
                        "i need to buy shoes", "can u give me skirt",
                        "Suspected Chinese hackers impersonated an Anthropic employee"):
            assert _is_product_or_business_question(message) is False, message

    def test_real_business_questions_still_go_to_the_answer_path(self):
        from src.agents.sales.nodes.requirement import _is_product_or_business_question

        for message in ("do you have a distributor in Indonesia?",
                        "what is the price for a 3x5 LED screen?",
                        "what is the warranty?"):
            assert _is_product_or_business_question(message) is True, message


class TestOrchestratorDoesNotSendChitChatToProductRag:

    def test_the_others_branch_guards_offtopic_before_calling_solution(self):
        source = open(
            os.path.join(project_root, "src", "orchestrator.py"), encoding="utf-8"
        ).read()
        # 分流判定必须在 others 分支里、且在调用 solution_agent 之前
        branch = source.split('elif next_action == "others":', 1)
        assert len(branch) == 2, "others 分支不见了"
        body = branch[1][:8000]
        assert "_is_product_or_business_question" in body, body[:600]
        guard = body.index("_is_product_or_business_question")
        call = body.index("self.solution_agent.run(")
        assert guard < call, "跑题判定必须在送产品 RAG 之前"
        # 跑题轮不带产品
        assert '"products": []' in body, body[:800]

    def test_short_answers_are_not_treated_as_chit_chat(self):
        """"yes" / "ok" 是客户在接上一问，不能被当成跑题挡掉。"""
        import src.orchestrator as orch

        source = open(orch.__file__, encoding="utf-8").read()
        # 短回答（<=3 词且不是问句）不判跑题，改由 "quote_confirmation" 分支承接
        assert "_words > 3" in source, "短回答护栏不见了"


class TestSteerFallbackNeverReportsAModel:

    def test_steer_answer_has_no_model_and_brings_the_talk_back(self):
        from src.rag.reply_composer import off_topic_steer_answer

        for lang in ("en", "zh"):
            text = off_topic_steer_answer(lang, 0)
            assert text, lang
            # 不许出现型号样式（TW11-3216-P4.0 / P4.0 之类）
            assert not re.search(r"\bTW\d+", text), text
            assert not re.search(r"\bP\d(?:\.\d+)?\b", text), text
        assert "model" in off_topic_steer_answer("en", 0).lower()
        assert "型号" in off_topic_steer_answer("zh", 0)

    def test_api_empty_branch_refuses_to_report_a_model_on_chit_chat(self):
        source = open(os.path.join(project_root, "src", "api.py"), encoding="utf-8").read()
        assert 'result.get("offtopic_turn")' in source, "api 缺少跑题轮护栏"
        # 更强的契约（客户口径 2026-10）：api **根本不再调用**"报最接近型号"的兜底——
        # 那句话已彻底删除，正文被清空时只邀请客户补充条件 / 给接话过渡。
        assert "product_fallback_answer(" not in source, "报型号兜底必须从 api 里删掉"
        assert "off_topic_steer_answer" in source, "跑题轮要有'接住+拉回'的兜底"
        assert "relaxation_answer" in source, "清空时要有邀请补充条件的兜底"
        guard = source.index('result.get("offtopic_turn")')
        relax = source.index("relaxation_answer(")
        assert guard < relax, "跑题护栏必须排在通用兜底之前"

    def test_the_canned_model_sentence_is_gone_from_every_call_site(self):
        """两个调用点都不许再报"最接近的型号"。"""
        for rel in ("src/api.py", "src/agents/solution/runner.py"):
            source = open(os.path.join(project_root, rel), encoding="utf-8").read()
            assert "product_fallback_answer(" not in source, rel
        # 函数本身已废弃，永远返回空串
        from src.rag.reply_composer import product_fallback_answer

        assert product_fallback_answer([{"metadata": {"model": "TW31-COB-P0.7H"}}]) == ""
