"""回归：推荐完之后客户回来说 "yes" → 应该是"我去准备报价单"，且要说人话。

客户口径（2026-10 实测）：

    客户: 那条户外教堂屏推荐完了 …（中间聊了电视 / 天气 / 游戏）
    客户: yes
    AI  : Let's take a slightly different angle, if one of the requirements can be
          relaxed (for example the pixel pitch, the screen size, or the viewing
          distance), I can match a model for you right away.       ← 和 "yes" 完全不搭

原因：``already_recommended`` + 本轮无新需求 → 被丢进自由问答（others）→ 正文被清空 →
落到 ``relaxation_answer``（"能不能放宽某个条件"）。

期望：客户是在**同意推进报价**，应回"我这就去准备报价单，稍等"，并且
**结合上下文润色**（不是每次同一句死板话术）。
"""
import os
import re
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestQuoteConfirmationFallback:

    def test_it_says_the_quotation_is_being_prepared(self):
        from src.rag.reply_composer import quote_confirmation_answer

        for lang, word in (("en", "quotation"), ("zh", "报价")):
            text = quote_confirmation_answer(lang, 0)
            assert word in text, text

    def test_it_never_invents_price_or_delivery(self):
        from src.rag.reply_composer import quote_confirmation_answer

        for lang in ("en", "zh"):
            for seed in range(6):
                text = quote_confirmation_answer(lang, seed).lower()
                assert not re.search(r"\b\d+\s*(?:day|week|usd|dollar|%)\b", text), text
                assert "discount" not in text and "折扣" not in text
                assert not re.search(r"\bTW\d+", text), text

    def test_it_is_not_one_rigid_line(self):
        """客户明确要求"不要死板的用一个话术" → 必须多句轮换。"""
        from src.rag.reply_composer import quote_confirmation_answer

        variants = {quote_confirmation_answer("en", i) for i in range(4)}
        assert len(variants) == 4, variants


class TestQuotePromptIsContextAware:

    def test_the_prompt_reads_the_conversation_and_forbids_templates(self):
        from src.dialogue.chat_reply import _QUOTE_PROMPT

        text = _QUOTE_PROMPT.format(message="yes", recent="(none)")
        assert "quotation" in text
        # 必须结合上下文（知道客户在确认哪块屏），且不许死板
        assert "read the conversation above" in text
        assert "do not sound like a template" in text
        # 禁止编造
        assert "Never invent facts" in text
        for forbidden in ("price", "delivery date", "stock", "warranty"):
            assert forbidden in text, forbidden

    def test_generate_chat_reply_accepts_the_purpose(self):
        import inspect

        from src.dialogue.chat_reply import generate_chat_reply

        sig = inspect.signature(generate_chat_reply)
        assert "purpose" in sig.parameters
        assert sig.parameters["purpose"].default == "chitchat"


class TestOrchestratorConfirmsInsteadOfAskingToRelax:

    def _source(self):
        return open(
            os.path.join(project_root, "src", "orchestrator.py"), encoding="utf-8"
        ).read()

    def test_confirmation_branch_exists_and_precedes_the_rag_call(self):
        source = self._source()
        assert "quote_confirmation_answer" in source, "缺少'报价在准备'的兜底"
        assert '"quote_confirmation"' in source, "没有把确认轮交给按语境生成"
        guard = source.index("_affirm = ")
        rag = source.index("self.solution_agent.run(", guard)
        assert guard < rag, "确认轮判定必须排在送产品 RAG 之前"

    def test_confirmation_is_semantic_not_a_keyword_or_word_count(self):
        """客户口径：不能只靠关键词/词数判断"是否推进"，要理解整句话 + 上下文。"""
        source = self._source()
        # orchestrator 用的是销售层语义判定的结论
        assert 'sales_result.get("quote_confirmation")' in source, source[:0] or "未使用语义判定"
        # 不许再拿"词数"当"客户同意"的依据
        assert "_already_recommended and 0 < _words" not in source, "还在用词数判同意"

    def test_the_sales_layer_judges_it_with_the_conversation_context(self):
        from src.dialogue.approval import _APPROVAL_PROMPT, understand_approval

        prompt = _APPROVAL_PROMPT.format(message="yes", recent="(none)")
        assert "结合上下文" in prompt and "看不出来就回答" in prompt
        # 判定不了必须保守（返回 None），绝不猜
        assert understand_approval("") is None

    def test_relaxation_is_never_used_once_a_model_was_matched(self):
        """客户原话：「推荐完后，还是会出现这句话，帮我彻底的解决」。

        型号都已经选出来了，还说"放宽某个条件我就能匹配" —— 等于告诉客户没匹配上。
        这句兜底只允许出现在"确实一个型号都没匹配到"的情形。
        """
        api = open(os.path.join(project_root, "src", "api.py"), encoding="utf-8").read()
        products_branch = api.index('elif result.get("products"):')
        relax = api.index("relaxation_answer(", products_branch)
        assert products_branch < relax, "有型号的分支必须在放宽条件之前"
        assert "quote_confirmation_answer" in api[products_branch:relax], "有型号时应回报价承接"

        runner = open(
            os.path.join(project_root, "src", "agents", "solution", "runner.py"),
            encoding="utf-8",
        ).read()
        assert "if response_products:" in runner, "runner 未按'有没有选到型号'分流"
        assert "quote_confirmation_answer()" in runner

    def test_every_relaxation_variant_is_recognised(self):
        """"放宽条件"文案有多个变体，必须全部能识别，否则拦不住。"""
        from src.rag.reply_composer import is_relaxation_answer, relaxation_answer

        n = 0
        for family in ("", "lcd", "ifp"):
            for lang in ("en", "zh"):
                for i in range(6):
                    text = relaxation_answer(lang, i, product_family=family)
                    assert is_relaxation_answer(text), (family, lang, i, text)
                    n += 1
        assert n >= 12
        # 不能误伤正常回复
        assert not is_relaxation_answer("The TW21-OD-P4 is the right fit.")
        assert not is_relaxation_answer(
            "Perfect — I'll get the quotation put together for your screen and send it over shortly."
        )

    def test_the_orchestrator_blocks_relaxation_after_a_recommendation(self):
        """收口处必须有闸门：已推荐过就绝不允许"放宽条件"漏给客户。

        实测它从多条内部路径漏出来（runner 空回答兜底 / trigger_solution 分支），
        所以按**会话事实**在唯一收口处统一拦。
        """
        source = self._source()
        assert "is_relaxation_answer" in source, "收口处缺少拦截"
        assert "has_recommendation" in source, "拦截必须以'是否已推荐过'为条件"
        guard = source.index("is_relaxation_answer")
        assert "quote_confirmation_answer" in source[guard : guard + 800]

    def test_the_orchestrator_re_judges_semantically_when_the_flag_is_missing(self):
        """只依赖上游一个标志不可靠：上游漏一次就会退回"放宽条件"（实测复现）。

        所以 orchestrator 自己也要在**生成回复的地方**语义判定一次。
        """
        source = self._source()
        assert "understand_approval" in source, "orchestrator 缺少兜底的语义判定"
        guard = source.index("understand_approval(_text")
        rag = source.index("self.solution_agent.run(", guard)
        assert guard < rag, "兜底语义判定必须排在送产品 RAG 之前"
        # 判定不了（None）不许当成同意
        assert "_approved is True" in source, "只有明确判为同意才算同意"


class TestApprovalJudgeIsContextDriven:

    def test_yes_after_a_quotation_offer_is_approval(self, monkeypatch):
        """有上下文时 "yes" 判为同意；同样一句没有上下文时不许判成同意。"""
        import src.core.llm as core_llm
        import src.dialogue.approval as approval

        class _Resp:
            def __init__(self, content):
                self.content = content

        class _LLM:
            def __init__(self, content):
                self._content = content

            def invoke(self, _prompt):
                return _Resp(self._content)

        def _patch(content):
            monkeypatch.setattr(core_llm, "get_llm", lambda **_: _LLM(content), raising=False)

        _patch("是")
        assert approval.understand_approval(
            "yes", recent="AI: Shall I prepare the quotation for you?"
        ) is True

        _patch("否")
        assert approval.understand_approval("which one is cheaper?") is False

        _patch("maybe?")
        assert approval.understand_approval("hmm") is None, "判不出来必须是 None，不许猜"
