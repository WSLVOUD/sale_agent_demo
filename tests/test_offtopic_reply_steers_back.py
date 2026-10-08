"""回归：跑题的客户消息，回复必须**委婉拉回产品**，不能"接一句就结束"。

客户口径（2026-10 实测）：

    客户: Suspended Chinese hackers ... (一条新闻)
    AI  : That's quite a story, thanks for sharing it with me.
    客户: do u have a shoe
    AI  : Haha, that's not really our thing here.
    客户: i need to buy shoes
    AI  : Sounds like you've got a few things on your plate today.
    客户: can u give me skirt
    AI  : Ha, got it, noted.
    客户: do u like watching tv?
    AI  : Based on your requirements, the closest match is TW21-OD-P2.5 …

两个毛病：**① 每句都是死胡同**（接住就没了，既没拉回产品、也没推进）；
**② 最后在完全无关的一句上突然报型号**，和上下文冲突。

要求：这类"我没设定过答案"的客户消息，AI 要和客户对话、**委婉把话题拉回自己的产品**，
不能答完就结束，也不能和上下文里已确认的情况自相矛盾。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestOfftopicAckSteersBackToTheProduct:

    def _ack_prompt(self):
        from src.agents.sales.nodes.requirement import _ACK_PROMPT

        return _ACK_PROMPT.format(language_rule="English only")

    def test_it_requires_coming_back_to_the_customer_requirement(self):
        text = self._ack_prompt()
        assert "回到客户的屏幕需求" in text, text
        assert "不要把话题停在闲聊上就结束" in text, text

    def test_it_forbids_being_a_bare_acknowledgement(self):
        text = self._ack_prompt()
        # 敷衍的单独客套被明确点名为不合格
        assert "敷衍" in text, text
        assert "got it" in text.lower(), text

    def test_it_still_forbids_fabrication_and_knowledge_answers(self):
        text = self._ack_prompt()
        assert "绝对不要" in text and "知识性、技术性问题" in text, text
        assert "不要给参数、型号、价格" in text, text

    def test_it_forbids_conflicting_with_the_context(self):
        text = self._ack_prompt()
        assert "不许和上下文冲突" in text, text
        assert "室内/户外" in text or "室内" in text, text
        assert "凭空冒出型号或报价" in text, text

    def test_it_asks_for_at_most_one_question(self):
        text = self._ack_prompt()
        assert "不要问两个问题" in text, text


class TestChatReplyPromptSteersBack:

    def _chat_prompt(self):
        from src.dialogue.chat_reply import _PROMPT

        return _PROMPT.format(message="do u like watching tv?", recent="(none)")

    def test_it_demands_a_bridge_back_to_the_product(self):
        text = self._chat_prompt()
        assert "Always bring it back to our product" in text, text
        # 单独一句"noted / got it"被点名不合格
        assert "is NOT acceptable" in text, text

    def test_it_demands_context_consistency(self):
        text = self._chat_prompt()
        assert "Stay consistent with the conversation above" in text, text
        assert "never invent a model or a price" in text, text

    def test_it_still_forbids_inventing_facts(self):
        text = self._chat_prompt()
        assert "Never invent facts" in text, text
