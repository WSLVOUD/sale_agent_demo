"""优化 v2.8 计划（对话自然话术生成）的可验证条目。

| 计划条目 | 这里锁住的行为 |
|---|---|
| §四 Phase 1 | `QuestionSpec.anchor` 是语义锚点（`text` 仅兼容别名） |
| §十二 Phase 5 | `ResponseShape` 告诉 LLM 这轮允许表达什么（不是 token 限制） |
| §十八 Phase 10 | Validator：多问题 / 问错槽位 / 多个回复块 / 语义不完整 / 机械复述事实 |
| §十一/§十九 | Prompt 用"语义任务范围"控制长度，禁止拼接独立话术块 |
"""
import io
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestQuestionSpecAnchor:

    def test_anchor_is_the_semantic_anchor(self):
        from src.dialogue import QuestionSpec

        spec = QuestionSpec(
            slot="viewing_distance",
            intent="collect typical viewer distance",
            anchor="ask roughly how far viewers will be from the screen",
        )
        assert spec.expression_anchor.startswith("ask roughly")
        # 兼容别名：两者始终一致
        assert spec.text == spec.anchor

    def test_legacy_text_still_works(self):
        from src.dialogue import QuestionSpec

        spec = QuestionSpec(slot="size", text="What screen size?")
        assert spec.anchor == "What screen size?"
        assert spec.expression_anchor == "What screen size?"


class TestResponseShape:

    def test_shape_follows_the_action(self):
        from src.dialogue import ResponseShape

        ask = ResponseShape.for_action("ASK", has_question=True)
        assert (ask.allow_answer, ask.allow_question, ask.max_questions) == (False, True, 1)
        answer_ask = ResponseShape.for_action("ANSWER_AND_ASK", has_question=True)
        assert (answer_ask.allow_answer, answer_ask.allow_question) == (True, True)

    def test_context_carries_the_shape_into_the_prompt(self):
        from src.dialogue import ResponseContext

        context = ResponseContext(
            action="ASK", question_slot="viewing_distance", question="How far?"
        )
        block = context.prompt_block()
        assert "Response shape" in block
        assert "max_questions=1" in block
        # 锚点必须标明"不是最终话术"
        assert "NOT the final wording" in block


class TestValidatorV28Checks:

    def _validate(self, text, **kwargs):
        from src.dialogue.response_validator import validate_response

        return validate_response(text, **kwargs)

    def test_multiple_response_blocks_are_flagged(self):
        text = (
            "First complete sales block that is long enough to stand alone.\n\n"
            "Second complete sales block that is also long enough.\n\n"
            "And a third one?"
        )
        result = self._validate(text)
        assert "multiple_response_blocks" in result.issues
        assert result.response_blocks >= 3

    def test_truncated_tail_is_flagged(self):
        result = self._validate("Sure, I can put a proposal together for you and")
        assert result.incomplete_tail is True
        assert "incomplete_response" in result.issues

    def test_mechanical_fact_repetition_is_flagged(self):
        result = self._validate(
            "You have an indoor screen, at 5 meters, and the 3x5 size, so let me help.",
            customer_message="indoor 5 meters 3x5 size, church",
        )
        assert len(result.repeated_known_facts) >= 3, result.repeated_known_facts
        assert "repeated_known_facts" in result.issues

    def test_single_fact_is_not_mechanical_repetition(self):
        result = self._validate(
            "An indoor screen works well here. What size do you need?",
            customer_message="indoor",
        )
        assert "repeated_known_facts" not in result.issues

    def test_passing_confirmation_is_allowed(self):
        result = self._validate(
            "Your 3x5 indoor setup is a good size. Roughly how far will viewers be?",
            customer_message="indoor 3x5",
        )
        assert "repeated_known_facts" not in result.issues

    def test_wrong_question_slot_is_flagged(self):
        result = self._validate(
            "Thanks. What pixel pitch would you like?",
            question_slot="viewing_distance",
        )
        assert "question_intent_mismatch" in result.issues or result.wrong_question_slot


class TestPromptUsesSemanticScope:

    def test_native_prompt_states_the_shape_rules(self):
        source = io.open(
            os.path.join(project_root, "src", "dialogue", "response_generator.py"),
            encoding="utf-8",
        ).read()
        for phrase in (
            "complete the current business action",
            "never introduce another question",
            "produce ONE coherent customer-facing response",
            "Do not concatenate independent response blocks",
        ):
            assert phrase in source, phrase
