"""优化 v2.9 计划里"上一轮标 🚧"的条目。

| 计划条目 | 这里锁住的行为 |
|---|---|
| §五~八 Phase 2/3 | ResponseContext 默认业务限制 + legacy 成句字段可观测 + 标注为"facts only" |
| §十六 Phase 8 | `_natural_reply()` 的动作来自 DialoguePolicy（action_bridge），不再自己推断 |
| §二十一 Phase 13 | 回复链路计数：total / llm_native / repair / fallback / legacy / validator_failures |
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_missing := (project_root not in sys.path):
    sys.path.insert(0, project_root)


class TestResponseContextSemantics:

    def test_default_restrictions_are_applied(self):
        from src.dialogue import ResponseContext

        context = ResponseContext(action="ASK", question_slot="viewing_distance")
        for rule in (
            "ask_only_one_question",
            "do_not_invent_facts",
            "do_not_repeat_known_facts",
        ):
            assert rule in context.restrictions

    def test_legacy_prose_fields_are_observable(self):
        from src.dialogue import ResponseContext

        context = ResponseContext(
            action="ANSWER_AND_ASK", answer="An indoor 3x5 screen works well.", opening="Got it"
        )
        fields = context.legacy_prose_fields()
        assert set(fields) == {"answer", "opening"}

    def test_prompt_labels_legacy_fields_as_facts_only(self):
        from src.dialogue import ResponseContext

        block = ResponseContext(
            action="ANSWER_AND_ASK", answer="indoor, 3x5", opening="confirmed size"
        ).prompt_block()
        assert "facts only" in block.lower()
        assert "NOT final wording" in block


class TestActionBridge:

    def test_policy_action_wins(self):
        from src.dialogue.action_bridge import (
            ANSWER_AND_ASK,
            ASK,
            DIRECT_ANSWER,
            RECOMMEND,
            expression_action,
        )

        state = {"dialogue_action": {"action": "ask_only"}}
        assert expression_action(state, question="How far?") == ASK
        # Policy 说"要问"；若同时有要传达的答案内容（如服务口径），
        # 表达层合并为"答 + 问"，不丢内容（但问题槽位仍以 Policy 为准）
        assert expression_action(state, answer="x", question="How far?") == ANSWER_AND_ASK
        assert (
            expression_action({"dialogue_action": {"action": "answer_only"}}, answer="x")
            == DIRECT_ANSWER
        )
        assert (
            expression_action(
                {"dialogue_action": {"action": "answer_then_ask"}}, answer="x", question="q"
            )
            == ANSWER_AND_ASK
        )
        assert (
            expression_action({"dialogue_action": {"action": "trigger_solution"}})
            == RECOMMEND
        )

    def test_fallback_only_when_policy_is_silent(self):
        from src.dialogue.action_bridge import ANSWER_AND_ASK, expression_action

        assert expression_action({}, answer="x", question="q") == ANSWER_AND_ASK

    def test_natural_reply_records_the_action_from_policy(self):
        """script_generator._natural_reply 不再自己推断动作（只记录映射结果）。"""
        import io
        import re

        source = io.open(
            os.path.join(project_root, "src", "agents", "sales", "nodes", "script_generator.py"),
            encoding="utf-8",
        ).read()
        assert "expression_action(state" in source
        assert 'state["response_action"] = action' in source
        # 旧的"按 answer/question 自己推断"的写法不得再出现
        assert not re.search(r"if answer and question:\s*\n\s*action = ANSWER_AND_ASK", source)


class TestResponseMetrics:

    def test_counters_track_the_chain(self):
        from src.dialogue import ResponseContext, generate_response
        from src.dialogue.response_metrics import reset, snapshot

        reset()
        # 没有 LLM → 结构化兜底
        generate_response(ResponseContext(action="ASK", question="How far?"), llm=None)
        data = snapshot()
        assert data["total_turns"] == 1
        assert data["structured_fallback_turns"] == 1
        assert data["legacy_reply_ratio"] == 0.0
        assert "legacy_reply_turns" in data

    def test_legacy_counter_is_recordable(self):
        from src.dialogue.response_metrics import record, reset, snapshot

        reset()
        record("total_turns")
        record("legacy_reply_turns")
        data = snapshot()
        assert data["legacy_reply_turns"] == 1
        assert data["legacy_reply_ratio"] == 1.0
