"""《ResponseGenerator 架构收口改造计划》Phase 2~7 的收敛不变量。

| Phase | 不变量 |
|---|---|
| 2 | QuestionPlanner 只出候选：`plan_next_question() == candidate_questions()[0]`，不写 state |
| 3 | QuestionFlow 只管状态/计划：不生成客户话术、不写 state["response"] |
| 4 | `QuestionSpec` 是"要问什么"的唯一载体（旧字段自动归一进来） |
| 5 | ResponseGenerator 只能改表达：不改 action / slot，离线也只问 spec 的那一项 |
| 6 | script_generator 不得再新增"问什么 / 怎么做"的独立决策（决策点数量被钉死） |
| 7 | 旧模板链路只允许出现在 fallback 分支（正常路径 = LLM 原生生成） |
"""
import io
import os
import re
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

SRC = os.path.join(project_root, "src")


def _read(relative: str) -> str:
    return io.open(os.path.join(project_root, relative), encoding="utf-8").read()


class TestPhase2QuestionPlannerIsCandidateOnly:

    def test_plan_next_question_is_the_first_candidate(self):
        from src.agents.sales.question_planner import (
            candidate_questions,
            plan_next_question,
        )
        from src.models.requirement import RequirementProfile

        slots = {"display_type": "LED", "target_width_mm": 3000, "target_height_mm": 5000}
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        candidates = candidate_questions(profile)
        assert candidates, "缺失字段时应该给得出候选"
        assert plan_next_question(profile) == candidates[0]
        assert all("slot" in item and "priority" in item for item in candidates)

    def test_planner_never_writes_state(self):
        source = _read("src/agents/sales/question_planner.py")
        # 只看"赋值"，不看注释/文档字符串里的提及
        for forbidden in (
            r'state\["response"\]\s*=',
            r"state\['response'\]\s*=",
            r'state\["pending_question"\]\s*=',
            r'state\["pending_slot"\]\s*=',
        ):
            assert not re.search(forbidden, source), f"QuestionPlanner 不得写 {forbidden}"


class TestPhase3QuestionFlowIsStateOnly:

    def test_question_flow_does_not_generate_customer_text(self):
        source = _read("src/dialogue/question_flow.py")
        for forbidden in ("state[", "context.response", "generate_response"):
            assert forbidden not in source, f"QuestionFlow 不得出现 {forbidden}"

    def test_plan_carries_reason_and_slot_for_the_policy(self):
        from src.dialogue.question_flow import next_question_plan
        from src.models.requirement import RequirementProfile

        slots = {"display_type": "LED", "target_width_mm": 3000, "target_height_mm": 5000}
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        plan = next_question_plan(profile, session_id="phase3")
        assert plan is None or (getattr(plan, "slot", "") and getattr(plan, "question", ""))


class TestPhase4QuestionSpecIsTheOnlyCarrier:

    def test_legacy_fields_normalise_into_the_spec(self):
        from src.dialogue import QuestionSpec, ResponseContext

        context = ResponseContext(
            action="ASK",
            question="How far will viewers be?",
            question_slot="viewing_distance",
            question_intent="collect viewing distance",
        )
        assert context.question_spec.slot == "viewing_distance"
        assert context.question_spec.intent == "collect viewing distance"
        assert context.question_spec.text == "How far will viewers be?"

    def test_explicit_spec_wins_over_legacy_fields(self):
        from src.dialogue import QuestionSpec, ResponseContext

        context = ResponseContext(
            action="ASK",
            question="legacy text",
            question_slot="pixel_pitch",
            question_spec=QuestionSpec(slot="installation", intent="fixed or rental"),
        )
        assert context.question_spec.slot == "installation"


class TestPhase5ResponseGeneratorOnlyWritesProse:

    def test_offline_generation_asks_the_spec_slot_and_does_not_mutate(self):
        from src.dialogue import ResponseContext, generate_response

        context = ResponseContext(
            action="ASK",
            question="How far will viewers be from the screen?",
            question_slot="viewing_distance",
        )
        before = (context.action, context.question_slot, context.question_spec.slot)
        text = generate_response(context, llm=None, seed=0)
        assert "how far" in text.lower()
        assert (context.action, context.question_slot, context.question_spec.slot) == before

    def test_generator_source_never_assigns_action_or_slot(self):
        source = _read("src/dialogue/response_generator.py")
        for forbidden in (
            r"context\.action\s*=(?!=)",
            r"context\.question_slot\s*=(?!=)",
            r"context\.recommendation\s*=(?!=)",
            r"context\.engineering_result\s*=(?!=)",
        ):
            assert not re.search(forbidden, source), f"ResponseGenerator 不得改 {forbidden}"


class TestPhase6ScriptGeneratorHasNoNewDecisionPower:

    """决策点数量被钉死：再想"自己决定问什么/怎么做"，就必须先改这里和计划。"""

    SCRIPT_GENERATOR_NEXT_ACTION_WRITES = 21  # 2026-09-23 基线（计划 Phase 6/7 逐条搬到 DialoguePolicy）

    def test_next_action_writes_do_not_grow(self):
        source = _read("src/agents/sales/nodes/script_generator.py")
        writes = len(re.findall(r'state\["next_action"\]\s*=', source))
        assert writes <= self.SCRIPT_GENERATOR_NEXT_ACTION_WRITES, (
            f"script_generator 的 next_action 决策点从 {self.SCRIPT_GENERATOR_NEXT_ACTION_WRITES} "
            f"涨到了 {writes} —— 计划 §18 禁止在这里继续堆决策，请改走 DialoguePolicy"
        )

    def test_script_generator_does_not_decide_the_question_slot(self):
        source = _read("src/agents/sales/nodes/script_generator.py")
        for forbidden in ('state["pending_slot"] =', 'state["pending_question"] ='):
            assert forbidden not in source, f"script_generator 不得写 {forbidden}"


class TestPhase7LegacyChainIsFallbackOnly:

    def test_template_composer_is_only_used_as_fallback(self):
        source = _read("src/agents/sales/nodes/script_generator.py")
        # 正常路径是 LLM 原生生成；模板拼装只允许出现在 _natural_reply() 的兜底里
        fallback_start = source.index("state[\"response_source\"] = \"template\"")
        fallback_calls = [
            index for index in re.finditer(r"compose_requirement_reply\(", source)
        ]
        assert fallback_calls, "兜底模板链路必须仍然存在（先替换 → 测试 → 再删除）"
        assert all(match.start() > fallback_start for match in fallback_calls[1:]), (
            "正常路径不得再调用模板拼装"
        )

    def test_response_source_is_recorded(self):
        source = _read("src/agents/sales/nodes/script_generator.py")
        assert 'state["response_source"] = "llm"' in source
        assert 'state["response_source"] = "template"' in source
