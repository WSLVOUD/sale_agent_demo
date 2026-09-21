"""v2.5++（僵硬话术优化 · 第十一/十二阶段）：Naturalness Golden Dataset + A/B。

数据集：`eval/naturalness_golden.json`（行为约束，不是固定文案）。

每个用例验证（计划 §14）：
    ✓ Action 正确        ✓ 问的问题正确（不重复问客户已经回答过的）
    ✓ 不机械重复客户      ✓ 不强制 ACK        ✓ 不强制 Connector
    ✓ 不超过一个问题      ✓ 无内部术语        ✓ 无虚假事实

A/B（计划 §15）：同一批输入跑 Strategy A（模板 → 润色）与 B（结构化 → 原生生成），
比较 ACK / Echo / Connector / 问卷腔 / 长度等指标，B 必须明显更自然。
"""
import json
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import (  # noqa: E402
    ANSWER_AND_ASK,
    ASK,
    STRATEGY_NATIVE,
    STRATEGY_TEMPLATE_POLISH,
    build_context,
    compare_strategies,
    compute_metrics,
    generate_response,
    next_question_plan,
    reset_llm_breaker,
    validate_response,
)
from src.models.requirement import RequirementProfile  # noqa: E402

GOLDEN_PATH = os.path.join(project_root, "eval", "naturalness_golden.json")


def _load_cases():
    with open(GOLDEN_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _profile(case) -> RequirementProfile:
    slots = dict(case["slots"])
    profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
    for slot in case.get("unknown_slots") or []:
        profile.mark_unknown(slot)
    if case.get("resolution"):
        from src.engineering import parse_resolution

        profile.resolution_requirement = parse_resolution(case["resolution"]).to_dict()
    return profile


def _reply_for(case):
    """按用例状态生成一轮回复（无 LLM → 结构化拼装，确定可断言）。"""
    profile = _profile(case)
    plan = next_question_plan(profile, session_id=f"golden-{case['id']}")
    question = plan.question if plan is not None else ""
    slot = plan.slot if plan is not None else ""
    answer = str(case.get("answer") or "")
    action = ANSWER_AND_ASK if (answer and question) else (ASK if question else ANSWER_AND_ASK)
    context = build_context(
        action=action,
        customer_message=case["customer_message"],
        question=question,
        answer=answer,
        missing_fields=[slot] if slot else [],
        required_question=slot,
    )
    text = generate_response(context)
    return profile, plan, text, context


class TestGoldenBehaviourConstraints:

    @pytest.mark.parametrize("case", _load_cases(), ids=lambda case: case["id"])
    def test_behaviour_constraints(self, case):
        profile, plan, text, context = _reply_for(case)
        answered = {str(slot) for slot in case.get("answered_slots") or []}

        # ① 问题正确：不再问客户已经回答过的那一项
        if plan is not None and plan.slot:
            assert plan.slot not in answered, f"重复追问了 {plan.slot}"

        # ② 行为约束（不是固定文案）
        check = validate_response(
            text,
            allow_ack=context.allow_ack,
            allow_connector=context.allow_connector,
            customer_question=bool(case.get("customer_asked")),
            answer=str(case.get("answer") or ""),
            customer_message=case["customer_message"],
            supported_parameters=["TW11-3216-P3.0", "P3.0"],
            required_question=context.question,
        )
        assert check.question_count <= 1, text
        assert not check.internal_terms, check.internal_terms
        assert not check.unsupported_parameters and not check.unsupported_models, text
        assert not check.changed_engineering_results, text
        assert not check.repeated_question_slot, text
        assert not check.questionnaire_pattern, text
        assert text, case["id"]

        # ③ 客户提问必须被回答
        if case.get("customer_asked"):
            assert check.customer_question_answered, text

    def test_golden_metrics_are_clean(self):
        """整份数据集的自然度指标：没有固定 ACK / 问卷腔 / 内部术语 / 虚假事实。"""
        samples = []
        for case in _load_cases():
            _, _, text, context = _reply_for(case)
            validation = validate_response(
                text,
                allow_ack=context.allow_ack,
                allow_connector=context.allow_connector,
                customer_question=bool(case.get("customer_asked")),
                answer=str(case.get("answer") or ""),
                customer_message=case["customer_message"],
                supported_parameters=["TW11-3216-P3.0", "P3.0"],
                required_question=context.question,
            )
            samples.append({
                "is_question": bool(case.get("customer_asked")),
                "validation": validation,
            })
        metrics = compute_metrics(samples)
        assert metrics["generic_ack_rate"] == 0.0
        assert metrics["questionnaire_pattern_rate"] == 0.0
        assert metrics["one_question_compliance"] == 1.0
        assert metrics["internal_term_leak_rate"] == 0.0
        assert metrics["unsupported_fact_rate"] == 0.0
        assert metrics["connector_repeat_rate"] == 0.0
        assert metrics["question_repeat_rate"] == 0.0


class _ScriptedLLM:
    """按策略给出不同风格的假 LLM：A 模板腔、B 自然短句。"""

    def __init__(self, strategy: str):
        self.strategy = strategy

    def invoke(self, prompt):
        text = str(prompt)

        class _Response:
            pass

        response = _Response()
        if "Rewrite the draft below" in text:
            draft = text.rsplit("Draft:", 1)[-1].split("Reply:", 1)[0].strip()
            response.content = f"Got it. Based on that, {draft}"
        else:
            # 原生生成：只问那一个问题，不客套、不复述
            question = ""
            for line in text.splitlines():
                if line.startswith("Question to ask"):
                    question = line.split(":", 1)[-1].strip()
            response.content = question or "What size screen do you need?"
        return response


class TestABStrategies:

    def _cases(self):
        cases = []
        for case in _load_cases():
            profile, plan, _, context = _reply_for(case)
            if plan is None or not plan.question:
                continue
            cases.append({"context": context, "is_question": True,
                          "supported_parameters": ["TW11-3216-P3.0"]})
        return cases

    def test_strategy_b_is_less_stiff_than_a(self):
        reset_llm_breaker()   # 别让前面用例的 LLM 失败熔断影响策略对比
        cases = self._cases()
        assert cases, "黄金数据集里应该有需要提问的用例"
        metrics = compare_strategies(
            cases,
            llm_factory=lambda strategy: _ScriptedLLM(strategy),
        )
        a = metrics[STRATEGY_TEMPLATE_POLISH]
        b = metrics[STRATEGY_NATIVE]
        # A 明显僵硬：泛 ACK / 过渡词 / 问卷腔 / 更长
        assert a["generic_ack_rate"] > 0
        assert a["questionnaire_pattern_rate"] > 0
        # B 干净：不客套、不衔接套话、一轮只问一个
        assert b["generic_ack_rate"] == 0.0
        assert b["questionnaire_pattern_rate"] == 0.0
        assert b["one_question_compliance"] == 1.0
        assert b["response_length"] < a["response_length"]

    def test_both_strategies_still_answer_the_customer(self):
        reset_llm_breaker()
        cases = self._cases()
        metrics = compare_strategies(
            cases, llm_factory=lambda strategy: _ScriptedLLM(strategy)
        )
        assert set(metrics) == {STRATEGY_TEMPLATE_POLISH, STRATEGY_NATIVE}
        assert metrics[STRATEGY_NATIVE]["internal_term_leak_rate"] == 0.0
