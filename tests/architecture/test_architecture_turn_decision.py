"""计划《项目架构瘦身与代码整理计划》Phase 3：Turn / Question 决策只有一个源。

核对结论（代码 + 文档双重确认）：`action` / `turn_action` / `question_order` /
`question_registry` / `question_planner` / `question_flow` **不是重复实现**，
而是一条流水线：

    顺序（question_order） → 状态（question_registry） → 问哪个（question_flow）
        → 怎么问（question_planner） → 表达动作（action） → 本轮决策（turn_action）

所以这一阶段不做合并（计划原则：有独立有效逻辑就保留），只把"谁负责什么"钉死，
并消除**同一个业务规则被写多份**的地方（一轮最多一个问题）。
"""
import os
import re
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

SRC = os.path.join(project_root, "src")


def _read(rel_path: str) -> str:
    with open(os.path.join(project_root, rel_path), encoding="utf-8") as handle:
        return handle.read()


def _scan_src(pattern: str):
    hits = []
    for root, _dirs, files in os.walk(SRC):
        if "__pycache__" in root:
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8", errors="ignore") as handle:
                if re.search(pattern, handle.read(), re.MULTILINE):
                    hits.append(os.path.relpath(path, project_root).replace("\\", "/"))
    return sorted(hits)


class TestOneDecisionEntry:

    def test_dialogue_decision_is_defined_once(self):
        assert _scan_src(r"^def decide_turn_action\b") == ["src/dialogue/turn_action.py"]

    def test_expression_action_has_one_bridge(self):
        assert _scan_src(r"^def expression_action\b") == ["src/dialogue/action_bridge.py"]
        assert "from ....dialogue.action_bridge import expression_action" in _read(
            "src/agents/sales/nodes/script_generator.py"
        )


class TestQuestionPipelineHasOneOwnerPerJob:

    def test_each_stage_is_defined_in_exactly_one_module(self):
        expectations = {
            r"^def shuffled_slots\b": "src/dialogue/question_order.py",
            r"^class AskedQuestionRegistry\b": "src/dialogue/question_registry.py",
            r"^def plan_question\b": "src/dialogue/question_planner.py",
            r"^def pass1_pending\b": "src/dialogue/question_flow.py",
        }
        for pattern, expected in expectations.items():
            assert _scan_src(pattern) == [expected], pattern

    def test_question_registry_is_session_state_not_memory(self):
        """问题状态机不是 Memory（计划 §21）：不允许碰 memory/store。"""
        registry = _read("src/dialogue/question_registry.py")
        assert "memory" not in registry.replace("这不是 Memory", ""), "问题状态机不该引用记忆层"

    def test_question_planner_does_not_pick_the_slot(self):
        """选槽位是 question_flow 的事；planner 只负责怎么问。"""
        planner = _read("src/dialogue/question_planner.py")
        assert "pass1_pending" not in planner
        assert "shuffled_slots" not in planner, "顺序归 question_order，不该在 planner 里再排一次"


class TestOneQuestionRuleHasOneSource:

    def test_the_budget_is_defined_once(self):
        defining = _scan_src(r"^MAX_QUESTIONS_PER_TURN = \d+")
        assert defining == ["src/dialogue/action.py"], defining

    def test_all_layers_use_the_same_budget(self):
        from src.dialogue.action import MAX_QUESTIONS_PER_TURN
        from src.dialogue.final_guard import FinalResponseGuard
        from src.dialogue.final_response import FinalResponseCoordinator
        from src.dialogue.response_context import ResponseShape

        assert FinalResponseGuard().max_questions == MAX_QUESTIONS_PER_TURN
        assert FinalResponseCoordinator().guard.max_questions == MAX_QUESTIONS_PER_TURN
        assert ResponseShape(allow_question=True).max_questions == MAX_QUESTIONS_PER_TURN

    def test_layers_take_the_budget_from_that_one_source(self):
        """不许各写各的 1：三层都必须引用同一个常量。"""
        for rel in (
            "src/dialogue/final_guard.py",
            "src/dialogue/final_response.py",
            "src/dialogue/response_context.py",
        ):
            text = _read(rel)
            assert "MAX_QUESTIONS_PER_TURN" in text, f"{rel} 应引用统一常量"
            assert not re.search(r"max_questions: int = 1\b", text), (
                f"{rel} 还在自己写一份默认值"
            )
