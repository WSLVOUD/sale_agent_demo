"""计划 2.0（Phase 12-3）：Response 层职责收口。

计划规定的最终形态：

    ResponseContext → ResponseGenerator → ResponseValidator

分工：

    ResponseContext    只描述"当前业务状态 / 动作 / 档案 / 问题 / 可用事实"
                       —— 不拼客户可见句子（计划 §3.1：Python 决定说什么）
    ResponseGenerator  只把已经定好的动作表达成自然语言
                       —— 不得重新决定产品类型 / 下一问题 / 推荐 / Gate / 计算
    ResponseValidator  只做最终检查（重复问题 / 多问题 / 未确认型号 / 幻觉 / 与档案冲突）

并规定：客户可见文本只有一个出口（orchestrator → FinalResponseCoordinator.build）。

引用分析结论（本轮实测）：

    response_generator → response_validator.validate_response
    response_coordinator → final_guard（售后口径 + 图片核对 + 守门）
    final_response → final_guard
    final_guard → response_validator.INTERNAL_TERMS
    natural_response / natural_continuation → 只被 dialogue 包出口与 orchestrator 使用

即：这条链**不是层层转发**，每一层都有独立职责（组装 / 守门 / 校验 / 表达辅助），
所以按计划"先引用分析、再决定合并"的要求，本轮只把边界钉死；
"表达辅助并入 ResponseGenerator"是一次纯搬迁，留待单独一轮执行（见计划文件的执行进度）。
"""
import os
import re
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _read(rel_path: str) -> str:
    with open(os.path.join(project_root, rel_path), encoding="utf-8") as handle:
        return handle.read()


_DOCSTRING_RE = re.compile(r'"""(?:.|\n)*?"""|\'\'\'(?:.|\n)*?\'\'\'')


def _code_only(text: str) -> str:
    without_docs = _DOCSTRING_RE.sub("", str(text or ""))
    return "\n".join(line.split("#", 1)[0] for line in without_docs.splitlines())


class TestResponseContextIsStructuredOnly:

    def test_context_does_not_build_customer_sentences(self):
        """ResponseContext 只装结构化事实，不装拼好的销售句子。"""
        code = _code_only(_read("src/dialogue/response_context.py"))
        for token in ("compose_requirement_reply", "relaxation_answer"):
            assert token not in code, f"ResponseContext 不该拼句子（{token}）"


class TestResponseGeneratorDoesNotRedecide:

    def test_generator_does_not_decide_business(self):
        code = _code_only(_read("src/dialogue/response_generator.py"))
        for token in (
            "route_display_type",          # 产品类型
            "check_recommendation_ready",  # Gate
            "recommendation_engine",       # 推荐
            "screen_calculator",           # 计算
            "pass1_pending",               # 选问题
            "shuffled_slots",
        ):
            assert token not in code, f"ResponseGenerator 不得重新决定业务（{token}）"

    def test_generator_does_not_mutate_the_profile(self):
        code = _code_only(_read("src/dialogue/response_generator.py"))
        assert re.findall(r"profile\.\w+\s*=[^=]", code) == []


class TestSingleFinalCheck:

    def test_validator_is_the_only_final_check_entry(self):
        code = _code_only(_read("src/dialogue/response_validator.py"))
        assert "def validate_response" in code
        # 生成器与 Guard 都只能"调用"校验，不能自己再写一套
        for rel in (
            "src/dialogue/response_generator.py",
            "src/dialogue/final_guard.py",
        ):
            other = _code_only(_read(rel))
            assert "def validate_response" not in other, f"{rel} 不该再定义一套校验"

    def test_guard_only_enforces_the_single_question_budget(self):
        """Final Guard 只做"最多一个问题 + 清内部术语"，不重排业务。"""
        code = _code_only(_read("src/dialogue/final_guard.py"))
        for token in ("route_display_type", "check_recommendation_ready", "field_action"):
            assert token not in code, f"Final Guard 不该参与业务决策（{token}）"
