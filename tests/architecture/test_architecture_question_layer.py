"""计划 2.0（Phase 12-2）：Question 层职责收口。

计划规定的三层分工：

    Question Policy   → 只决定"下一步最该问什么"
    QuestionSpec      → 只决定"这个问题怎么表达"（问法库在 rag.readiness）
    ResponseGenerator → 只决定"整段回复怎么自然地说"

并明确禁止：

    QuestionPlanner 不得生成完整回复 / 决定推荐 / 修改 RequirementProfile
    ResponseGenerator 不得重新决定问什么、重新选 slot、重新决定推荐

另外把"槽位全集"收敛成一处：`rag.field_policy.RECOMMENDATION_SLOTS`。
`question_flow.ASK_POOL` 只是它内部"参与提问"的子集，不允许自成一套名单。
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
    """只留真正会被执行的代码：去掉三引号文档串与 # 注释。"""
    without_docs = _DOCSTRING_RE.sub("", str(text or ""))
    return "\n".join(line.split("#", 1)[0] for line in without_docs.splitlines())


QUESTION_LAYER_MODULES = (
    "src/dialogue/question_planner.py",
    "src/dialogue/question_flow.py",
    "src/dialogue/question_order.py",
    "src/dialogue/question_registry.py",
    "src/agents/sales/question_planner.py",
)


class TestSlotUniverseIsSingleSourced:

    def test_ask_pool_is_a_subset_of_the_slot_universe(self):
        from src.dialogue.question_flow import ASK_POOL
        from src.rag.field_policy import RECOMMENDATION_SLOTS

        assert set(ASK_POOL) <= set(RECOMMENDATION_SLOTS), (
            f"提问池里出现了槽位全集之外的槽位：{set(ASK_POOL) - set(RECOMMENDATION_SLOTS)}"
        )
        assert set(RECOMMENDATION_SLOTS) - set(ASK_POOL) == {
            "size_axis",       # 只有"客户给了裸尺寸"时才单独确认方向
            "display_type",    # 走产品类型闸门（ProductTypeRouter），不在需求提问池
            "content_type",    # 客户口径：不主动问，只记录
        }, "不参与提问的槽位必须只有这三个（有专门入口）"


class TestQuestionLayerBoundaries:

    def test_question_layer_never_writes_the_response(self):
        for rel in QUESTION_LAYER_MODULES:
            code = _code_only(_read(rel))
            assert 'state["response"]' not in code, f"{rel} 不许生成客户可见回复"

    def test_question_layer_never_decides_recommendation(self):
        forbidden = ("recommendation_engine", "recommendation_coordinator", "RecommendationService")
        for rel in QUESTION_LAYER_MODULES:
            code = _code_only(_read(rel))
            for token in forbidden:
                assert token not in code, f"{rel} 不许自己决定推荐（{token}）"

    def test_question_layer_does_not_mutate_the_profile(self):
        """判断"该不该问"必须只读档案，不允许回头改需求。"""
        for rel in QUESTION_LAYER_MODULES:
            code = _code_only(_read(rel))
            writes = re.findall(r"profile\.\w+\s*=[^=]", code)
            assert writes == [], f"{rel} 修改了 RequirementProfile：{writes}"

    def test_response_generator_does_not_pick_the_slot(self):
        """ResponseGenerator 只表达，不再重选 slot。"""
        code = _code_only(_read("src/dialogue/response_generator.py"))
        for token in ("pass1_pending", "shuffled_slots", "field_action"):
            assert token not in code, f"ResponseGenerator 不该自己选问题（{token}）"


class TestLegacyQuestionHelpersAreGone:

    def test_next_in_order_is_not_re_exported_without_a_caller(self):
        """没有调用方的历史辅助函数不再挂在包出口上（避免"看起来还在用"）。"""
        init_text = _code_only(_read("src/dialogue/__init__.py"))
        assert "next_in_order" not in init_text, (
            "next_in_order 已无调用方 → 不该继续挂在 dialogue 包出口"
        )
        assert "def next_in_order" not in _code_only(_read("src/dialogue/question_order.py"))
