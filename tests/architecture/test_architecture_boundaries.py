"""计划《项目架构瘦身与代码整理计划》Phase 4~8：层与层之间的边界。

核对结论（用包级 import 矩阵实测）：

    rag      → 不引用 dialogue / agents          ✅ RAG 只做知识，不决定对话或推荐
    engineering → 只引用 models                   ✅ 计算层独立（不被塞进 Agent / 推荐）
    vision   → core / models / rag / config       ✅ 只抽需求，不推荐、不决定问什么
    memory   → 只有 config                        ✅ 会话层轻量

已知的**边界例外**（本次不贸然搬动，先钉住数量，避免继续扩散）：

  1. `dialogue/product_router.py` 的 LEDPolicy 为了给出"下一个问题"，
     在**函数内**延迟导入 `agents.sales.question_planner`（1 处）。
     要拆干净必须先把 LED 链路的问题计划迁到共享层 —— 属于后续阶段。
  2. `models/requirement.py` 在**函数内**延迟导入 rag 的两个辅助函数 + 一个 Gate（3 处）。
     同样属于"要搬迁先做迁移"的动作。

所以本文件做两件事：**守住不许新增的边界**，以及**钉住已知例外的数量**。
"""
import os
import re
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

SRC = os.path.join(project_root, "src")


def _py_files(pkg: str):
    base = os.path.join(SRC, pkg)
    for root, _dirs, files in os.walk(base):
        if "__pycache__" in root:
            continue
        for name in files:
            if name.endswith(".py"):
                yield os.path.join(root, name)


def _imports(path: str):
    """返回 (模块级 import 列表, 函数级 import 列表)。"""
    module_level, nested = [], []
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not re.match(r"^\s*(?:from|import)\s", line):
                continue
            (nested if line.startswith((" ", "\t")) else module_level).append(line.strip())
    return module_level, nested


class TestRagIsKnowledgeOnly:

    def test_rag_never_imports_dialogue_or_agents(self):
        offenders = []
        for path in _py_files("rag"):
            level, nested = _imports(path)
            for line in level + nested:
                if re.search(r"\b(dialogue|agents)\b", line):
                    offenders.append((os.path.relpath(path, project_root), line))
        assert offenders == [], f"RAG 不能伸手进对话/Agent 层：{offenders}"

    def test_rag_does_not_decide_recommendation_by_itself(self):
        """推荐决策只允许从 readiness（Gate）与 recommendation_engine 出。"""
        for path in _py_files("rag"):
            name = os.path.basename(path)
            if name in ("readiness.py", "recommendation_engine.py", "recommendation_coordinator.py",
                        "recommendation_service.py", "validation.py", "hard_filter.py"):
                continue
            text = open(path, encoding="utf-8", errors="ignore").read()
            assert "check_recommendation_ready" not in text or name == "readiness.py", (
                f"{name} 不该自己判推荐就绪"
            )


class TestKnownBoundaryExceptionsArePinned:
    """已知例外：只允许这么多处，且必须是函数内延迟导入（不能变成模块级依赖）。"""

    def test_models_only_lazily_reach_into_rag(self):
        hits, module_level = [], []
        for path in _py_files("models"):
            level, nested = _imports(path)
            for line in level:
                if re.search(r"\brag\b", line):
                    module_level.append((os.path.basename(path), line))
            for line in nested:
                if re.search(r"\brag\b", line):
                    hits.append((os.path.basename(path), line))
        assert module_level == [], f"models 不允许模块级依赖 rag：{module_level}"
        assert len(hits) == 3, f"已知例外应恰好 3 处，实际 {len(hits)}：{hits}"
        joined = " ".join(line for _f, line in hits)
        assert "_detect_purpose" in joined and "check_recommendation_ready" in joined

    def test_dialogue_only_lazily_reaches_into_agents(self):
        hits, module_level = [], []
        for path in _py_files("dialogue"):
            level, nested = _imports(path)
            for line in level:
                if re.search(r"\bagents\b", line):
                    module_level.append((os.path.basename(path), line))
            for line in nested:
                if re.search(r"\bagents\b", line):
                    hits.append((os.path.basename(path), line))
        assert module_level == [], f"dialogue 不允许模块级依赖 agents：{module_level}"
        assert len(hits) == 1, f"已知例外应恰好 1 处（product_router 的 LEDPolicy）：{hits}"
        assert hits[0][0] == "product_router.py"


class TestVisionStaysAnExtractor:

    def test_vision_never_decides_recommendation_or_questions(self):
        forbidden = (
            "recommendation_engine",
            "check_recommendation_ready",
            "plan_next_question",
            "decide_turn_action",
        )
        for path in _py_files("vision"):
            text = open(path, encoding="utf-8", errors="ignore").read()
            for token in forbidden:
                assert token not in text, f"{os.path.basename(path)} 不该出现 {token}"


class TestCalculationStaysIndependent:

    def test_engineering_and_calculator_do_not_import_agents_or_dialogue(self):
        offenders = []
        targets = list(_py_files("engineering")) + [os.path.join(SRC, "tools", "screen_calculator.py")]
        for path in targets:
            level, nested = _imports(path)
            for line in level + nested:
                if re.search(r"\b(agents|dialogue)\b", line):
                    offenders.append((os.path.relpath(path, project_root), line))
        assert offenders == [], f"计算层必须独立：{offenders}"


class TestOneVisibleResponse:

    def test_customer_visible_text_is_built_in_one_place(self):
        """客户可见文本只有一个出口：orchestrator → FinalResponseCoordinator.build。"""
        orchestrator = open(
            os.path.join(SRC, "orchestrator.py"), encoding="utf-8", errors="ignore"
        ).read()
        assert "_final_response_coordinator().build(" in orchestrator
        assert "_response_coordinator().finalize(" in orchestrator

        # 其它模块不允许自己拼"最终回复"对象
        builders = []
        for root, _dirs, files in os.walk(SRC):
            if "__pycache__" in root:
                continue
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(root, name)
                text = open(path, encoding="utf-8", errors="ignore").read()
                if "build_final_response(" in text or "FinalResponseCoordinator(" in text:
                    builders.append(os.path.relpath(path, project_root).replace("\\", "/"))
        assert sorted(builders) == [
            "src/dialogue/final_response.py",
            "src/orchestrator.py",
        ], builders
