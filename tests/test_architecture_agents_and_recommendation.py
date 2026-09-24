"""计划《项目架构瘦身与代码整理计划》Phase 5 / Phase 6 契约。

Phase 5（Sales / Solution 去重复）—— 核对结论：

    · Sales 与 Solution **共用** `core.requirement_extractor` → `RequirementProfile`；
    · Solution 侧只把档案投影成旧字段（`profile_to_solution_requirement`）给旧代码读；
    · 旧的自然语言抽取（`_build_requirement_prompt`）保留为**没有档案时的兜底**，
      不再是主路径。

Phase 6（Recommendation 单一入口）—— 核对结论：

    生产链路：solution/nodes/recommend → RecommendationService（Gate 必过）
              → recommendation_coordinator（编排 + 可行性 + 审计）→ RecommendationEngine（评分排序）

    即：引擎只被 coordinator 引用、coordinator 只被 service 引用，业务侧只认 service。
"""
import os
import re
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

SRC = os.path.join(project_root, "src")


def _read(rel_path: str) -> str:
    with open(os.path.join(project_root, rel_path), encoding="utf-8") as handle:
        return handle.read()


def _importers_of(symbol: str, *, exclude: tuple = ()):
    hits = []
    for root, _dirs, files in os.walk(SRC):
        if "__pycache__" in root:
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(root, name), project_root).replace("\\", "/")
            if rel in exclude or rel.endswith(f"{symbol}.py"):
                continue
            text = open(os.path.join(root, name), encoding="utf-8", errors="ignore").read()
            if re.search(rf"\b{symbol}\b", text):
                if re.search(rf"(from|import)[^\n]*\b{symbol}\b", text):
                    hits.append(rel)
    return sorted(hits)


class TestSalesAndSolutionShareOneRequirementSystem:

    def test_solution_reuses_the_shared_extractor(self):
        text = _read("src/agents/solution/nodes/requirement.py")
        assert "RequirementExtractor" in text or "get_requirement_extractor" in text
        assert "RequirementProfile" in text
        assert "profile_to_solution_requirement" in text, "旧字段只允许作为投影"

    def test_solution_does_not_define_a_second_requirement_model(self):
        text = _read("src/agents/solution/nodes/requirement.py")
        assert not re.search(r"^class \w*Requirement(Profile)?\b", text, re.M)

    def test_profile_path_comes_before_the_legacy_fallback(self):
        """主路径先看档案；旧的自然语言抽取只做兜底。"""
        text = _read("src/agents/solution/nodes/requirement.py")
        start = text.index("def understand_node(")
        end = text.index("\ndef ", start + 1)
        body = text[start:end]
        assert "requirement_profile" in body
        assert "_build_requirement_prompt(" in body
        assert body.index("requirement_profile") < body.index("_build_requirement_prompt("), (
            "understand_node 里应当先走统一档案，再考虑旧的自然语言兜底"
        )


class TestSingleRecommendationEntry:

    def test_engine_is_reached_only_through_the_coordinator(self):
        importers = _importers_of("recommendation_engine", exclude=(
            "src/rag/recommendation_engine.py",
            "src/rag/recommendation_coordinator.py",
        ))
        assert importers == [], f"生产代码不该绕过 coordinator 直接用引擎：{importers}"

    def test_coordinator_is_reached_only_through_the_service(self):
        importers = _importers_of("recommendation_coordinator", exclude=(
            "src/rag/recommendation_coordinator.py",
            "src/rag/recommendation_service.py",
        ))
        assert importers == [], f"生产代码不该绕过 service 直接用 coordinator：{importers}"

    def test_business_side_calls_the_service(self):
        assert "RecommendationService" in _read("src/agents/solution/nodes/recommend.py")
        assert "recommendation_service" in _read("src/agents/solution/nodes/recommend.py")

    def test_gate_is_part_of_the_entry(self):
        """推荐入口必过 Gate：service 与 engine 都要保留 require_ready 语义。"""
        assert "check_recommendation_ready" in _read("src/rag/recommendation_coordinator.py")
        assert "require_ready" in _read("src/rag/recommendation_engine.py")
