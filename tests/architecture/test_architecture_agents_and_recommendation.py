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

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


class TestRecommendedDoesNotReParseRequirements:
    """Phase 12-8 护栏 6：Recommendation 不允许重新解析客户需求。"""

    def test_engine_only_reaches_for_a_profile_in_one_documented_fallback(self):
        """引擎里的"没有 profile 就自己解析"只允许存在一处（兼容回退），且必须标注。"""
        import re as _re

        text = _read("src/rag/recommendation_engine.py")
        calls = _re.findall(r"understand_query\(", text)
        assert len(calls) == 1, f"兼容回退只允许一处，实际 {len(calls)} 处"
        marker = "LEGACY-COMPAT"
        assert marker in text, "兼容回退必须显式标注 LEGACY-COMPAT"

    def test_production_entry_already_has_a_profile(self):
        """生产入口必须把档案传进来（否则就会退化成兼容回退）。"""
        assert "RequirementProfile" in _read("src/rag/recommendation_coordinator.py")
        service = _read("src/rag/recommendation_service.py")
        assert "RequirementProfile" in service
        assert "profile" in service


class TestLegacyRequirementWritesAreMarkedCompatible:
    """Phase 12-6：legacy requirement 的写入只允许留在兼容分支，并且要标出来。"""

    def test_solution_node_writes_are_marked(self):
        text = _read("src/agents/solution/nodes/requirement.py")
        assert text.count("LEGACY-COMPAT") >= 2, "legacy requirement 的写入处必须标注"

    def test_legacy_dict_readers_are_confined(self):
        """旧字典的读者只允许是兼容消费者（others 的自由问答 + runner 的收尾整理）。"""
        import os as _os
        import re as _re

        readers = []
        for root, _dirs, files in _os.walk(os.path.join(project_root, "src")):
            if "__pycache__" in root:
                continue
            for name in files:
                if not name.endswith(".py"):
                    continue
                rel = _os.path.relpath(_os.path.join(root, name), project_root).replace("\\", "/")
                text = _read(rel)
                if _re.search(r'state\.get\("requirement"\)|state\["requirement"\]', text):
                    readers.append(rel)
        assert readers == [
            "src/agents/solution/runner.py",
            "src/agents/solution/nodes/others.py",
            "src/agents/solution/nodes/requirement.py",
            "src/rag/model_guard.py",
        ], (
            "旧 requirement 字典的读者只允许这 4 个兼容消费者："
            "runner（组装 state）/ others（自由问答取上下文）/ "
            "solution requirement 节点（无档案时的兼容分支）/ model_guard（兼容旧字段形状）"
        )
