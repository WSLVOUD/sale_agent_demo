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

    def test_solution_does_not_extract_requirements_itself_merged(self):
        """合并自 3 条同类测试（瘦身；断言全部保留）。"""

        # ── test_solution_does_not_extract_requirements_itself ──
        """计划 update_v2.9.x 第三/四阶段：Solution 不再理解需求。

        （本条取代上一版"Solution 复用统一抽取器"的旧断言 —— 现在连复用都不允许：
        需求只由 Sales 侧的 RequirementExtractor 产出一次。）
        """
        text = _read("src/agents/solution/nodes/requirement.py")
        for token in ("RequirementExtractor", "get_requirement_extractor", "get_llm", "llm.invoke"):
            assert token not in text, f"Solution 不该再自己理解需求（{token}）"
        assert "RequirementProfile" in text, "只能读档案"

        # ── test_solution_does_not_define_a_second_requirement_model ──
        text = _read("src/agents/solution/nodes/requirement.py")
        assert not re.search(r"^class \w*Requirement(Profile)?\b", text, re.M)

        # ── test_understand_node_is_a_pure_adapter ──
        """understand_node 只做：读档案 → Gate 状态 → 返回（不再拼 prompt / 解析 JSON）。"""
        text = _read("src/agents/solution/nodes/requirement.py")
        start = text.index("def understand_node(")
        end = text.index("\ndef ", start + 1)
        body = text[start:end]
        assert "requirement_profile" in body
        assert "check_recommendation_ready" in body, "状态判断用确定性 Gate"
        for token in ("_build_requirement_prompt", "_parse_requirement_response", "json.loads"):
            assert token not in body, f"understand_node 不该再有旧需求解析（{token}）"


class TestSingleRecommendationEntry:

    def test_engine_is_reached_only_through_the_coordinator_merged(self):
        """合并自 4 条同类测试（瘦身；断言全部保留）。"""

        # ── test_engine_is_reached_only_through_the_coordinator ──
        importers = _importers_of("recommendation_engine", exclude=(
            "src/rag/recommendation_engine.py",
            "src/rag/recommendation_coordinator.py",
        ))
        assert importers == [], f"生产代码不该绕过 coordinator 直接用引擎：{importers}"

        # ── test_coordinator_is_reached_only_through_the_service ──
        importers = _importers_of("recommendation_coordinator", exclude=(
            "src/rag/recommendation_coordinator.py",
            "src/rag/recommendation_service.py",
        ))
        assert importers == [], f"生产代码不该绕过 service 直接用 coordinator：{importers}"

        # ── test_business_side_calls_the_service ──
        assert "RecommendationService" in _read("src/agents/solution/nodes/recommend.py")
        assert "recommendation_service" in _read("src/agents/solution/nodes/recommend.py")

        # ── test_gate_is_part_of_the_entry ──
        """推荐入口必过 Gate：service 与 engine 都要保留 require_ready 语义。"""
        assert "check_recommendation_ready" in _read("src/rag/recommendation_coordinator.py")
        assert "require_ready" in _read("src/rag/recommendation_engine.py")


class TestRecommendedDoesNotReParseRequirements:
    """Phase 12-8 护栏 6：Recommendation 不允许重新解析客户需求。"""

    def test_engine_only_reaches_for_a_profile_in_one_documented_fallback_merged(self):
        """合并自 2 条同类测试（瘦身；断言全部保留）。"""

        # ── test_engine_only_reaches_for_a_profile_in_one_documented_fallback ──
        """引擎里的"没有 profile 就自己解析"只允许存在一处（兼容回退），且必须标注。"""
        import re as _re

        text = _read("src/rag/recommendation_engine.py")
        calls = _re.findall(r"understand_query\(", text)
        assert len(calls) == 1, f"兼容回退只允许一处，实际 {len(calls)} 处"
        marker = "LEGACY-COMPAT"
        assert marker in text, "兼容回退必须显式标注 LEGACY-COMPAT"

        # ── test_production_entry_already_has_a_profile ──
        """生产入口必须把档案传进来（否则就会退化成兼容回退）。"""
        assert "RequirementProfile" in _read("src/rag/recommendation_coordinator.py")
        service = _read("src/rag/recommendation_service.py")
        assert "RequirementProfile" in service
        assert "profile" in service


class TestLegacyRequirementWritesAreMarkedCompatible:
    """计划 update_v2.9.x 第五/九阶段：Solution 侧的 legacy 需求写入已全部删除。"""

    def test_solution_nodes_do_not_write_the_legacy_requirement_merged(self):
        """合并自 2 条同类测试（瘦身；断言全部保留）。"""

        # ── test_solution_nodes_do_not_write_the_legacy_requirement ──
        """Solution 的需求节点不再往旧字典里写东西（只有 runner 的投影与兼容消费者）。"""
        text = _read("src/agents/solution/nodes/requirement.py")
        writes = re.findall(r'"requirement"\s*:', text)
        assert writes == [], f"Solution 节点不该再写 legacy requirement：{writes}"

        # ── test_legacy_dict_readers_are_confined ──
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
