"""计划《项目架构瘦身与代码整理计划》Phase 1：需求只允许有一个核心模型。

这条契约是瘦身期的护栏 —— 后面每动一次代码，都要保证它仍然成立：

  1. 全项目**只有一处**定义 ``RequirementProfile``；
  2. 需求的生产者（extractor / query_understanding / vision）都产出 ``RequirementProfile``；
  3. 决策链（Gate / 推荐 / 校验 / Vision 并入）统一吃 ``RequirementProfile``；
  4. ``RequirementBook`` 只是"多产品需求登记簿"（哪个品类有一条需求），
     **不是**第二套需求模型：决策链不得引用它；
  5. legacy 投影是只读的：``profile_to_legacy()`` 不能改动 ``RequirementProfile``。
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


class TestSingleRequirementModel:

    def test_requirement_profile_is_defined_in_exactly_one_place(self):
        defining = _scan_src(r"^class RequirementProfile\b")
        assert defining == ["src/models/requirement.py"], defining

    def test_requirement_producers_return_the_single_model(self):
        """生产者（抽取器 / 查询理解 / 图片并入）都必须落到 RequirementProfile。"""
        for rel in (
            "src/core/requirement_extractor.py",
            "src/rag/query_understanding.py",
            "src/rag/parameter_inference.py",
            "src/vision/integration.py",
            "src/vision/pipeline.py",
        ):
            assert "models.requirement import" in _read(rel), rel
            assert "RequirementProfile" in _read(rel), rel

    def test_decision_path_only_consumes_the_single_model(self):
        """Gate / 推荐 / 校验 / 会话切换都只认 RequirementProfile。"""
        for rel in (
            "src/rag/recommendation_engine.py",
            "src/rag/recommendation_coordinator.py",
            "src/rag/recommendation_service.py",
            "src/rag/validation.py",
            "src/rag/session_switch.py",
            "src/rag/multi_screen.py",
        ):
            text = _read(rel)
            assert "RequirementProfile" in text, rel
            assert "RequirementBook" not in text, f"{rel} 不该引用需求簿"
            assert "requirement_book" not in text, f"{rel} 不该引用需求簿"

        # Gate 是"用档案判断，不重建需求"：它从同一个模型里取常量，按档案对象做判断
        gate = _read("src/rag/readiness.py")
        assert "def check_recommendation_ready" in gate
        assert "models.requirement import" in gate
        assert "RequirementBook" not in gate and "requirement_book" not in gate

    def test_requirement_book_is_a_registry_not_a_second_model(self):
        """需求簿只登记"哪个品类有一条需求"，不参与需求判定。"""
        book_module = _read("src/dialogue/turn_understanding.py")
        # 登记簿的数据只有：产品域 + 状态 + 槽位快照
        assert "class Requirement:" in book_module
        for forbidden in ("missing_slots", "is_recommendation_ready", "to_legacy"):
            assert forbidden not in book_module, forbidden
        # 用它的人只有 classify（登记/切换）与理解层本身
        users = [p for p in _scan_src(r"RequirementBook|requirement_book")
                 if not p.endswith("turn_understanding.py")]
        assert users == ["src/agents/sales/nodes/classify.py", "src/memory/store.py"], users

    def test_legacy_projection_is_read_only(self):
        """legacy 投影（profile → 旧 dict）不得反向改动 RequirementProfile。"""
        from src.models.legacy_adapter import profile_to_legacy, profile_to_solution_requirement
        from src.models.requirement import RequirementProfile

        profile = RequirementProfile.from_slots(
            {"display_type": "LED", "environment": "indoor", "purpose": "conference"},
            explicit_keys={"display_type", "environment", "purpose"},
        )
        before = profile.model_dump()

        legacy = profile_to_legacy(profile)
        solution_legacy = profile_to_solution_requirement(profile)

        assert isinstance(legacy, dict) and legacy
        assert isinstance(solution_legacy, dict)
        assert profile.model_dump() == before, "投影不能改原档案"

    def test_session_persists_the_profile_in_one_json_form(self):
        """会话里的档案只有一份：dict 形态持久化，可无损还原成 RequirementProfile。"""
        from src.memory.store import memory
        from src.models.requirement import RequirementProfile

        session_id = "arch-single-model"
        memory.clear(session_id)
        try:
            profile = RequirementProfile.from_slots({"environment": "indoor"}, explicit_keys={"environment"})
            memory.set_requirement_profile(session_id, profile)
            stored = memory.get_requirement_profile(session_id)
            assert isinstance(stored, dict), "会话里存的是 JSON 形态（供跨轮/跨进程）"
            assert stored.get("environment") == "indoor"
            restored = RequirementProfile.model_validate(stored)
            assert restored.environment == "indoor"
        finally:
            memory.clear(session_id)
