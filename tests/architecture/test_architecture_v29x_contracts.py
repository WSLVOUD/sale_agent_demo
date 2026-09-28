"""计划 update_v2.9.x 第十三阶段：架构约束测试（Solution 不再重复理解需求）。

计划要求至少覆盖 15 条，这里给出其中的**结构约束**部分（行为类由既有用例覆盖，
见文件末尾的对照表）：

    1  Solution 不允许调用 LLM 做需求提取            → 本文件
    2  Solution 不允许重新调用 RequirementExtractor   → 本文件
    3  Solution 不允许修改 RequirementProfile 核心字段 → 本文件
    4  Solution 不允许自行决定 display_type          → 本文件
    5  ProductTypeRouter 是唯一 display_type 决策入口 → 本文件
    6  Recommendation 只能由 Gate 放行               → 本文件
    7  requirement 不再参与新的业务决策               → 本文件
    8  LCD 不会被自动变成 LED                        → 本文件（行为）
    9  IFP 明确意图仍然能够锁定 IFP                   → 本文件（行为）
    10 Outdoor 仍然能够进入 LED                      → tests/dialogue/test_v293_product_type_router.py
    11 Indoor + 未指定类型不会直接强制 LED            → tests/dialogue/test_v294_*（收口层闸门）
    12 一轮客户消息只能产生一个问题                   → tests/dialogue/test_one_question_per_turn.py
    13 多条消息聚合后只执行一次需求理解               → tests/test_message_aggregation_chain.py
    14 原有 LED 推荐测试全部保持通过                  → 全量 pytest
    15 Calculator / Validation 测试保持通过           → tests/test_screen_calculator.py + eval
"""
import os
import re
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

SOLUTION_DIR = os.path.join(project_root, "src", "agents", "solution")


def _read(rel_path: str) -> str:
    with open(os.path.join(project_root, rel_path), encoding="utf-8") as handle:
        return handle.read()


_DOCSTRING_RE = re.compile(r'"""(?:.|\n)*?"""|\'\'\'(?:.|\n)*?\'\'\'')


def _code_only(text: str) -> str:
    without_docs = _DOCSTRING_RE.sub("", str(text or ""))
    return "\n".join(line.split("#", 1)[0] for line in without_docs.splitlines())


def _solution_modules():
    for root, _dirs, files in os.walk(SOLUTION_DIR):
        if "__pycache__" in root:
            continue
        for name in files:
            if name.endswith(".py"):
                yield os.path.relpath(os.path.join(root, name), project_root).replace("\\", "/")


class TestSolutionDoesNotUnderstandRequirements:

    def test_no_llm_requirement_extraction_in_solution(self):
        """Solution 里不许再有"需求理解"链。

        注意：响应生成（others / recommend 组织话术）用 LLM 是允许的 ——
        禁止的是**用 LLM 再抽一遍需求**，所以 `get_llm(` 只在需求节点里禁。
        """
        forbidden_anywhere = (
            "_build_requirement_prompt",
            "_parse_requirement_response",
            "RequirementExtractor",
            "get_requirement_extractor",
        )
        for rel in _solution_modules():
            code = _code_only(_read(rel))
            for token in forbidden_anywhere:
                assert token not in code, f"{rel} 仍在 Solution 里理解需求（{token}）"
            if rel.endswith("nodes/requirement.py"):
                assert "get_llm(" not in code, f"{rel} 不许可再用 LLM 提取需求"

    def test_solution_never_writes_requirement_fields(self):
        """Solution 不许改 RequirementProfile 的核心需求字段（只读）。"""
        for rel in _solution_modules():
            code = _code_only(_read(rel))
            writes = re.findall(r"\b(?:profile|_profile|requirement_profile)\.\w+\s*=[^=]", code)
            assert writes == [], f"{rel} 修改了需求档案：{writes}"

    def test_solution_never_decides_display_type(self):
        for rel in _solution_modules():
            code = _code_only(_read(rel))
            for token in ("infer_display_type", "route_display_type", "_IFP_RE", "_LED_RE"):
                assert token not in code, f"{rel} 不该自己决定产品类型（{token}）"


class TestProductTypeRouterIsTheOnlyDecider:

    def test_router_is_the_single_display_type_decision(self):
        hits = []
        for root, _dirs, files in os.walk(os.path.join(project_root, "src")):
            if "__pycache__" in root:
                continue
            for name in files:
                if not name.endswith(".py"):
                    continue
                rel = os.path.relpath(os.path.join(root, name), project_root).replace("\\", "/")
                if re.search(r"^def route_display_type\b", _read(rel), re.M):
                    hits.append(rel)
        assert hits == ["src/dialogue/product_type_router.py"], hits

    def test_lcd_is_never_silently_downgraded_to_led(self):
        from src.dialogue.product_type_router import route_display_type

        lcd = route_display_type("i need an LCD video wall")
        assert lcd.display_type == "LCD"

        later = route_display_type("it is for a meeting room, indoor", current=lcd)
        assert later.display_type == "LCD", "已确认的 LCD 不能被场景悄悄改成 LED"

    def test_explicit_ifp_intent_still_locks_ifp(self):
        from src.dialogue.product_type_router import SUBTYPE_IFP, route_display_type

        decision = route_display_type("we need handwriting on an interactive whiteboard")
        assert decision.display_type == "LCD"
        assert decision.subtype == SUBTYPE_IFP, "IFP 意图必须仍然被识别（作为 LCD 子类型）"


class TestRecommendationOnlyByGate:

    def test_solution_recommend_path_checks_the_gate(self):
        code = _code_only(_read("src/agents/solution/nodes/recommend.py"))
        assert "check_recommendation_ready" in code or "RecommendationService" in code, (
            "推荐必须过 Gate（或走会过 Gate 的统一入口）"
        )
        assert "recommendation_status" in code or "require_ready" in code

    def test_gate_module_is_the_only_readiness_source(self):
        for rel in ("src/rag/recommendation_engine.py", "src/rag/recommendation_coordinator.py"):
            assert "check_recommendation_ready" in _read(rel), rel


class TestLegacyRequirementIsNotABusinessSource:

    def test_business_decision_modules_only_use_the_profile(self):
        """Gate / 路由器 / 需求理解 / 工程推断都不允许把 legacy requirement 当决策输入。"""
        for rel in (
            "src/rag/readiness.py",
            "src/rag/query_understanding.py",
            "src/dialogue/product_type_router.py",
            "src/agents/sales/nodes/classify.py",
        ):
            code = _code_only(_read(rel))
            assert '"requirement"' not in code, f"{rel} 不该读 legacy requirement"
            assert "state.get(\"requirements\")" not in code, f"{rel} 不该读 legacy requirements"

    def test_parameter_inference_prefers_the_profile(self):
        """工程推断的唯一真值是档案；旧字典只补齐缺口。"""
        code = _read("src/rag/parameter_inference.py")
        assert "profile_to_legacy(_profile)" in code
        assert "requirement[_key] = _value" in code, "档案值必须覆盖旧字典同名键"
