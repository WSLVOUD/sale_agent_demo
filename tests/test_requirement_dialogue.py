"""
v2.0 Phase 4 / Phase 9 / Phase 21：对话与 Gate 测试。

覆盖 v2.0 文档「二十一、测试计划」中的必测场景：
  1. 推荐过早（LED / LED+Indoor / Indoor+Conference → 都不推荐）
  2. 正常推荐（Indoor + Conference + Fixed + 5m → 推荐）
  3. 明确参数（Indoor + Fixed + P2.5 → 可以推荐）
  4. 环境冲突（Outdoor → 不能推荐 Indoor）
  5. 计算（5m × 3m → Cabinet / Module / Actual Size 正确）
  6. 多轮（场景 → 环境 → 安装方式 → 观看距离 → 推荐）
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.readiness import (  # noqa: E402
    check_calculation_ready,
    check_recommendation_ready,
)
from src.rag.recommendation_engine import RecommendationEngine  # noqa: E402
from src.tools.screen_calculator import calculate_screen  # noqa: E402


def _ready(slots: dict) -> bool:
    """构造"客户明确说出全部字段"的档案并判定（测试 helper）。"""
    return check_recommendation_ready(
        RequirementProfile.from_slots(slots, explicit_keys=set(slots))
    ).ready


@pytest.fixture
def sales_llm(monkeypatch):
    """把 Sales Agent 的 ChatOpenAI 换成返回指定 JSON 的假实现。

    默认只返回"场景 + 室内外"，用于验证：**只有这两项时不得触发推荐**。
    """
    import src.agents.sales.nodes.requirement as sales_req

    class _Response:
        content = '{"usage": "会议室", "location_type": "室内"}'

    class _FakeChat:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, *args, **kwargs):
            return _Response()

    monkeypatch.setattr(sales_req, "ChatOpenAI", _FakeChat)
    return sales_req


def _run_turn(sales_req, message, requirements=None, messages=None):
    """驱动一轮 Sales Agent 需求挖掘。"""
    state = {
        "messages": list(messages or []) + [{"role": "user", "content": message}],
        "current_message": message,
        "requirements": dict(requirements or {}),
        "additional_requirements": [],
        "intent": "need_query",
        "next_action": "ask",
        "should_generate_solution": False,
        "response": "",
    }
    return sales_req.requirement_mining(state)


class TestInferredValuesMustNotTriggerRecommendation:
    """回归：只知道"室内外 + 场景"绝不能直接推荐（用户实测反馈的问题）。

    根因是系统会**凭空估算**观看距离（"能容纳 15 个人" → 3~5 米），
    把 Ready Gate 的四个条件凑齐。推断值现在必须被标记为 inferred，
    只能用于打分提示，不能打开 Gate。
    """

    @pytest.mark.parametrize("message", [
        "室内会议室用LED屏",
        "能容纳15个人的会议室",
        "会议室20平米左右",
        "会议室大概15人用，推荐一块显示屏",
        "我想在展厅放一块屏",
    ])
    def test_scene_only_does_not_recommend(self, sales_llm, message):
        turn = _run_turn(sales_llm, message)
        assert turn["should_generate_solution"] is False, (
            f"{message} 不应触发推荐（Gate={turn.get('recommendation_gate')}）"
        )
        assert turn.get("pending_question"), "未就绪时必须继续追问"

    def test_estimated_distance_is_marked_inferred(self, sales_llm):
        turn = _run_turn(sales_llm, "能容纳15个人的会议室")
        profile = turn["requirement_profile"]
        # 彻底修复：不再从"人数/面积"估算观看距离 —— 没有就是没有，交给追问
        assert profile.viewing_distance_m is None
        assert profile.sources.get("viewing_distance_m") is None
        assert turn["should_generate_solution"] is False

    def test_explicit_installation_and_distance_do_recommend(self, sales_llm):
        turn = _run_turn(sales_llm, "室内会议室，固定安装，5米视距")
        assert turn["should_generate_solution"] is True
        profile = turn["requirement_profile"]
        assert profile.sources.get("installation") == "explicit"
        assert profile.sources.get("viewing_distance_m") == "explicit"

    def test_purpose_default_installation_is_not_treated_as_confirmed(self, sales_llm):
        """场景默认"固装"是系统猜的，必须继续追问安装方式"""
        turn = _run_turn(sales_llm, "室内会议室用LED屏")
        missing = (turn.get("recommendation_gate") or {}).get("missing") or []
        assert "installation" in missing
        assert "viewing_distance" in missing

    def test_solution_gate_also_blocks_inferred_profile(self):
        """方案侧 Gate 用同一规则：只带推断值的 requirement 不得进入检索"""
        from src.agents.solution.nodes.requirement import recommendation_gate_node

        result = recommendation_gate_node({
            "requirement": {
                "usage": "会议室", "location_type": "室内",
                "viewing_distance": "3-5米", "size": "约42平米",
                "_inferred_slots": ["viewing_distance", "size"],
            },
            "messages": [],
            "current_message": "能容纳15个人的会议室",
        })
        assert result["next_action"] == "clarify"
        assert result["recommendation_gate"]["ready"] is False

class TestScenario1NotTooEarly:
    """场景 1：推荐过早 —— 都必须先追问"""

    def test_led_only(self):
        decision = check_recommendation_ready(RequirementProfile.from_slots({"display_type": "LED"}))
        assert decision.ready is False
        assert decision.next_question, "未就绪时必须给出一个追问"

    def test_led_plus_indoor(self):
        assert _ready({"display_type": "LED", "environment": "indoor"}) is False

    def test_indoor_conference_room(self):
        """v2.0 Case 2：Indoor + Conference room 仍然不够"""
        decision = check_recommendation_ready(
            RequirementProfile.from_slots(
                {"environment": "indoor", "purpose": "conference"},
                explicit_keys={"environment", "purpose"},
            )
        )
        assert decision.ready is False
        assert set(decision.missing) == {"installation", "viewing_distance"}

    def test_missing_is_asked_in_priority_order(self):
        decision = check_recommendation_ready(RequirementProfile.from_slots({"display_type": "LED"}))
        assert decision.missing[0] == "environment"
        assert "indoors or outdoors" in decision.next_question


class TestScenario2Recommend:
    """场景 2 / 3：可以推荐的情形"""

    def test_full_scene_is_ready(self):
        """v2.0 Case 3"""
        slots = {
            "environment": "indoor", "purpose": "conference",
            "installation": "fixed", "viewing_distance_m": 5,
        }
        decision = check_recommendation_ready(
            RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        )
        assert decision.ready is True
        assert decision.missing == []

    def test_explicit_specs_are_ready_without_distance(self):
        """v2.0 Case 4：客户直接给出点间距即可推荐"""
        slots = {
            "environment": "indoor", "installation": "fixed", "pixel_pitch_mm": 2.5,
        }
        decision = check_recommendation_ready(
            RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        )
        assert decision.ready is True
        assert "技术规格" in decision.reason

    def test_named_model_is_ready(self):
        decision = check_recommendation_ready(
            RequirementProfile.from_slots({"model": "TW21-3216-P2.5"}, explicit_keys={"model"})
        )
        assert decision.ready is True


class TestScenario4EnvironmentConflict:
    """场景 4：Outdoor 需求不得推荐 Indoor 产品"""

    def test_outdoor_never_returns_indoor(self):
        engine = RecommendationEngine()
        profile = RequirementProfile.from_slots({
            "environment": "outdoor", "purpose": "advertising",
            "installation": "fixed", "viewing_distance_m": 20,
        })
        result = engine.recommend(profile=profile)
        assert result["recommendations"]
        assert all(rec["outdoor"] for rec in result["recommendations"])
        assert result["violations"] == []


class TestScenario5CalculationGate:
    """场景 5：Calculation Ready Gate"""

    def test_without_size_recommends_but_skips_calculation(self):
        profile = RequirementProfile.from_slots({
            "environment": "indoor", "purpose": "conference",
            "installation": "fixed", "viewing_distance_m": 5,
        })
        decision = check_calculation_ready(profile)
        assert decision.ready is False
        assert set(decision.missing) == {"width", "height"}

    def test_with_size_calculation_is_ready(self):
        profile = RequirementProfile.from_slots({
            "environment": "indoor", "purpose": "conference",
            "installation": "fixed", "viewing_distance_m": 5,
            "target_width_mm": 5000, "target_height_mm": 3000,
        })
        assert check_calculation_ready(profile).ready is True

        calc = calculate_screen("TW21-3216-P2.5", 5000, 3000)
        assert calc["cabinet_count"] == 56
        assert calc["total_modules"] == 336
        assert (calc["actual_width_mm"], calc["actual_height_mm"]) == (5120, 3360)


class TestScenario5bSizeDoesNotChangeSelection:
    """v2.0 Phase 9 闭环：先推荐、后补尺寸 → 型号不变，只是把计算补上"""

    def test_adding_size_keeps_the_same_model(self):
        import json

        engine = RecommendationEngine()
        with open(os.path.join(project_root, "eval", "golden_dataset.json"), encoding="utf-8") as handle:
            cases = json.load(handle)["cases"]

        checked = 0
        no_match = 0
        for case in cases:
            slots = dict(case.get("slots") or {})
            if not slots:
                continue
            profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
            if not check_recommendation_ready(profile).ready:
                continue
            without = engine.recommend(profile=profile)["recommendations"]
            size_slots = {**slots, "target_width_mm": 5000, "target_height_mm": 3000}
            with_size = engine.recommend(profile=RequirementProfile.from_slots(
                size_slots, explicit_keys=set(size_slots)
            ))["recommendations"]
            if not without or not with_size:
                # 产品库缺该品类（如 LCD / IFP 或户外租赁）→ 属于数据边界，不参与本断言
                no_match += 1
                continue
            checked += 1
            assert without[0]["model"] == with_size[0]["model"], (
                f"{case['id']}: 补尺寸后选型发生变化 "
                f"{without[0]['model']} → {with_size[0]['model']}"
            )
        print(f"\n  可比对用例 {checked} 条，数据边界（无匹配）{no_match} 条")
        assert checked >= 20, f"可判定用例过少（{checked}），检查黄金用例的 slots 是否完整"


class TestScenario6MultiTurn:
    """场景 6：多轮采集（场景 → 环境 → 安装方式 → 观看距离 → 推荐）

    直接驱动 Sales Agent 的 ``requirement_mining`` 节点（LLM 被 mock 掉），
    验证 Gate 在真实节点里按预期逐轮放行。
    """

    def test_multi_turn_asks_then_recommends(self, sales_llm):
        messages = []
        requirements = {}
        asked = []

        # 第 1 轮：只给产品类型 → 追问
        turn = _run_turn(sales_llm, "I need an LED display", requirements, messages)
        assert turn["should_generate_solution"] is False
        assert turn.get("pending_question")
        asked.append(turn.get("pending_question"))
        requirements = dict(turn["requirements"])
        messages = list(turn["messages"])

        # 第 2 轮：给出场景 → 仍然追问（v2.0 Case 2）
        turn = _run_turn(sales_llm, "It is for a conference room", requirements, messages)
        assert turn["should_generate_solution"] is False, "只有场景时不应触发推荐"
        asked.append(turn.get("pending_question"))
        requirements = dict(turn["requirements"])
        messages = list(turn["messages"])

        # 第 3 轮：给出安装方式 → 仍缺观看距离，继续追问
        turn = _run_turn(sales_llm, "Fixed installation", requirements, messages)
        assert turn["should_generate_solution"] is False
        asked.append(turn.get("pending_question"))
        requirements = dict(turn["requirements"])
        messages = list(turn["messages"])

        # 第 4 轮：补观看距离 → Gate 放行
        turn = _run_turn(sales_llm, "Viewing distance is about 5 meters", requirements, messages)
        assert turn["should_generate_solution"] is True, (
            f"四类信息齐备后应触发推荐；已问过: {asked}"
        )

    def test_explicit_specs_shortcut_the_dialogue(self, sales_llm):
        """客户一上来就给足规格 → 第一轮即推荐"""
        turn = _run_turn(
            sales_llm,
            "I need an indoor fixed LED screen, P2.5, for a conference room",
            {},
            [],
        )
        assert turn["should_generate_solution"] is True


class TestSolutionGraphGate:
    """Gate 必须真正接入 Solution Agent 图"""

    def test_graph_contains_recommendation_gate(self):
        from src.agents.solution.graph import build_solution_graph

        nodes = set(build_solution_graph().nodes)
        assert "recommendation_gate_check" in nodes

    def test_gate_node_routes_to_clarify_when_not_ready(self):
        from src.agents.solution.nodes.requirement import recommendation_gate_node

        result = recommendation_gate_node({
            "requirement": {"display_type": "LED"},
            "messages": [],
            "current_message": "I need an LED display",
        })
        assert result["next_action"] == "clarify"
        assert result["recommendation_gate"]["ready"] is False
        assert result["pending_question"]

    def test_gate_node_routes_to_retrieve_when_ready(self):
        from src.agents.solution.nodes.requirement import recommendation_gate_node

        result = recommendation_gate_node({
            "requirement": {},
            # Gate 只信"客户原话"：把事实放进用户消息，而不是 requirement
            "messages": [{"role": "user", "content": "室内会议室5米视距，固定安装"}],
            "current_message": "室内会议室5米视距，固定安装",
        })
        assert result["next_action"] == "retrieve"
        assert result["recommendation_gate"]["ready"] is True


class TestScenarioAnswerIsNotOthers:
    """回归：客户回答"用在什么场景"时，即使被 classify 误判为 others，
    也必须继续需求采集，绝不能绕到 Solution 的自由问答检索。"""

    def test_scenario_answer_reclassifies_to_need_query(self, monkeypatch):
        import src.agents.sales.nodes.requirement as sales_req

        class _Response:
            content = '{"usage": "church"}'

        class _FakeChat:
            def __init__(self, *args, **kwargs):
                pass

            def invoke(self, *args, **kwargs):
                return _Response()

        monkeypatch.setattr(sales_req, "ChatOpenAI", _FakeChat)

        # 模拟真实 bug：classify 已经判成 others
        state = {
            "messages": [{"role": "user", "content": "church"}],
            "current_message": "church",
            "requirements": {
                "display_type": "LED", "location_type": "室内",
                "indoor": True, "outdoor": False,
            },
            "additional_requirements": [],
            "intent": "others",
            "next_action": "others",
            "should_generate_solution": False,
            "response": "",
        }
        result = sales_req.requirement_mining(state)

        assert result["intent"] == "need_query"
        assert result["should_generate_solution"] is False
        assert result.get("pending_question"), "必须继续追问，而不是放行到自由问答"
        gate = result.get("recommendation_gate") or {}
        assert "installation" in gate.get("missing", [])

    def test_church_is_recognized_as_scenario(self):
        from src.rag.query_understanding import extract_slots
        from src.rag.parameter_inference import detect_intent

        assert extract_slots("church").get("purpose") == "church"
        assert detect_intent("church") == "recommendation"
