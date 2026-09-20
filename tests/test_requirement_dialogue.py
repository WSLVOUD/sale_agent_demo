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
        # 客户口径（2026-09-18）：硬性条件 = 尺寸 + P值/视距 + 室内外 + 固装租赁
        turn = _run_turn(sales_llm, "室内会议室，5米x3米，固定安装，5米视距，放视频为主，价格优先")
        assert turn["should_generate_solution"] is True
        profile = turn["requirement_profile"]
        assert profile.sources.get("installation") == "explicit"
        assert profile.sources.get("viewing_distance_m") == "explicit"

    def test_purpose_default_installation_is_not_treated_as_confirmed(self, sales_llm):
        """场景默认"固装"是系统猜的，必须继续追问安装方式"""
        turn = _run_turn(sales_llm, "室内会议室用LED屏")
        missing = (turn.get("recommendation_gate") or {}).get("missing") or []
        assert "installation" in missing
        assert "pixel_pitch" in missing
        assert "size" in missing

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
        # 客户口径（2026-09-18）：只差硬性条件（固装租赁 / P值 / 尺寸）
        assert set(decision.missing) == {"installation", "pixel_pitch", "size"}
        assert decision.missing.index("installation") < decision.missing.index("pixel_pitch")
        assert decision.missing.index("pixel_pitch") < decision.missing.index("size")

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
            "content_type": "mixed", "installation": "fixed", "viewing_distance_m": 5,
            "price_preference": "price",
            "target_width_mm": 5000, "target_height_mm": 3000,
        }
        decision = check_recommendation_ready(
            RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        )
        assert decision.ready is True
        assert decision.missing == []

    def test_explicit_specs_are_ready_without_distance(self):
        """v2.0 Case 4：客户直接给出点间距 + 尺寸 + 室内外 + 固装租赁即可推荐"""
        slots = {
            "environment": "indoor", "installation": "fixed", "pixel_pitch_mm": 2.5,
            "purpose": "conference", "content_type": "mixed",
            "price_preference": "price",
            "target_width_mm": 5000, "target_height_mm": 3000,
        }
        decision = check_recommendation_ready(
            RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        )
        assert decision.ready is True
        assert decision.status == "READY"

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


class TestScenario5bSizeDoesNotBreakSelection:
    """v2.0 Phase 9 闭环：尺寸参与打分（箱体排布 20%），但不得破坏硬约束。

    注：客户口径（2026-09-18）把"尺寸"升级成**硬性条件**（推荐前必须拿到），
    所以这里不再断言"补尺寸后型号一定不变"——尺寸本身会影响箱体排布的匹配度；
    要守住的是：**选出来的必须是真实型号，且仍然满足环境 / 安装方式等硬条件**。
    """

    def test_adding_size_keeps_a_valid_model(self):
        import json
        from src.config import config
        from src.rag.json_loader import canonical_model_index

        engine = RecommendationEngine()
        model_index = canonical_model_index(config.DATA_DIR)
        with open(os.path.join(project_root, "eval", "golden_dataset.json"), encoding="utf-8") as handle:
            cases = json.load(handle)["cases"]

        checked = 0
        no_match = 0
        for case in cases:
            slots = dict(case.get("slots") or {})
            if not slots:
                continue
            # 客户口径：推荐前还需要"内容类型"与"价格/质量取向"（都不改变本断言的结论）
            slots.setdefault("content_type", "mixed")
            slots.setdefault("price_preference", "price")
            profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
            # 客户口径（2026-09-18）：尺寸成为硬性条件后，黄金用例大多没有尺寸
            # → 这里不再用 Gate 过滤（引擎本身不需要尺寸也能选型），
            # 只要求用例带选型依据（环境或场景）。
            if not (slots.get("environment") or slots.get("purpose")):
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
            for picked in (without[0], with_size[0]):
                assert picked["model"] in model_index, (case["id"], picked["model"])
            if slots.get("environment") == "indoor":
                assert with_size[0]["indoor"] is True, (case["id"], with_size[0])
            elif slots.get("environment") == "outdoor":
                assert with_size[0]["outdoor"] is True, (case["id"], with_size[0])
            if slots.get("installation") == "rental":
                assert with_size[0]["installation"] == "rental", (case["id"], with_size[0])
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
        # 客户口径：明显的室内场景（会议室）不再问"室内还是室外"
        assert turn["requirement_profile"].environment == "indoor"
        asked.append(turn.get("pending_question"))
        requirements = dict(turn["requirements"])
        messages = list(turn["messages"])

        # 第 3 轮：给出安装方式 → 还缺点间距，继续追问
        turn = _run_turn(sales_llm, "Fixed installation", requirements, messages)
        assert turn["should_generate_solution"] is False
        asked.append(turn.get("pending_question"))
        requirements = dict(turn["requirements"])
        messages = list(turn["messages"])

        # 第 4 轮：点间距不知道 → 按规则转问观看距离
        turn = _run_turn(sales_llm, "I don't know", requirements, messages)
        assert turn["should_generate_solution"] is False
        asked.append(turn.get("pending_question"))
        requirements = dict(turn["requirements"])
        messages = list(turn["messages"])

        # 第 5 轮：补观看距离 → 还缺尺寸
        turn = _run_turn(sales_llm, "Viewing distance is about 5 meters", requirements, messages)
        assert turn["should_generate_solution"] is False
        asked.append(turn.get("pending_question"))
        requirements = dict(turn["requirements"])
        messages = list(turn["messages"])

        # 第 6 轮：补尺寸 → 硬性条件齐备，Gate 放行（不再问内容类型/价格取向）
        turn = _run_turn(sales_llm, "The screen is 5m x 3m", requirements, messages)
        assert turn["should_generate_solution"] is True, (
            f"信息齐备后应触发推荐；已问过: {asked}"
        )
        assert turn["requirement_profile"].has_target_size

    def test_explicit_specs_shortcut_the_dialogue(self, sales_llm):
        """客户一上来就给足规格 → 第一轮即推荐"""
        turn = _run_turn(
            sales_llm,
            "I need an indoor fixed LED screen, P2.5, 5m x 3m, for a conference room",
            {},
            [],
        )
        assert turn["should_generate_solution"] is True


class TestCloseAnswerIsNotClosing:
    """实测回归（客户日志）：

    客户用 "close"（意思是"近"）回答观看距离，被意图分类器判成 closing，
    结果整轮被强制不推荐、只回了一句"我会准备报价"，既不推荐产品也不问尺寸。
    这里锁定三件事：
      1. "close / near / far / 5-10 米"这类模糊回答能被解析成观看距离；
      2. "close" 不会被判成"结束对话"，但 "bye / 不用了" 仍然会；
      3. Gate 已经就绪时，非明确结束的 closing 不能吞掉推荐。
    """

    def test_rough_distance_answers_are_parsed(self):
        from src.rag.query_understanding import extract_slots

        assert extract_slots("close").get("viewing_distance_m") == 3.0
        assert extract_slots("near").get("viewing_distance_m") == 3.0
        assert extract_slots("very close").get("viewing_distance_m") == 2.0
        assert extract_slots("far").get("viewing_distance_m") == 15.0
        assert extract_slots("5-10 metres").get("viewing_distance_m") == 7.5
        assert extract_slots("more than 10m").get("viewing_distance_m") == 15.0
        assert extract_slots("within 3m").get("viewing_distance_m") == 1.8

    def test_rough_parser_does_not_over_trigger(self):
        from src.rag.query_understanding import extract_slots

        assert extract_slots("close the deal").get("viewing_distance_m") is None
        assert extract_slots("i dont know").get("viewing_distance_m") is None

    def _classify(self, monkeypatch, message, llm_intent):
        # 注意：nodes/__init__.py 里 `from .classify import classify` 会把子模块名遮蔽掉，
        # 所以必须从 sys.modules 取真正的模块对象
        import sys

        classify_mod = sys.modules["src.agents.sales.nodes.classify"]

        class _Response:
            content = llm_intent

        class _FakeChat:
            def __init__(self, *args, **kwargs):
                pass

            def invoke(self, *args, **kwargs):
                return _Response()

        monkeypatch.setattr(classify_mod, "ChatOpenAI", _FakeChat)
        state = {"current_message": message, "intent": "", "next_action": ""}
        return classify_mod.classify(state)

    def test_close_is_not_treated_as_closing(self, monkeypatch):
        state = self._classify(monkeypatch, "close", "closing")
        assert state["intent"] == "need_query", "「近」不能被当成「结束对话」"

    def test_explicit_closing_still_closes(self, monkeypatch):
        state = self._classify(monkeypatch, "thanks, that's all. bye", "closing")
        assert state["intent"] == "closing"

        state = self._classify(monkeypatch, "不用了，谢谢", "closing")
        assert state["intent"] == "closing"

    def _ready_profile(self):
        slots = {
            "display_type": "LED",
            "environment": "indoor",
            "purpose": "conference",
            "content_type": "mixed",
            "installation": "fixed",
            "price_preference": "price",
            "viewing_distance_m": 3,
            "target_width_mm": 5000,
            "target_height_mm": 3000,
        }
        return RequirementProfile.from_slots(slots, explicit_keys=set(slots))

    def test_ready_gate_still_recommends_when_intent_is_closing(self, sales_llm):
        state = {
            "messages": [
                {"role": "user", "content": "I need a display like this"},
                {"role": "user", "content": "close"},
            ],
            "current_message": "close",
            "requirements": {},
            "additional_requirements": [],
            "intent": "closing",
            "next_action": "end",
            "should_generate_solution": False,
            "response": "",
            "requirement_profile": self._ready_profile(),
            "session_id": "closing-guard-ready",
        }
        out = sales_llm.requirement_mining(state)

        assert out["should_generate_solution"] is True, "Gate 已就绪就必须推荐"
        assert out["next_action"] == "router"
        assert out["intent"] == "need_query"

    def test_explicit_closing_does_not_force_recommendation(self, sales_llm):
        state = {
            "messages": [{"role": "user", "content": "bye, thanks a lot"}],
            "current_message": "bye, thanks a lot",
            "requirements": {},
            "additional_requirements": [],
            "intent": "closing",
            "next_action": "end",
            "should_generate_solution": False,
            "response": "",
            "requirement_profile": self._ready_profile(),
            "session_id": "closing-guard-stop",
        }
        out = sales_llm.requirement_mining(state)
        assert out["should_generate_solution"] is False


class TestPitchAskedBeforeDistance:
    """客户口径：先问点间距；客户不知道 → 不再问第二遍点间距，转问观看距离。

    （观看距离仍然保持"最多问两遍"的老规则。）
    """

    def _profile_without_pitch(self):
        # 只缺点间距与观看距离（其它都已确认，这样"先问点间距"才看得出来）
        slots = {"display_type": "LED", "environment": "indoor", "purpose": "conference",
                 "content_type": "mixed", "installation": "fixed", "price_preference": "price",
                 "target_width_mm": 5000, "target_height_mm": 3000}
        return RequirementProfile.from_slots(slots, explicit_keys=set(slots))

    def _turn(self, module, message, profile):
        state = {
            "messages": [{"role": "user", "content": message}],
            "current_message": message,
            "session_id": "pitch-order",
            "requirements": {},
            "additional_requirements": [],
            "intent": "need_query",
            "next_action": "ask",
            "should_generate_solution": False,
            "response": "",
            "pending_question": "",
            "pending_slot": "",
            "requirement_profile": profile,
        }
        return module.requirement_mining(state)

    def test_pitch_is_asked_before_viewing_distance(self):
        from src.rag.readiness import check_recommendation_ready

        decision = check_recommendation_ready(self._profile_without_pitch())
        assert decision.missing[0] == "pixel_pitch", decision.missing
        # 先问点间距；只有客户说"不知道"时才转问观看距离
        assert decision.missing == ["pixel_pitch"]
        assert "pitch" in (decision.next_question or "").lower()

    def test_customer_gives_pitch_skips_distance_question(self):
        from src.rag.readiness import check_recommendation_ready

        slots = {"pixel_pitch_mm": 3}
        profile = self._profile_without_pitch().merge(
            RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        )
        decision = check_recommendation_ready(profile)
        assert decision.ready is True, "客户明确给了点间距 → 按客户的选型，直接放行"

    def test_dont_know_pitch_moves_to_viewing_distance(self, sales_llm):
        state = self._turn(sales_llm, "we need an indoor LED screen for a conference room",
                           self._profile_without_pitch())
        assert state["pending_slot"] == "pixel_pitch"
        assert state["requirement_profile"].ask_count("pixel_pitch") == 1

        # 客户说不知道 → 只问一次，直接转观看距离
        state = self._turn(sales_llm, "I don't know", state["requirement_profile"])
        profile = state["requirement_profile"]
        assert profile.ask_count("pixel_pitch") == 1, "点间距不该问第二遍"
        assert profile.is_unknown("pixel_pitch") is True
        assert state["pending_slot"] == "viewing_distance"

    def test_delegating_pitch_stops_asking(self, sales_llm):
        """v2.1：客户说"你决定吧"是**授权 AI 决定**（DELEGATED），不是"不知道"。

        计划 4.2 要求 DELEGATED 与 UNKNOWN 严格区分；计划 14 / 15 要求授权之后
        走 Python 确定性推导（点间距按环境 + 观看距离 / 场景推导），不再追问同一项。
        """
        state = self._turn(sales_llm, "we need an indoor LED screen for a conference room",
                           self._profile_without_pitch())
        assert state["pending_slot"] == "pixel_pitch"
        state = self._turn(sales_llm, "你决定吧", state["requirement_profile"])
        profile = state["requirement_profile"]
        assert profile.field_decision("pixel_pitch") == "DELEGATED"
        assert profile.is_delegated("pixel_pitch") is True
        assert state["pending_slot"] != "pixel_pitch", "客户授权 AI 决定后不能再问点间距"

    def test_viewing_distance_still_asked_twice(self, sales_llm):
        """观看距离保持老规则：不知道 → 降门槛再问一次 → 仍不知道才跳过。"""
        profile = self._profile_without_pitch()
        profile.record_ask("pixel_pitch")
        profile.mark_unknown("pixel_pitch", "customer_does_not_know")

        state = self._turn(sales_llm, "5 meters", profile)
        # 客户直接给了距离 → 放行推荐
        assert state["recommendation_gate"]["status"] in ("READY", "DEGRADED_READY")

        profile2 = self._profile_without_pitch()
        profile2.record_ask("pixel_pitch")
        profile2.mark_unknown("pixel_pitch", "customer_does_not_know")
        profile2.record_ask("viewing_distance")
        profile2.mark_unknown("viewing_distance", "customer_does_not_know")
        assert profile2.ask_count("viewing_distance") == 1
        assert profile2.is_unknown("viewing_distance") is False, "第一次不知道还不该跳过"

    def test_pitch_parsing_forms(self):
        from src.rag.query_understanding import extract_slots

        assert extract_slots("P2.5").get("pixel_pitch_mm") == 2.5
        assert extract_slots("p3").get("pixel_pitch_mm") == 3.0
        assert extract_slots("3mm pitch").get("pixel_pitch_mm") == 3.0
        assert extract_slots("点间距3mm").get("pixel_pitch_mm") == 3.0

    def test_pitch_question_has_variants(self):
        from src.rag.readiness import EASIER_QUESTIONS, QUESTION_VARIANTS

        assert len(QUESTION_VARIANTS["pixel_pitch"]["en"]) >= 3
        assert len(QUESTION_VARIANTS["pixel_pitch"]["zh"]) >= 3
        assert EASIER_QUESTIONS["pixel_pitch"]["en"], "第二问（答非所问时）也要有降门槛说法"


class TestAfterRecommendationMustAnswer:
    """实测反馈：推荐完产品后，客户接着问什么，AI 都又推荐一遍，接不住话。

    根因：`has_scenario` 用的是"历史里已经收集到的 usage"，
    于是场景一旦确定，之后**每一句话**都被当成"在报需求"→ need_query →
    Gate 已就绪 → 又推荐一遍（客户问代理商、付款、认证都得到同一份推荐）。

    现在：只看**本轮这句话**有没有需求信息，并且提问（？/吗/有没有…）永远走回答路径；
    已经推荐过、本轮又没有新需求时，不再重复推荐。
    """

    def _ready_profile(self):
        slots = {
            "display_type": "LED",
            "environment": "indoor",
            "purpose": "conference",
            "content_type": "mixed",
            "installation": "fixed",
            "price_preference": "price",
            "viewing_distance_m": 5,
            "target_width_mm": 3000,
            "target_height_mm": 5000,
        }
        return RequirementProfile.from_slots(slots, explicit_keys=set(slots))

    @pytest.fixture
    def no_usage_llm(self, monkeypatch):
        """假 LLM：不返回 usage（模拟真实 LLM 对"纯提问"返回 null 的场景）。"""
        import sys

        sales_req = sys.modules["src.agents.sales.nodes.requirement"]

        class _Response:
            content = '{"usage": null, "additional_requirements": [], "ack": ""}'

        class _FakeChat:
            def __init__(self, *args, **kwargs):
                pass

            def invoke(self, *args, **kwargs):
                return _Response()

        monkeypatch.setattr(sales_req, "ChatOpenAI", _FakeChat)
        return sales_req

    def _state(self, message, intent="others", already_recommended=True, profile=None):
        return {
            "messages": [{"role": "user", "content": message}],
            "current_message": message,
            "session_id": "after-recommendation",
            "requirements": {"usage": "会议室", "location_type": "室内"},
            "additional_requirements": [],
            "intent": intent,
            "next_action": "ask",
            "should_generate_solution": False,
            "response": "",
            "pending_question": "",
            "pending_slot": "",
            "already_recommended": already_recommended,
            "requirement_profile": profile or self._ready_profile(),
        }

    def test_agent_question_is_not_re_recommended(self, no_usage_llm):
        state = self._state("你们在肯尼亚有代理商吗？")
        out = no_usage_llm.requirement_mining(state)

        assert out["should_generate_solution"] is False, "客户在提问，不该又推荐一遍"
        assert out["intent"] == "others", "提问必须留给回答问题的路径"

    def test_question_is_not_reclassified_as_requirement_answer(self, no_usage_llm):
        """带问号的句子不能被当成"在报需求"（哪怕历史里已经有 usage）。"""
        state = self._state("你们在肯尼亚有代理商吗？")
        out = no_usage_llm.requirement_mining(state)
        assert out["intent"] != "need_query"

    def test_company_question_gets_company_answer(self, no_usage_llm):
        """端到端（节点级）：代理商问题必须由公司信息回答，含"自有工厂 + 成本"。"""
        from src.agents.sales.nodes.script_generator import script_generator

        state = self._state("你们在肯尼亚有代理商吗？")
        state = no_usage_llm.requirement_mining(state)
        state = script_generator(state)

        assert state["next_action"] == "ask"
        answer = state["response"]
        lowered = answer.lower()
        # 语言跟随系统策略（默认英文），只校验内容要点：只有中国一个点 + 自有工厂 + 成本更低
        assert "factory" in lowered or "工厂" in answer, answer
        assert any(
            word in lowered for word in ("cost", "price", "middleman")
        ) or any(word in answer for word in ("成本", "开销", "中间")), answer
        assert "shenzhen" in lowered or "深圳" in answer, answer

    def test_product_question_after_recommendation_is_answered(self, no_usage_llm):
        state = self._state("what is the pixel pitch of that model?", intent="product_question")
        out = no_usage_llm.requirement_mining(state)

        assert out["should_generate_solution"] is False
        assert out["intent"] == "product_question"

    def test_new_requirements_still_trigger_a_new_recommendation(self, sales_llm):
        """客户改了需求（这一轮给了新信息）→ 必须重新推荐。"""
        state = self._state("actually make it outdoor instead")
        out = sales_llm.requirement_mining(state)

        assert out["requirement_profile"].environment == "outdoor"
        assert out["should_generate_solution"] is True

    def test_explicit_request_still_recommends(self, sales_llm):
        state = self._state("can you recommend another model?", intent="product_question")
        out = sales_llm.requirement_mining(state)
        assert out["should_generate_solution"] is True

    def test_first_recommendation_is_not_blocked(self, sales_llm):
        """没推荐过的时候照旧：Gate 就绪就推荐。"""
        state = self._state(
            "I need an indoor fixed LED screen for a conference room",
            intent="need_query",
            already_recommended=False,
        )
        out = sales_llm.requirement_mining(state)
        assert out["should_generate_solution"] is True


class TestReplayOfReportedConversation:
    """完整复刻客户日志的那一轮对话（图片 → permanent → i dont know → close）。

    旧行为：最后一轮 ="close" 被判成 closing → 不推荐、只回"我会准备报价"。
    新行为：close 被理解成"近"（约 3m）→ Gate 放行 → 进入推荐（并会继续问尺寸）。
    """

    @pytest.fixture
    def replay_llm(self, monkeypatch):
        import sys

        sales_req = sys.modules["src.agents.sales.nodes.requirement"]
        classify_mod = sys.modules["src.agents.sales.nodes.classify"]
        extractor_mod = sys.modules["src.core.requirement_extractor"]

        class _Response:
            def __init__(self, content):
                self.content = content

        class _Fake:
            def __init__(self, *args, **kwargs):
                pass

            def invoke(self, messages, *args, **kwargs):
                parts = messages if isinstance(messages, list) else [messages]
                system = str(getattr(parts[0], "content", ""))
                user = str(getattr(parts[-1], "content", ""))
                if "意图分类器" in system:
                    # 复刻实测：DeepSeek 把 "close" 判成了 closing
                    return _Response("closing" if user.strip().lower() == "close" else "need_query")
                return _Response('{"usage": "conference", "additional_requirements": [], "ack": ""}')

        monkeypatch.setattr(sales_req, "ChatOpenAI", _Fake)
        monkeypatch.setattr(classify_mod, "ChatOpenAI", _Fake)
        monkeypatch.setattr(
            extractor_mod.RequirementExtractor,
            "_llm_semantic_extract",
            lambda self, message, rule_slots, session_id="": {},
        )
        extractor_mod.RequirementExtractor._semantic_cache.clear()
        return sales_req

    def _turn(self, module, message, profile):
        state = {
            "messages": [{"role": "user", "content": message}],
            "current_message": message,
            "session_id": "replay-close",
            "requirements": {},
            "additional_requirements": [],
            "intent": "need_query",
            "next_action": "ask",
            "should_generate_solution": False,
            "response": "",
            "pending_question": "",
            "pending_slot": "",
        }
        if profile is not None:
            state["requirement_profile"] = profile
        return module.requirement_mining(state)

    def test_final_turn_recommends_and_keeps_distance(self, replay_llm, monkeypatch):
        import sys

        classify_mod = sys.modules["src.agents.sales.nodes.classify"]
        sales_req = replay_llm

        # 第 1 轮：客户发图 + "i need a display like this"
        # （图片结果由视觉模块合并进档案：LED / indoor / conference）
        # 另外照新流程把"内容类型 / 价格取向"也当作已确认（这两项不影响本用例要看的行为）
        vision_slots = {
            "display_type": "LED", "environment": "indoor", "purpose": "conference",
            "content_type": "mixed", "price_preference": "price",
            # 硬性条件里的尺寸已经确定（客户给出的目标尺寸）
            "target_width_mm": 5000, "target_height_mm": 3000,
        }
        profile = RequirementProfile.from_slots(vision_slots, explicit_keys=set(vision_slots))

        turn = self._turn(sales_req, "i need a display like this", profile)
        assert turn["pending_slot"] == "installation"

        turn = self._turn(sales_req, "permanent", turn["requirement_profile"])
        # 客户口径：安装方式之后**先问点间距**
        assert turn["pending_slot"] == "pixel_pitch"

        # 点间距客户不知道 → 只问一次就跳过，转问观看距离
        turn = self._turn(sales_req, "i dont know", turn["requirement_profile"])
        assert turn["recommendation_gate"]["status"] == "CONTINUE_ASKING"
        assert turn["pending_slot"] == "viewing_distance"
        assert turn["requirement_profile"].ask_count("pixel_pitch") == 1

        # 最后一轮："close" —— 之前这里被判成 closing，导致既不推荐也不问尺寸
        intent = classify_mod.classify(
            {"current_message": "close", "intent": "", "next_action": ""}
        )
        assert intent["intent"] == "need_query"

        state = {
            "messages": [{"role": "user", "content": "close"}],
            "current_message": "close",
            "session_id": "replay-close",
            "requirements": {},
            "additional_requirements": [],
            "intent": intent["intent"],
            "next_action": intent["next_action"],
            "should_generate_solution": False,
            "response": "",
            "pending_question": "",
            "pending_slot": "",
            "requirement_profile": turn["requirement_profile"],
        }
        final = sales_req.requirement_mining(state)

        assert final["requirement_profile"].viewing_distance_m == 3.0, "close 应被理解为「近」"
        assert final["recommendation_gate"]["status"] == "READY"
        assert final["should_generate_solution"] is True, "这一轮必须进入推荐"


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
            "messages": [{"role": "user", "content": "室内会议室，5米x3米，固定安装，5米视距"}],
            "current_message": "室内会议室，5米x3米，固定安装，5米视距",
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


class TestObviousSceneSettlesEnvironment:
    """实测反馈：客户说了 "It for church"，系统还在追问"室内还是室外"。

    规则：会议室 / 教室 / 教堂 / 展厅 / 机场… = 室内，户外广告 / 体育场 = 室外，
    这类"一眼就能定"的场景直接把环境**填好**（scenario_derived）；但因为这是
    系统推断的，客户口径（2026-09-17）要求**再主动确认一次**，问过一次就放行；
    舞台 / 演唱会 / 租赁这类室内外都可能的场景仍然要问。
    """

    def _fake_llm(self, monkeypatch, usage: str):
        import src.agents.sales.nodes.requirement as sales_req

        class _Response:
            content = '{"usage": "%s", "additional_requirements": [], "ack": ""}' % usage

        class _FakeChat:
            def __init__(self, *args, **kwargs):
                pass

            def invoke(self, *args, **kwargs):
                return _Response()

        monkeypatch.setattr(sales_req, "ChatOpenAI", _FakeChat)
        return sales_req

    def _turn(self, sales_req, message, requirements=None):
        state = {
            "messages": [
                {"role": "user", "content": "i need a led display"},
                {"role": "assistant", "content": "Which display type do you need?"},
                {"role": "user", "content": message},
            ],
            "current_message": message,
            "session_id": "obvious-scene-session",
            "requirements": dict(requirements or {}),
            "additional_requirements": [],
            "intent": "industry",
            "next_action": "ask",
            "should_generate_solution": False,
            "response": "",
        }
        return sales_req.requirement_mining(state)

    def test_church_fills_indoor_without_asking_environment(self, monkeypatch):
        """客户口径（2026-09-18）：明显的室内/室外场景**直接确定环境，不再问室内外**。

        客户只说了 church（原话里也没写 indoor）→ 环境按场景判定为 indoor，
        下一问直接进入硬性条件（固装/租赁），不会再问"室内还是室外"。
        """
        sales_req = self._fake_llm(monkeypatch, "church")
        result = self._turn(
            sales_req, "It for church", {"display_type": "LED", "size": "10米x5米"}
        )

        profile = result["requirement_profile"]
        assert profile.environment == "indoor"
        assert profile.sources.get("environment") == "scenario_derived"
        assert result["pending_slot"] != "environment", "明显场景不再问室内外"
        assert "indoor or outdoor" not in (result["pending_question"] or "").lower()
        assert result["pending_slot"] == "installation"

    def test_outdoor_advertising_fills_outdoor_without_asking(self, monkeypatch):
        sales_req = self._fake_llm(monkeypatch, "outdoor advertising")
        # 注意：客户这轮没有直接说 "outdoor" 这个词，环境是系统从场景推断的
        result = self._turn(sales_req, "advertising screen", {"display_type": "LED"})

        profile = result["requirement_profile"]
        assert profile.environment == "outdoor"
        assert profile.sources.get("environment") == "scenario_derived"
        assert result["pending_slot"] != "environment", "明显场景不再问室内外"

    def test_stage_still_asks_environment(self, monkeypatch):
        """舞台 / 演唱会室内外都可能 → 必须继续问。"""
        sales_req = self._fake_llm(monkeypatch, "stage performance")
        result = self._turn(sales_req, "stage performance screen", {"display_type": "LED"})

        assert result["pending_slot"] == "environment"
        assert result["requirement_profile"].environment is None

    def test_ai_judged_scene_settles_environment(self):
        """客户口径：不要只锁定几个场景词，要让 AI 从客户原话判断"明显室内/室外"。

        AI 判断必须带客户原话证据，且结果记为 scenario_derived（不再问客户室内外）。
        """
        from src.core.requirement_extractor import get_requirement_extractor

        extractor = get_requirement_extractor()
        # 注意：这里刻意用一个"不在固定场景表里"的场地（珠宝店精品廊），
        # 这样才是在验证"AI 从客户原话判断"，而不是命中关键词表。
        message = "we need a display for our jewellery boutique"
        profile = extractor.extract(
            message,
            semantic_override={
                "purpose": "other",
                "environment_implied_by_scene": "indoor",
                "environment_implied_evidence": "jewellery boutique",
            },
            use_llm=False,
            session_id="",
        )
        assert profile.environment == "indoor"
        assert profile.sources["environment"] == "scenario_derived"
        assert "environment" not in check_recommendation_ready(profile).missing

        # 没有原话证据 → 不能凭 AI 猜就落定环境
        profile2 = extractor.extract(
            message,
            semantic_override={
                "purpose": "other",
                "environment_implied_by_scene": "indoor",
                "environment_implied_evidence": "totally different words",
            },
            use_llm=False,
            session_id="",
        )
        assert profile2.environment is None
        assert "environment" in check_recommendation_ready(profile2).missing


class TestPriceQuestionWhileCollecting:
    """实测日志：客户问价格（intent=objection），系统回了 "Sure, let me find the right
    products for you..."，既没回答价格、又被旧判定拖去推荐（0 个产品）。

    正确行为：说明"需要先确认产品才能报价"，紧接着继续问需求，且绝不触发推荐。
    """

    MESSAGE = "What is the price of smd screen wedth 2.5 feet and 1.2m"

    def _state(self, message=None, **overrides):
        state = {
            "messages": [{"role": "user", "content": message or self.MESSAGE}],
            "current_message": message or self.MESSAGE,
            "session_id": "price-session",
            "intent": "objection",
            "next_action": "ask",
            "requirements": {
                "usage": "showroom",
                "location_type": "室内",
                "indoor": True,
                "display_type": "LED",
                "size": "1.2192米宽",
            },
            "additional_requirements": [],
            "should_generate_solution": False,
            "response": "",
            "pending_question": "Should I quote this as a fixed install or as a rental solution?",
            "pending_slot": "installation",
            "suppress_greeting": False,
        }
        state.update(overrides)
        return state

    def test_price_question_answers_policy_then_keeps_collecting(self):
        from src.agents.sales.nodes.script_generator import script_generator

        result = script_generator(self._state())

        assert result["next_action"] == "ask"
        assert result["should_generate_solution"] is False
        # 先说明"要确认产品才能报价"
        assert "quot" in result["response"].lower()
        # 紧接着继续问需求（同一个待问项；引导语后面的问句首字母会被小写化）
        assert result["response"].rstrip().lower().endswith(
            result["pending_question"].lower()
        )

    def test_price_question_does_not_trigger_solution_on_usage(self):
        """回归：旧版这里会因为 requirements 里有 usage 就直接 trigger_solution。"""
        from src.agents.sales.nodes.script_generator import script_generator

        result = script_generator(self._state())
        assert result["next_action"] != "trigger_solution"
        assert result["response"] != "Sure, let me find the right products for you..."

    def test_price_question_still_triggers_when_gate_ready(self):
        from src.agents.sales.nodes.script_generator import script_generator

        result = script_generator(self._state(should_generate_solution=True))
        assert result["next_action"] == "trigger_solution"

    def test_need_query_does_not_trigger_on_usage_alone(self):
        """回归：Gate 未就绪时，仅凭 usage 不得触发推荐。"""
        from src.agents.sales.nodes.script_generator import script_generator

        result = script_generator(
            self._state(
                message="we need a screen for the showroom",
                intent="need_query",
                pending_question="",
                pending_slot="",
            )
        )
        assert result["next_action"] == "ask"
        assert result["should_generate_solution"] is False
        assert result["response"]
