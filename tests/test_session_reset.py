"""
会话内需求重置测试：客户拿到推荐后要换产品 / 换项目 / 改需求时，
系统必须清空旧需求、重新采集，而不是拿旧需求再次推荐。

覆盖：
  1. 纯规则检测器（中英文、误伤护栏）
  2. MemoryStore 的推荐状态与重置接口
  3. SalesAgentRunner：重置后清空需求 + 追加口语确认
  4. Sales Agent graph：重置轮必须回到"问下一个关键问题"
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.session_switch import (  # noqa: E402
    RESET_ACKS,
    detect_requirement_reset,
    find_requirement_conflicts,
    reset_acknowledgement,
)


def _profile(**slots) -> RequirementProfile:
    """构造"客户明确说过"的需求档案。"""
    return RequirementProfile.from_slots(slots, explicit_keys=set(slots))


CHURCH_REQUIREMENTS = {
    "display_type": "LED",
    "location_type": "室内",
    "indoor": True,
    "outdoor": False,
    "usage": "church",
    "distance": "5米",
}


class TestExplicitResetDetection:
    """客户明确说"重新来 / 换产品" → 全量重置。"""

    @pytest.mark.parametrize("message", [
        "我想换个产品",
        "换一款看看",
        "换成别的型号",
        "重新推荐一下",
        "重新来",
        "重来一遍",
        "不要这个了，再来一个方案",
        "我想要另外一款",
        "请重新帮我梳理需求",
        "给我推荐别的",
        "我要别的产品",
    ])
    def test_chinese_reset_phrases(self, message):
        decision = detect_requirement_reset(
            message, requirements=CHURCH_REQUIREMENTS, recommended=True
        )
        assert decision.should_reset, message
        assert decision.reason == "explicit_request"

    @pytest.mark.parametrize("message", [
        "I want a different product",
        "can we start over",
        "please reset",
        "I changed my mind",
        "switch to another model",
        "show me another one",
    ])
    def test_english_reset_phrases(self, message):
        decision = detect_requirement_reset(
            message, requirements=CHURCH_REQUIREMENTS, recommended=True
        )
        assert decision.should_reset, message

    def test_reset_without_context_does_nothing(self):
        """"重新来"在空会话里没有东西可清空 → 不重置。"""
        decision = detect_requirement_reset("重新来", requirements={}, recommended=False)
        assert not decision.should_reset

    @pytest.mark.parametrize("message", [
        "还有别的型号吗",
        "有没有其他推荐",
        "再推荐几款看看",
        "有别的推荐吗",
        "有别的产品吗",
        "P2.5 的亮度是多少",
        "能重新计算一下箱体吗",
        "any other options?",
        "more models please",
    ])
    def test_no_false_reset(self, message):
        """想看更多选项 / 参数提问，不是换需求 → 不清空。"""
        decision = detect_requirement_reset(
            message, requirements=CHURCH_REQUIREMENTS, recommended=True
        )
        assert not decision.should_reset, message


class TestConflictDetection:
    """已有推荐后，客户给出冲突的环境或场景 → 需要重置。"""

    def test_environment_conflict_after_recommendation(self):
        profile = _profile(environment="indoor", purpose="conference")
        assert find_requirement_conflicts("actually it will be outdoors", profile) == ["environment"]
        decision = detect_requirement_reset(
            "actually it will be outdoors",
            requirements={"location_type": "室内"},
            profile=profile,
            recommended=True,
        )
        assert decision.should_reset
        assert decision.reason == "requirement_conflict"
        assert "environment" in decision.conflicts

    def test_purpose_conflict_after_recommendation(self):
        profile = _profile(environment="indoor", purpose="conference")
        decision = detect_requirement_reset(
            "其实是给教堂用的",
            requirements={"usage": "会议室"},
            profile=profile,
            recommended=True,
        )
        assert decision.should_reset
        assert "purpose" in decision.conflicts

    def test_conflict_before_recommendation_just_updates_slot(self):
        """还没推荐过时，客户只是纠正一个事实 → 不清空，只更新槽位。"""
        profile = _profile(environment="indoor", purpose="conference")
        decision = detect_requirement_reset(
            "actually it will be outdoors",
            requirements={"location_type": "室内"},
            profile=profile,
            recommended=False,
        )
        assert not decision.should_reset

    def test_same_purpose_is_not_conflict(self):
        profile = _profile(environment="indoor", purpose="church")
        assert find_requirement_conflicts("church", profile) == []

    def test_size_or_pitch_change_is_not_conflict(self):
        """尺寸 / 点间距改动只需要更新槽位，不该清空重来。"""
        profile = _profile(
            environment="indoor", purpose="conference",
            viewing_distance_m=5, target_width_mm=5000, target_height_mm=3000,
        )
        assert find_requirement_conflicts("尺寸改成 8m x 4m", profile) == []
        assert find_requirement_conflicts("换成 P3 的", profile) == []


class TestDisplayTypeSwitch:

    def test_display_type_switch_resets(self):
        decision = detect_requirement_reset(
            "we prefer LCD now",
            requirements=CHURCH_REQUIREMENTS,
            recommended=True,
            display_type_change=("LED", "LCD"),
        )
        assert decision.should_reset
        assert decision.reason == "display_type_switch"


class TestAcknowledgementVariants:

    def test_multiple_phrasings(self):
        zh = {reset_acknowledgement("zh", seed) for seed in range(6)}
        en = {reset_acknowledgement("en", seed) for seed in range(6)}
        assert len(zh) >= 2 and len(en) >= 2
        assert all(text.strip() for text in zh | en)


class TestMemoryRecommendationState:

    def test_mark_and_reset(self):
        from src.memory.store import memory

        session_id = "reset-memory-test"
        memory.clear(session_id)
        try:
            memory.set_requirements(session_id, dict(CHURCH_REQUIREMENTS))
            memory.set_requirement_profile(
                session_id, _profile(environment="indoor", purpose="church")
            )
            assert not memory.has_recommendation(session_id)

            memory.mark_recommendation_done(session_id, [{"model": "TW11-3216-P2.5"}])
            assert memory.has_recommendation(session_id)
            assert memory.get_recommendation(session_id)["models"] == ["TW11-3216-P2.5"]

            memory.reset_requirement_state(session_id)
            assert memory.get_requirements(session_id) == {}
            assert memory.get_requirement_profile(session_id) is None
            assert not memory.has_recommendation(session_id)
        finally:
            memory.clear(session_id)


class _StubGraph:
    """返回固定结果的假 graph，用于单测 runner 的重置 / 持久化行为。"""

    def __init__(
        self,
        response="Will the screen be indoors or outdoors?",
        next_action="ask",
        solutions=None,
    ):
        self.response = response
        self.next_action = next_action
        self.solutions = list(solutions or [])
        self.seen_state = None

    def invoke(self, state):
        self.seen_state = dict(state)
        return {
            **state,
            "response": self.response,
            "requirements": dict(state.get("requirements") or {}),
            "solutions": list(self.solutions),
            "next_action": self.next_action,
        }


class TestRunnerClearsRequirements:

    def _runner(self, graph):
        from src.agents.sales.runner import SalesAgentRunner
        from src.memory.store import memory

        runner = SalesAgentRunner(sales_search=None, solution_runner=None, memory_store=memory)
        runner.graph = graph  # 跳过 setup()，直接注入假 graph
        return runner

    def test_switch_product_clears_requirements(self):
        from src.memory.store import memory

        session_id = "reset-runner-test"
        memory.clear(session_id)
        try:
            memory.add(session_id, "user", "indoor LED for a church")
            memory.add(session_id, "assistant", "TW11-3216-P2.5 is our recommendation ...")
            memory.set_requirements(session_id, dict(CHURCH_REQUIREMENTS))
            memory.mark_recommendation_done(session_id, [{"model": "TW11-3216-P2.5"}])

            graph = _StubGraph()
            runner = self._runner(graph)
            result = runner.run(session_id, "I want a different product")

            # 旧需求必须被清空，并写回 memory
            assert graph.seen_state["requirements"] == {}
            assert graph.seen_state["requirements_reset"] is True
            assert memory.get_requirements(session_id) == {}
            assert not memory.has_recommendation(session_id)
            # 回复要有"重新来"的确认
            assert any(result["response"].startswith(ack) for ack in RESET_ACKS["en"])
            assert graph.response in result["response"]
        finally:
            memory.clear(session_id)

    def test_normal_turn_keeps_requirements(self):
        from src.memory.store import memory

        session_id = "reset-runner-keep-test"
        memory.clear(session_id)
        try:
            memory.set_requirements(session_id, dict(CHURCH_REQUIREMENTS))

            graph = _StubGraph(response="Could you tell me the target screen size?")
            runner = self._runner(graph)
            result = runner.run(session_id, "5m x 3m")

            assert graph.seen_state["requirements"] == CHURCH_REQUIREMENTS
            assert graph.seen_state["requirements_reset"] is False
            assert result["response"] == "Could you tell me the target screen size?"
        finally:
            memory.clear(session_id)

    def test_empty_trigger_does_not_mark_recommendation(self):
        """回归：next_action=trigger_solution 但没给出产品时，不能标记"已推荐"。

        （实测日志里出现过 products=0 却 Marked recommendation delivered，
        会让下一轮客户说"换个产品"时被误判成"看完推荐要换"。）
        """
        from src.memory.store import memory

        session_id = "reset-empty-trigger-test"
        memory.clear(session_id)
        try:
            runner = self._runner(
                _StubGraph(response="Sure, let me find the right products for you...",
                           next_action="trigger_solution", solutions=[])
            )
            runner.run(session_id, "I need a screen")
            assert not memory.has_recommendation(session_id)

            # 真的给出了产品才算
            runner2 = self._runner(
                _StubGraph(response="TW11-3216-P2.5 is our recommendation.",
                           next_action="trigger_solution",
                           solutions=[{"model": "TW11-3216-P2.5"}])
            )
            runner2.run(session_id, "indoor conference room, 5m, fixed")
            assert memory.has_recommendation(session_id)
        finally:
            memory.clear(session_id)


class TestGraphAsksAgainAfterReset:
    """重置轮必须回到需求采集：不得走 others / product_question 的自由问答。"""

    @pytest.fixture
    def fake_llm(self, monkeypatch):
        # 注意：包的 __init__ 里 `classify` 这个名字被节点函数占用了，
        # 所以必须从 sys.modules 取子模块本体。
        import sys

        classify_mod = sys.modules["src.agents.sales.nodes.classify"]
        sales_req = sys.modules["src.agents.sales.nodes.requirement"]

        class _ClassifyResponse:
            content = "others"

        class _RequirementResponse:
            content = '{"usage": null, "additional_requirements": []}'

        class _FakeClassify:
            def __init__(self, *args, **kwargs):
                pass

            def invoke(self, *args, **kwargs):
                return _ClassifyResponse()

        class _FakeRequirement:
            def __init__(self, *args, **kwargs):
                pass

            def invoke(self, *args, **kwargs):
                return _RequirementResponse()

        monkeypatch.setattr(classify_mod, "ChatOpenAI", _FakeClassify)
        monkeypatch.setattr(sales_req, "ChatOpenAI", _FakeRequirement)

    def test_reset_turn_asks_first_question_again(self, fake_llm):
        from src.agents.sales.graph import build_sales_graph

        graph = build_sales_graph()
        state = {
            "messages": [
                {"role": "user", "content": "indoor LED for a church"},
                {"role": "assistant", "content": "TW11-3216-P2.5 is our recommendation"},
                {"role": "user", "content": "I want a different product"},
            ],
            "current_message": "I want a different product",
            "session_id": "reset-graph-test",
            "intent": "",
            "requirements": {},
            "additional_requirements": [],
            "required_met": False,
            "required_missing": [],
            "should_generate_solution": False,
            "solutions": [],
            "response": "",
            "next_action": "ask",
            "turn_count": 0,
            "suppress_greeting": False,
            "requirement_profile": None,
            "pending_question": "",
            "requirements_reset": True,
            "reset_reason": "explicit_request",
        }
        result = graph.invoke(state)

        assert result["should_generate_solution"] is False
        assert result["next_action"] == "ask"
        assert result["intent"] == "need_query"
        assert result["pending_question"], "重置后必须重新追问一个关键问题"
        assert result["response"] == result["pending_question"]
        assert result["solutions"] == []

    def test_reset_turn_reuses_new_message_facts(self, fake_llm):
        """重置的同时客户给了新信息（户外）→ 只问还缺的那一项。"""
        from src.agents.sales.graph import build_sales_graph
        from src.rag.readiness import check_recommendation_ready

        graph = build_sales_graph()
        state = {
            "messages": [
                {"role": "assistant", "content": "TW11-3216-P2.5 is our recommendation"},
                {"role": "user", "content": "actually I want an outdoor one"},
            ],
            "current_message": "actually I want an outdoor one",
            "session_id": "reset-graph-facts-test",
            "intent": "",
            "requirements": {},
            "additional_requirements": [],
            "required_met": False,
            "required_missing": [],
            "should_generate_solution": False,
            "solutions": [],
            "response": "",
            "next_action": "ask",
            "turn_count": 0,
            "suppress_greeting": False,
            "requirement_profile": None,
            "pending_question": "",
            "requirements_reset": True,
            "reset_reason": "explicit_request",
        }
        result = graph.invoke(state)

        profile = result["requirement_profile"]
        assert profile.environment == "outdoor"
        assert "purpose" in check_recommendation_ready(profile).missing
        assert result["should_generate_solution"] is False


class TestMultiTurnSwitchProductEndToEnd:
    """端到端多轮：客户拿到推荐后在同一会话里换产品 → 清空 + 重新采集。"""

    @pytest.fixture
    def fake_llm(self, monkeypatch):
        import sys

        classify_mod = sys.modules["src.agents.sales.nodes.classify"]
        sales_req = sys.modules["src.agents.sales.nodes.requirement"]

        class _Response:
            def __init__(self, content):
                self.content = content

        class _Fake:
            """按 system prompt 判断当前是意图分类还是需求抽取。"""

            def __init__(self, *args, **kwargs):
                pass

            def invoke(self, messages, *args, **kwargs):
                system = str(getattr(messages[0], "content", ""))
                if "意图分类器" in system:
                    return _Response("others")
                return _Response('{"usage": null, "additional_requirements": []}')

        monkeypatch.setattr(classify_mod, "ChatOpenAI", _Fake)
        monkeypatch.setattr(sales_req, "ChatOpenAI", _Fake)

    def _seed_recommended_session(self, memory, session_id):
        memory.add(session_id, "user", "indoor LED display for a church")
        memory.add(session_id, "assistant", "TW11-3216-P2.5 is our recommendation for you.")
        memory.set_requirements(session_id, dict(CHURCH_REQUIREMENTS))
        memory.set_requirement_profile(
            session_id, _profile(environment="indoor", purpose="church", viewing_distance_m=5)
        )
        memory.mark_recommendation_done(session_id, [{"model": "TW11-3216-P2.5"}])

    def test_switch_product_resets_and_reasks(self, fake_llm):
        from src.agents.sales.runner import SalesAgentRunner
        from src.memory.store import memory

        session_id = "e2e-switch-product"
        memory.clear(session_id)
        try:
            self._seed_recommended_session(memory, session_id)
            runner = SalesAgentRunner(memory_store=memory)
            runner.setup()

            result = runner.run(session_id, "I want a different product")

            assert result["products"] == []
            assert result["next_action"] == "ask"
            assert any(result["response"].startswith(ack) for ack in RESET_ACKS["en"])
            assert result["response"].rstrip().endswith("?")
            # 旧需求（含教堂 / 5 米 / 室内）必须已被清空
            assert memory.get_requirements(session_id) == {}
            # 档案里不允许留下任何旧的需求事实（只允许保留"这一轮刚问过什么"的
            # 提问记账 —— 那是"同一字段最多问两次"状态机跨轮必需的状态）。
            persisted = memory.get_requirement_profile(session_id)
            if persisted is not None:
                bookkeeping = ("sources", "ask_counts", "unknown_reasons", "last_asked_slot", "conflicts")
                facts = {k: v for k, v in persisted.items() if k not in bookkeeping}
                assert all(v in (None, "", [], {}) for v in facts.values()), facts
            assert not memory.has_recommendation(session_id)
        finally:
            memory.clear(session_id)

    def test_after_reset_system_collects_from_scratch(self, fake_llm):
        """重置后的下一轮：客户回答"室外"，系统继续问场景，而不是直接推荐。"""
        from src.agents.sales.runner import SalesAgentRunner
        from src.memory.store import memory

        session_id = "e2e-switch-product-2"
        memory.clear(session_id)
        try:
            self._seed_recommended_session(memory, session_id)
            runner = SalesAgentRunner(memory_store=memory)
            runner.setup()

            runner.run(session_id, "I want a different product")
            result = runner.run(session_id, "outdoor")

            requirements = memory.get_requirements(session_id)
            assert requirements.get("display_type") == "LED"
            assert requirements.get("outdoor") is True
            assert "church" not in str(requirements.get("usage") or "")
            assert result["products"] == []
            assert result["next_action"] == "ask"
        finally:
            memory.clear(session_id)
