"""
"先回应客户 + 再追问需求" 组合回复测试。

对应客户实测反馈：
  - 客户问 "Do u have P 1.2 COB Led" → 不能只回一句反问，要先正面回答
  - 客户问 "can I get ur representative in Indonesia" → 答完要接着问需求
  - 追问话术不能每轮都一样
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rag.reply_composer import (  # noqa: E402
    acknowledge,
    availability_answer,
    compose_requirement_reply,
    is_price_question,
    price_policy_answer,
    requirement_echo,
)
from src.rag.readiness import question_for  # noqa: E402


class TestAvailabilityAnswer:
    """客户问"有没有某规格" → 用产品数据核实后正面回答。"""

    def test_p12_cob_is_confirmed(self):
        answer = availability_answer("Do u have  P 1.2 COB Led")
        assert answer and answer.lower().startswith("yes")
        assert "1.2" in answer and "COB" in answer

    def test_p25_led_is_confirmed(self):
        answer = availability_answer("do you have a P2.5 LED display")
        assert answer and "2.5" in answer

    def test_named_series_is_confirmed(self):
        answer = availability_answer("do you have TW31-COB")
        assert answer and "TW31-COB" in answer

    def test_unknown_spec_is_not_fabricated(self):
        """目录里没有的规格不能硬说"有"。"""
        answer = availability_answer("do you have a P0.5 COB LED")
        assert answer is not None
        assert not answer.lower().startswith("yes")

    def test_non_availability_question_returns_none(self):
        assert availability_answer("can i get ur representative at Indonesia. Thanks") is None
        assert availability_answer("what's your warranty?") is None


class TestRequirementEcho:

    def test_echoes_scenario(self):
        assert "church" in (requirement_echo("church") or "")

    def test_echoes_distance(self):
        echo = requirement_echo("about 5m") or ""
        assert "5" in echo and "distance" in echo

    def test_echoes_scenario_and_distance(self):
        echo = requirement_echo("indoor conference room, 5m viewing distance") or ""
        assert "conference" in echo and "5" in echo

    def test_nothing_to_echo(self):
        assert requirement_echo("can i get ur representative at Indonesia") is None


class TestAcknowledgeVariety:

    def test_phrasing_varies_by_seed(self):
        texts = {acknowledge("church", seed=seed) for seed in range(8)}
        assert len(texts) >= 3, texts
        assert all(text for text in texts)

    def test_question_variants_are_plentiful(self):
        """追问话术池要够大，避免每轮都是同一句。"""
        for slot in ("environment", "purpose", "installation", "viewing_distance", "size"):
            texts = {question_for(slot, "en", seed) for seed in range(12)}
            assert len(texts) >= 4, (slot, texts)


class TestLlmAcknowledgement:
    """销售必须先回应客户这句话（自我介绍 / 提问 / 要报价），而不是只追问。"""

    def test_llm_ack_is_used(self):
        ack = acknowledge(
            "My name is Ar Majeed Akbar from Pakistan, we need a smart screen for our university",
            llm_ack="Nice to meet you, Ar Majeed — happy to help with the smart classroom setup.",
        )
        assert ack.startswith("Nice to meet you")

    def test_verified_availability_beats_llm_ack(self):
        """"有没有 P1.2 COB" 属于可核实事实，优先用产品数据回答。"""
        ack = acknowledge("do you have P1.2 COB LED", llm_ack="Let me think about it.")
        assert ack.lower().startswith("yes")

    def test_llm_ack_with_question_is_dropped(self):
        ack = acknowledge("church", llm_ack="Is this indoors or outdoors?")
        assert "?" not in ack and "？" not in ack
        # 退回规则复述
        assert "church" in ack

    def test_llm_ack_cleaned(self):
        ack = acknowledge(
            "thanks for the details",
            llm_ack='  "Understood,\nI will prepare the quotation."  ',
        )
        assert ack == "Understood, I will prepare the quotation."

    def test_generic_fallback_never_empty(self):
        """没有任何可回应内容时也不能空着 —— 否则又变成"只追问"。"""
        ack = acknowledge("87")
        assert ack and len(ack) > 2

    def test_chinese_ack_is_dropped_under_english_policy(self):
        """实测 bug：回复出现 "Sello 你好，很高兴认识你。Will it be an indoor or outdoor
        setup?" —— 中文回应 + 英文追问。英文策略下必须丢弃中文回应。"""
        ack = acknowledge("LED screen", llm_ack="Sello 你好，很高兴认识你。")
        assert ack
        assert not any("\u4e00" <= ch <= "\u9fff" for ch in ack), ack

    def test_chinese_ack_kept_under_chinese_language(self):
        ack = acknowledge("LED screen", llm_ack="Sello 你好，很高兴认识你。", language="zh")
        assert "你好" in ack

    def test_compose_uses_llm_ack_before_echo(self):
        text = compose_requirement_reply(
            question="Is the installation going to be indoors or outdoors?",
            slot="environment",
            message="we are going for smart class room in our university",
            language="en",
            llm_ack="Happy to help with your smart classroom project.",
        )
        assert text.startswith("Happy to help with your smart classroom project.")
        assert text.endswith("Is the installation going to be indoors or outdoors?")


class TestPriceQuestion:
    """需求还没问清时客户问价格：先说"要确认产品才能报价"，紧接着继续问需求。"""

    @pytest.mark.parametrize("message", [
        "What is the price of smd screen wedth 2.5 feet",
        "how much does it cost?",
        "can you send me a quotation",
        "这个多少钱",
        "价格能便宜点吗",
    ])
    def test_price_question_detected(self, message):
        assert is_price_question(message), message

    @pytest.mark.parametrize("message", [
        "indoor conference room",
        "do you have P1.2 COB LED",
        "fixed installation",
    ])
    def test_non_price_message(self, message):
        assert not is_price_question(message), message

    def test_policy_answer_varies_and_mentions_product_first(self):
        texts = {price_policy_answer("en", seed) for seed in range(8)}
        assert len(texts) >= 3
        assert all("quot" in t.lower() or "price" in t.lower() for t in texts)

    def test_price_policy_plus_next_question(self):
        question = question_for("installation", "en", 0)
        text = compose_requirement_reply(
            answer=price_policy_answer("en", 0),
            question=question,
            slot="installation",
            message="What is the price of smd screen wedth 2.5 feet",
            language="en",
            seed=0,
        )
        # 先回答价格问题，再继续问需求，且只问一个问题
        assert "quot" in text.lower()
        assert text.count("?") == 1
        assert text.rstrip().endswith("?")

    def test_chinese_policy(self):
        assert "报价" in price_policy_answer("zh", 0)


class TestNoSelfContradiction:
    """实测 bug：回复 "Got it, about 100 feet viewing distance. About how many
    metres away will people be sitting?" —— 先确认视距，再追问视距。"""

    def test_vieweing_distance_ack_does_not_precede_distance_question(self):
        text = compose_requirement_reply(
            question="About how many metres away will people be sitting?",
            slot="viewing_distance",
            message="about 100 feet viewing distance",
            language="en",
            llm_ack="Got it, about 100 feet viewing distance.",
        )
        assert text.count("?") == 1
        # 前面不能再出现"确认视距"的说法
        assert "100 feet" not in text
        assert "100 ft" not in text
        assert text.endswith("About how many metres away will people be sitting?")

    def test_ack_kept_when_it_does_not_touch_the_pending_slot(self):
        text = compose_requirement_reply(
            question="Is it a fixed installation, or do you need it for rental or events?",
            slot="installation",
            message="about 100 feet viewing distance",
            language="en",
            llm_ack="Got it, about 100 feet viewing distance.",
        )
        assert "100 feet" in text
        assert text.endswith("Is it a fixed installation, or do you need it for rental or events?")

    @pytest.mark.parametrize("ack,slot,expected", [
        ("Got it, indoor.", "environment", True),
        ("Got it, indoor.", "installation", False),
        ("Sure, fixed installation.", "installation", True),
        ("Got it, a church.", "purpose", True),
        ("Nice to meet you, Ar Majeed.", "purpose", False),
        ("Yes — we do carry a 1.2 mm COB LED.", "display_type", True),
        ("Yes — we do carry a 1.2 mm COB LED.", "environment", False),
        ("129.2 cm as the length and 45 cm as the width.", "size_axis", True),
        ("Got it — 129.2 cm noted.", "size_axis", False),
    ])
    def test_conflict_detection(self, ack, slot, expected):
        from src.rag.reply_composer import ack_conflicts_with_slot

        assert ack_conflicts_with_slot(ack, slot, "en") is expected


class TestComposeRequirementReply:

    def test_answer_plus_question(self):
        text = compose_requirement_reply(
            answer="I don't have specific info on that right now.",
            question="Will the screen be installed indoors or outdoors?",
            slot="environment",
            message="can i get ur representative at Indonesia. Thanks",
            language="en",
            seed=1,
        )
        assert "I don't have specific info" in text
        assert "indoors or outdoors" in text
        # 问句只出现一次
        assert text.count("?") == 1

    def test_ack_prepended_when_answer_already_asks_same_slot(self):
        question = question_for("environment", "en", 0)
        text = compose_requirement_reply(
            answer=question,
            question=question,
            slot="environment",
            message="Do u have  P 1.2 COB Led",
            language="en",
            seed=0,
        )
        assert text.lower().startswith("yes")
        assert text.count("?") == 1
        assert "indoors or outdoors" in text

    def test_no_answer_falls_back_to_ack_plus_question(self):
        text = compose_requirement_reply(
            question="What will the screen mainly be used for?",
            slot="installation",
            message="church",
            language="en",
            seed=0,
        )
        assert "church" in text
        assert "mainly be used for" in text

    def test_echo_never_confirms_the_slot_being_asked(self):
        """"先确认、再追问同一件事"是不允许的。"""
        text = compose_requirement_reply(
            question="What will the screen mainly be used for?",
            slot="purpose",
            message="church",
            language="en",
            seed=0,
        )
        assert "church" not in text.lower()
        assert "mainly be used for" in text

    def test_echo_keeps_other_facts_when_slot_excluded(self):
        text = compose_requirement_reply(
            question="What will the screen mainly be used for?",
            slot="purpose",
            message="yes indoor, about 100 feet viewing distance",
            language="en",
            seed=0,
        )
        assert "indoor" in text.lower()
        assert "100 ft" in text
        assert "mainly be used for" in text

    def test_no_question_returns_answer(self):
        assert compose_requirement_reply(answer="Hello there.") == "Hello there."

    def test_include_ack_can_be_disabled(self):
        text = compose_requirement_reply(
            question="Will the screen be installed indoors or outdoors?",
            slot="environment",
            message="church",
            language="en",
            include_ack=False,
        )
        assert text == "Will the screen be installed indoors or outdoors?"

    def test_chinese(self):
        text = compose_requirement_reply(
            question="这块屏是装在室内还是室外？",
            slot="environment",
            message="教堂用",
            language="zh",
            seed=0,
        )
        assert "教堂" in text
        assert "室内还是室外" in text


class _StubSales:
    def __init__(self, result):
        self.result = result
        self.memory_store = None
        self.solution_runner = None

    def run(self, session_id, message, has_vision=False, **_kwargs):
        return dict(self.result)


class TestSalesGraphAcksThenAsks:
    """Sales Agent 自己收需求这一支：回复必须先接住客户的话，再问下一个问题。"""

    @pytest.fixture
    def fake_llm(self, monkeypatch):
        import importlib

        classify_mod = importlib.import_module("src.agents.sales.nodes.classify")
        sales_req = importlib.import_module("src.agents.sales.nodes.requirement")

        class _Response:
            def __init__(self, content):
                self.content = content

        def _make(intent):
            class _Fake:
                def __init__(self, *args, **kwargs):
                    pass

                def invoke(self, messages, *args, **kwargs):
                    system = str(getattr(messages[0], "content", ""))
                    if "意图分类器" in system:
                        return _Response(intent)
                    return _Response('{"usage": null, "additional_requirements": []}')

            return _Fake

        return classify_mod, sales_req, _make

    def _state(self, message, messages=None):
        return {
            "messages": list(messages or []) + [{"role": "user", "content": message}],
            "current_message": message,
            "session_id": "composer-sales-graph",
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
            "pending_slot": "",
            "requirements_reset": False,
            "reset_reason": "",
        }

    def test_scenario_answer_is_acknowledged_then_question(self, fake_llm, monkeypatch):
        classify_mod, sales_req, _make = fake_llm
        monkeypatch.setattr(classify_mod, "ChatOpenAI", _make("need_query"))
        monkeypatch.setattr(sales_req, "ChatOpenAI", _make("need_query"))

        from src.agents.sales.graph import build_sales_graph

        result = build_sales_graph().invoke(self._state("church"))

        assert result["next_action"] == "ask"
        assert result["pending_question"]
        # 先接住客户这句话（复述场景），再问同一个 Gate 问题
        assert "church" in result["response"]
        assert result["response"].endswith(result["pending_question"])

    def test_product_question_keeps_question_for_orchestrator(self, fake_llm, monkeypatch):
        """product_question 由 orchestrator 拼接：Sales 侧只负责准备好待问项。"""
        classify_mod, sales_req, _make = fake_llm
        monkeypatch.setattr(classify_mod, "ChatOpenAI", _make("product_question"))
        monkeypatch.setattr(sales_req, "ChatOpenAI", _make("product_question"))

        from src.agents.sales.graph import build_sales_graph

        result = build_sales_graph().invoke(self._state("Do u have  P 1.2 COB Led"))

        assert result["next_action"] == "product_question"
        assert result["pending_question"], "product_question 也要带上待问项"
        assert result["pending_slot"]
        assert result["should_generate_solution"] is False

    def test_self_introduction_is_answered_not_just_questioned(self, monkeypatch):
        """实测 bug 复现：客户自我介绍 + 说明场景 + 要规格报价，不能只回一句追问。

        旧日志：`Alright, a restaurant. Is the installation going to be indoors or
        outdoors?`（"Akbar" 里的 "bar" 被当成餐厅场景）。
        """
        import importlib

        classify_mod = importlib.import_module("src.agents.sales.nodes.classify")
        sales_req = importlib.import_module("src.agents.sales.nodes.requirement")

        ack = "Nice to meet you, Ar Majeed — happy to help with the smart classroom project."
        message = (
            "My name is Ar Majeed Akbar from Pakistan, actually we are going for smart "
            "class room in our university so I need smart screen, so could you help me "
            "out about it's specifications and details also need quotation for further "
            "discussion"
        )

        class _Response:
            def __init__(self, content):
                self.content = content

        class _Fake:
            def __init__(self, *args, **kwargs):
                pass

            def invoke(self, messages, *args, **kwargs):
                system = str(getattr(messages[0], "content", ""))
                if "意图分类器" in system:
                    return _Response("product_question")
                return _Response(
                    '{"usage": "smart class room", "additional_requirements": [], '
                    f'"ack": "{ack}"}}'
                )

        monkeypatch.setattr(classify_mod, "ChatOpenAI", _Fake)
        monkeypatch.setattr(sales_req, "ChatOpenAI", _Fake)

        from src.agents.sales.graph import build_sales_graph

        result = build_sales_graph().invoke(self._state(message))

        # 场景识别正确（不是 restaurant）。
        # M1 之后 requirements 是 RequirementProfile 的投影，usage 用的是规范化 token
        assert result["requirements"].get("usage") == "classroom"
        assert result["requirement_profile"].purpose == "classroom"
        assert result["intent"] == "need_query"
        # 先回应客户这句话，再追问；而不是只有一句追问
        assert result["response"].startswith(ack)
        # 回应后可能带过渡语（客户同时问了报价），问句本体来自 Gate
        tail = result["pending_question"].rsplit("—", 1)[-1].strip().lower()
        assert result["response"].lower().rstrip().endswith(tail)
        assert "restaurant" not in result["response"].lower()
        assert result["should_generate_solution"] is False

    def test_volunteered_distance_is_used_not_re_asked(self, fake_llm, monkeypatch):
        """客户抢答的需求（100 feet 视距）必须被采纳，不能继续追问同一项。"""
        classify_mod, sales_req, _make = fake_llm
        monkeypatch.setattr(classify_mod, "ChatOpenAI", _make("need_query"))
        monkeypatch.setattr(sales_req, "ChatOpenAI", _make("need_query"))

        from src.agents.sales.graph import build_sales_graph

        message = "we need about 100 feet viewing distance"
        result = build_sales_graph().invoke(self._state(message))

        profile = result["requirement_profile"]
        assert profile.viewing_distance_m is not None
        assert abs(profile.viewing_distance_m - 30.48) < 0.01
        assert profile.sources.get("viewing_distance_m") == "explicit"
        # 已经拿到的信息不再追问：问的是还缺的环境
        assert result["pending_slot"] == "environment"
        assert "viewing distance" not in result["pending_question"].lower()

    def test_bare_measurement_stays_in_requirement_flow(self, fake_llm, monkeypatch):
        """实测 bug：客户答 "129,2cm" 被当成 others 绕到自由问答，倒出一堆型号。
        正确行为：留在需求采集里，先确认这个数字是宽 / 高 / 对角线。"""
        classify_mod, sales_req, _make = fake_llm
        monkeypatch.setattr(classify_mod, "ChatOpenAI", _make("others"))
        monkeypatch.setattr(sales_req, "ChatOpenAI", _make("others"))

        from src.agents.sales.graph import build_sales_graph

        result = build_sales_graph().invoke(self._state("129,2cm"))

        assert result["intent"] == "need_query"
        assert result["next_action"] == "ask"
        assert result["should_generate_solution"] is False
        assert result["solutions"] == []
        assert result["pending_slot"] == "size_axis"
        assert "129.2 cm" in result["pending_question"]
        assert result["response"].endswith(result["pending_question"])


class TestBareMeasurementEndToEnd:
    """多轮：客户报裸尺寸 → 系统确认方向 → 记成宽高，再继续问别的需求。"""

    @pytest.fixture
    def fake_llm(self, monkeypatch):
        import importlib

        classify_mod = importlib.import_module("src.agents.sales.nodes.classify")
        sales_req = importlib.import_module("src.agents.sales.nodes.requirement")

        class _Response:
            def __init__(self, content):
                self.content = content

        class _Fake:
            def __init__(self, *args, **kwargs):
                pass

            def invoke(self, messages, *args, **kwargs):
                system = str(getattr(messages[0], "content", ""))
                if "意图分类器" in system:
                    return _Response("others")
                return _Response('{"usage": null, "additional_requirements": [], "ack": ""}')

        monkeypatch.setattr(classify_mod, "ChatOpenAI", _Fake)
        monkeypatch.setattr(sales_req, "ChatOpenAI", _Fake)

    def test_axis_answer_resolves_the_size(self, fake_llm):
        from src.agents.sales.runner import SalesAgentRunner
        from src.memory.store import memory

        session_id = "e2e-bare-measurement"
        memory.clear(session_id)
        try:
            runner = SalesAgentRunner(memory_store=memory)
            runner.setup()

            first = runner.run(session_id, "129,2cm")
            assert first["next_action"] == "ask"
            assert "129.2 cm" in first["response"]

            second = runner.run(session_id, "it is the width")
            profile = memory.get_requirement_profile(session_id) or {}
            assert abs(float(profile.get("target_width_m") or 0) - 1.292) < 1e-6, profile
            # 已经确认过的不再追问：下一问是环境
            assert second["pending_slot"] == "environment"
            assert "129.2 cm" not in second["response"]
        finally:
            memory.clear(session_id)

    def test_two_dimensions_in_one_sentence_are_captured(self, fake_llm):
        """实测日志原句：客户一次给出长边 + 宽边（还带两个笔误），
        系统必须同时拿下两个尺寸，并且不再追问方向。"""
        from src.agents.sales.runner import SalesAgentRunner
        from src.memory.store import memory

        session_id = "e2e-two-dimensions"
        memory.clear(session_id)
        try:
            runner = SalesAgentRunner(memory_store=memory)
            runner.setup()

            runner.run(session_id, "129,2cm")   # 先报一个裸尺寸
            second = runner.run(
                session_id, "129,2cm us the length and 45xm is width"
            )

            profile = memory.get_requirement_profile(session_id) or {}
            assert abs(float(profile.get("target_width_m") or 0) - 1.292) < 1e-6, profile
            assert abs(float(profile.get("target_height_m") or 0) - 0.45) < 1e-6, profile
            # 方向已经确认，不许再问同一个问题
            assert second["pending_slot"] != "size_axis"
            assert "width, the height, or the diagonal" not in second["response"]
            assert second["next_action"] == "ask"
        finally:
            memory.clear(session_id)


class _StubSolution:
    def __init__(self, result):
        self.result = result

    def run(self, message, history=None, session_id=None, **_kwargs):
        return dict(self.result)


class TestOrchestratorComposesAnswerAndQuestion:
    """端到端：客人问别的问题时，orchestrator 必须"答完再问"。"""

    def _orchestrator(self, sales_result, solution_result):
        from src.memory.store import memory
        from src.orchestrator import DualAgentOrchestrator

        return DualAgentOrchestrator(
            sales_agent=_StubSales(sales_result),
            solution_agent=_StubSolution(solution_result),
        )

    def test_others_answer_then_question(self):
        from src.memory.store import memory

        session_id = "compose-others-test"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        try:
            orch = self._orchestrator(
                sales_result={
                    "intent": "others",
                    "next_action": "others",
                    "response": "Sure.",
                    "requirements": {"display_type": "LED"},
                    "pending_question": "Will the screen be installed indoors or outdoors?",
                    "pending_slot": "environment",
                },
                solution_result={
                    "answer": "We do work with partners in the region.",
                    "products": [],
                    "route": "agent",
                },
            )
            result = orch.process_message("can i get ur representative at Indonesia. Thanks", session_id)

            assert "We do work with partners in the region" in result["response"]
            assert "indoors or outdoors" in result["response"]
            assert result["next_action"] == "follow_up"
        finally:
            memory.clear(session_id)

    def test_others_route_carries_profile_and_intent(self):
        """实测回归：客户中途问"你们在肯尼亚有代理商吗？"时，

        旧行为是把这一句话单独丢给 Solution Agent → 它重建需求 → 又回头问
        "Are we talking about an indoor or an outdoor install?"（明明早就知道）。
        现在：Orchestrator 必须把**已收集的需求档案**和 Sales 定的意图一起传过去。
        """
        from src.memory.store import memory
        from src.models.requirement import RequirementProfile
        from src.orchestrator import DualAgentOrchestrator

        session_id = "compose-others-profile"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        try:
            slots = {
                "display_type": "LED", "environment": "indoor", "purpose": "conference",
                "installation": "fixed", "viewing_distance_m": 10,
            }
            memory.set_requirement_profile(
                session_id, RequirementProfile.from_slots(slots, explicit_keys=set(slots))
            )

            captured = {}

            class _CapturingSolution:
                def run(self, message, history=None, session_id=None, profile=None,
                        requirements=None, intent="", **_kwargs):
                    captured["profile"] = profile
                    captured["intent"] = intent
                    return {"answer": "We handle overseas projects directly from Shenzhen.",
                            "products": [], "route": "agent"}

            orch = DualAgentOrchestrator(
                sales_agent=_StubSales({
                    "intent": "others",
                    "next_action": "others",
                    "response": "Sure.",
                    "requirements": {},
                    "products": [],
                }),
                solution_agent=_CapturingSolution(),
            )

            result = orch.process_message("你们在肯尼亚有代理商吗？", session_id)

            assert captured["intent"] == "others", "意图由 Sales 定，Solution 不该再自己判一遍"
            profile = captured["profile"]
            assert profile is not None, "必须带上已收集的需求档案"
            assert profile.environment == "indoor"
            assert profile.purpose == "conference"
            assert "Shenzhen" in result["response"]
        finally:
            memory.clear(session_id)

    def test_product_question_does_not_repeat_the_question(self):
        from src.memory.store import memory

        session_id = "compose-product-test"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        try:
            question = question_for("environment", "en", 2)
            orch = self._orchestrator(
                sales_result={
                    "intent": "product_question",
                    "next_action": "product_question",
                    "response": "Sure.",
                    "requirements": {"display_type": "LED"},
                    "pending_question": question,
                    "pending_slot": "environment",
                },
                solution_result={"answer": question, "products": [], "route": "normal"},
            )
            result = orch.process_message("Do u have  P 1.2 COB Led", session_id)

            assert result["response"].count("?") == 1
            assert result["response"].lower().startswith("yes")
        finally:
            memory.clear(session_id)
