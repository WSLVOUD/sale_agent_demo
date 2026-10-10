"""回归：客户**直接指名型号**时，要围绕这个型号问需求（LED 与 LCD 都要支持）。

客户口径（2026-10 实测）：

    客户: i need a TW-OD-11
    AI  : … are you after an LED display, or something like an LCD video wall?  ← 还在问 LED/LCD
    客户: LED
    AI  : … Is this going to be set up indoors or outdoors?                    ← 还在问室内外
    客户: outdoor

`TW-OD-11` 是 `TW11-OD` 的错序写法；目录里 `TW11-OD` 是真实系列（P2.5–P10）。
型号本身已经确定的属性（LED/LCD、室内外）**不该再问客户**。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _apply(message):
    """跑一次预填，返回 (state, profile, series)。"""
    from src.agents.sales.nodes.requirement import _apply_mentioned_model
    from src.models.requirement import RequirementProfile

    state = {"requirements": {}}
    profile = RequirementProfile()
    series = _apply_mentioned_model(state, profile, message)
    return state, profile, series


class TestLedNamedModel:

    def test_misordered_model_resolves_to_the_real_series(self):
        state, profile, series = _apply("i need a TW-OD-11")
        assert series == "TW11-OD", series
        assert state["specified_model"] == "TW11-OD"

    def test_the_type_and_environment_come_from_the_model(self):
        state, profile, _ = _apply("i need a TW-OD-11")
        assert profile.display_type == "LED", profile.display_type
        assert profile.environment == "outdoor", profile.environment
        # 来源标注为"客户指的型号"，不是推断
        assert profile.sources.get("display_type") == "customer_model"
        assert profile.sources.get("environment") == "customer_model"

    def test_the_gate_inputs_are_prefilled_too(self):
        """.requirements 是 Gate 实际读的那份，也要同步。"""
        state, _, _ = _apply("i need a TW-OD-11")
        reqs = state["requirements"]
        assert reqs.get("display_type") == "LED", reqs
        assert reqs.get("environment") == "outdoor", reqs
        assert reqs.get("outdoor") is True and reqs.get("indoor") is False, reqs


class TestLcdNamedModel:

    def test_a_simple_lcd_model_name_resolves(self):
        for message, series in (
            ("do you have P65?", "P65"),
            ("we need P110", "P110"),
            ("H5530LN-B 可以吗", "H5530LN-B"),
        ):
            state, profile, got = _apply(message)
            assert got == series, (message, got)
            assert profile.display_type == "LCD", (message, profile.display_type)
            assert state["requirements"].get("display_type") == "LCD"


class TestItNeverOverwritesOrGuesses:

    def test_a_customer_stated_fact_is_not_overwritten(self):
        """客户已经说过的事实不许被型号推断覆盖。"""
        from src.agents.sales.nodes.requirement import _apply_mentioned_model
        from src.models.requirement import RequirementProfile

        state = {"requirements": {"environment": "indoor"}}
        profile = RequirementProfile()
        profile.environment = "indoor"
        profile.sources["environment"] = "explicit"
        profile.display_type = "LCD"
        profile.sources["display_type"] = "customer_explicit"

        _apply_mentioned_model(state, profile, "i need a TW-OD-11")

        assert profile.environment == "indoor", "客户说的室内被型号覆盖了"
        assert profile.display_type == "LCD", "客户说的 LCD 被型号覆盖了"

    def test_non_model_messages_change_nothing(self):
        for message in ("how much is it", "i need a led screen", "3*5 outdoor", ""):
            state, profile, series = _apply(message)
            assert series == "", (message, series)
            assert profile.display_type is None, (message, profile.display_type)
            assert profile.environment is None, (message, profile.environment)
            assert "specified_model" not in state.get("requirements", {}), message

    def test_the_node_calls_it_before_the_gate_decides(self):
        source = open(
            os.path.join(project_root, "src", "agents", "sales", "nodes", "requirement.py"),
            encoding="utf-8",
        ).read()
        call = source.index("_apply_mentioned_model(state, profile, current_msg_text)")
        gate = source.index("[RecommendationGate]", call)
        assert call < gate, "预填必须发生在 Gate 决定问什么之前"


class TestTheProductDomainComesFromTheModel:
    """客户实测："i need a TW11-OD-P4" 之后仍被问 "LED 还是 LCD？"。

    根因：`detect_product_domain` 只认关键词，型号里没有 "LED" 字样 → 域判成
    UNKNOWN → 触发 PRODUCT_SELECTION（"LED / LCD / 交互平板？"）。
    型号属于哪个域，目录里写着，就直接用它。
    """

    def test_a_named_model_decides_the_domain(self):
        from src.dialogue.turn_kind import detect_product_domain

        for message, domain in (
            ("i need a TW11-OD-P4", "LED"),
            ("i need a TW-OD-11", "LED"),       # 错序写法照样认
            ("do you have P65?", "LCD"),
            ("H5530LN-B", "LCD"),
        ):
            assert detect_product_domain(message) == domain, message

    def test_without_a_model_the_domain_stays_unknown(self):
        """没有型号可依据时不许乱猜（仍会正常问品类）。"""
        from src.dialogue.turn_kind import detect_product_domain

        assert detect_product_domain("how much is it") == "UNKNOWN"
        assert detect_product_domain("i need a screen") == "UNKNOWN"


class TestTheTypeGateClosesForANamedModel:
    """真正的开关：类型问题只在 `display_type_decision` 未 CONFIRMED 时问。

    实测：客户说 "i need a TW11-OD-P4"（句子里没有 "LED" 字样）→ 类型判不出来 →
    类型闸门一直问 "are you after an LED display, or … an LCD solution …"。
    所以**必须在类型判定源头**给出 CONFIRMED，消费方才不会再问。
    """

    def _gate_question(self, message):
        from src.agents.sales.nodes.script_generator import _product_type_gate_question
        from src.dialogue.product_type_router import route_display_type

        decision = route_display_type(message)
        return (
            decision,
            _product_type_gate_question(
                {"display_type_decision": decision.to_dict(), "current_message": message}
            ),
        )

    def test_a_named_led_model_confirms_the_type_and_silences_the_question(self):
        decision, question = self._gate_question("i need a TW11-OD-P4")
        assert decision.display_type == "LED", decision.display_type
        assert decision.status == "CONFIRMED", decision.status
        assert question == "", f"不该再问品类，却问了：{question}"

    def test_a_misordered_model_also_confirms(self):
        decision, question = self._gate_question("i need a TW-OD-11")
        assert decision.display_type == "LED" and question == ""

    def test_a_named_lcd_model_confirms(self):
        decision, question = self._gate_question("do you have P65?")
        assert decision.display_type == "LCD", decision.display_type
        assert decision.status == "CONFIRMED"
        assert question == "", f"不该再问品类，却问了：{question}"

    def test_without_a_model_the_type_question_is_still_asked(self):
        """不能把闸门弄死：没有型号依据时，该问还得问。"""
        decision, question = self._gate_question("i need a screen")
        assert decision.status != "CONFIRMED"
        assert question != "", "没有依据却不问品类了 —— 闸门被弄死了"
