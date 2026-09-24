"""计划 v2.9.4 §十三（LCD 链路重新建立）：入口可用、需求链按计划留白。

计划 §十五 原文：

    新的 LCD 链路不要直接继承旧链路全部逻辑。
    先只建立：LCD → LCD Requirement Chain →【具体需求链暂留白】→ LCD Recommendation
    后续再独立设计：LCD 场景 / 尺寸 / 拼接 / 拼接缝 / 安装 / 使用环境 / 观看距离 / 产品选择

所以这一阶段的验收口径是"入口正确 + 不越界"：

  1. 客户确认 LCD 之后，本轮**只回一句自然承接**，不套用 LED 的需求问题；
  2. 不触发 LED 的推荐链路（不推产品）；
  3. 入口留痕（``state["lcd_entry"]``），需求链按计划留白（``LCDPolicy`` 仍未实现）。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _lcd_state(subtype: str = "", pending_question: str = "", pending_slot: str = ""):
    return {
        "intent": "need_query",
        "current_message": "we will go with the video wall",
        "display_type_decision": {
            "display_type": "LCD",
            "subtype": subtype,
            "status": "CONFIRMED",
            "source": "CUSTOMER",
            "locked": True,
        },
        "pending_question": pending_question,
        "pending_slot": pending_slot,
        "requirements": {},
    }


class TestLcdEntry:

    def test_lcd_confirmed_turn_is_a_natural_ack_not_an_led_question(self):
        """LED 的需求问题（室内外 / 点间距…）不能顺延到 LCD 上。"""
        from src.agents.sales.nodes.script_generator import script_generator

        state = _lcd_state(
            pending_question="Is this going to be installed indoors or outdoors?",
            pending_slot="environment",
        )
        out = script_generator(state)
        reply = str(out.get("response") or "")
        lowered = reply.lower()

        assert reply, "LCD 入口必须有回复"
        assert "lcd" in lowered, reply
        assert "indoors or outdoors" not in lowered, reply
        assert out.get("lcd_entry", {}).get("status") == "REQUIREMENT_CHAIN_PENDING"
        assert not out.get("should_generate_solution"), "LCD 需求链留白，本轮不该推产品"

    def test_lcd_entry_keeps_the_ifp_subtype(self):
        """§十四：IFP 是 LCD 的子类型 —— 入口要按 IFP 称呼，但仍走 LCD_ENTRY。"""
        from src.agents.sales.nodes.script_generator import script_generator

        out = script_generator(_lcd_state(subtype="IFP"))
        reply = str(out.get("response") or "").lower()

        assert "interactive flat panel" in reply, out.get("response")
        assert out.get("lcd_entry", {}).get("subtype") == "IFP"
        assert out.get("lcd_entry", {}).get("product_domain") == "LCD"

    def test_lcd_policy_still_marks_the_requirement_chain_as_pending(self):
        """计划 §十五：具体需求链暂留白 —— 策略层如实报告"未实现"，不假装能推。"""
        from src.dialogue.product_router import LCDPolicy, policy_for

        policy = policy_for("LCD")
        assert isinstance(policy, LCDPolicy)
        assert policy.implemented is False
        assert policy.get_next_question(None) is None
        assert policy.get_missing_requirements(None) == []
        assert policy.can_recommend(None) is False
        assert policy.recommend(None) == []

    def test_led_entry_is_not_affected_by_the_lcd_branch(self):
        """LED 链路保持不动（计划 §十四）：LED 已确认时不该出现 LCD 入口留痕。"""
        from src.agents.sales.nodes.script_generator import script_generator

        state = {
            "intent": "need_query",
            "current_message": "we will go with LED",
            "display_type_decision": {
                "display_type": "LED",
                "status": "CONFIRMED",
                "source": "CUSTOMER",
                "locked": True,
            },
            "requirements": {},
            "requirements_reset": True,
            "pending_question": "Is this going to be installed indoors or outdoors?",
            "pending_slot": "environment",
        }
        out = script_generator(state)

        assert "lcd_entry" not in out, "LED 链路不得被 LCD 入口接管"
