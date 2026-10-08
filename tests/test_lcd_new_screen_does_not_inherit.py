"""回归：客户开始说"新的一块屏"时，**不继承**上一块的需求事实。

客户口径（2026-10）：

    "如果一开始客户说的就是要两个屏幕就多需求并存；
     如果推荐完一款客户说还需要一款，就照着这一款接着问需求。" + "不继承上一块的需求"

实测 bug：

    客户（展览会那条）：… exhibition / 3x3 视频墙 / 65 寸 …
    客户：我还需要一款屏幕，可有手写的会议室使用的
    AI  ：For your exhibition, we recommend the Omni T65-K4/K4C …
          Your 3x3 video wall layout will use 9 of these panels

两个问题叠在一起：
  1. 场景：``purpose`` 以前是"只写一次"，旧的 ``exhibition`` 永远改不掉
     （``apply_lcd_facts`` 里 ``if not getattr(profile, "purpose", None)``）；
  2. 需求事实：旧的 ``lcd_is_splicing / layout / count / size`` 全被继承 ——
     会议平板还被套上了 3x3 拼接。

修法：识别"新的一块屏"（复用 LED 多屏那套显式信号）→ 清掉上一块的需求事实，
只保留产品类型（链路不变），新这块屏的场景按客户当下说的重新判定。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _exhibition_lcd_profile():
    """上一块屏：展览会 3x3 视频墙。"""
    from src.models.requirement import RequirementProfile

    profile = RequirementProfile()
    profile.display_type = "LCD"
    profile.purpose = "exhibition"
    profile.environment = "indoor"
    profile.installation = "fixed"
    profile.lcd_category = "advertising"
    profile.lcd_size_inch = 65.0
    profile.lcd_size_source = "customer_explicit"
    profile.lcd_resolution = "4K"
    profile.lcd_is_splicing = True
    profile.lcd_splicing_layout = "3x3"
    profile.lcd_screen_count = 9
    profile.lcd_touch_required = False
    profile.sources.update(
        {
            "purpose": "explicit",
            "environment": "explicit",
            "installation": "explicit",
            "lcd_category": "understanding",
            "lcd_size_inch": "explicit",
            "lcd_size_source": "explicit",
            "lcd_resolution": "explicit",
            "lcd_is_splicing": "explicit",
            "lcd_splicing_layout": "explicit",
            "lcd_screen_count": "explicit",
            "lcd_touch_required": "explicit",
        }
    )
    return profile


class TestNewItemDetection:

    def test_the_customer_phrasing_is_recognised(self):
        """"我还需要一款屏幕 …" 必须被认成"新的一块屏"（以前的说法匹配不到）。"""
        from src.rag.project_items import detect_new_item

        profile = _exhibition_lcd_profile()
        is_new, why = detect_new_item("我还需要一款屏幕，可有手写的会议室使用的", profile)
        assert is_new is True, why

    def test_normal_turns_do_not_start_a_new_item(self):
        """普通轮次（回答参数 / 要别的推荐 / 问库存）不能误判成新的一块屏。"""
        from src.rag.project_items import detect_new_item

        profile = _exhibition_lcd_profile()
        for message in ("i need 3x3", "any other recommendation?", "do you have stock?"):
            is_new, why = detect_new_item(message, profile)
            assert is_new is False, (message, why)


class TestResetDoesNotInherit:

    def test_reset_clears_scene_and_lcd_facts_but_keeps_product_type(self):
        from src.dialogue.lcd_decision import reset_lcd_requirement_facts

        profile = _exhibition_lcd_profile()
        cleared = reset_lcd_requirement_facts(profile)

        # 产品类型（走哪条链路）必须保留
        assert profile.display_type == "LCD"
        # 上一块的场景 / 需求事实全部清掉
        for name in (
            "purpose", "environment", "installation", "lcd_category",
            "lcd_size_inch", "lcd_resolution", "lcd_is_splicing",
            "lcd_splicing_layout", "lcd_screen_count", "lcd_touch_required",
        ):
            assert getattr(profile, name, None) in (None, "", [], {}, ()), (name, getattr(profile, name))
        assert "purpose" in cleared and "lcd_is_splicing" in cleared
        # 来源也要一起清，否则旧来源会继续"锁住"新值
        for name in ("purpose", "lcd_category", "lcd_is_splicing", "lcd_splicing_layout"):
            assert name not in (profile.sources or {}), name

    def test_purpose_is_writable_after_reset(self):
        """关键：清掉之后场景能被**新这块屏**重新写进去（旧实现只写一次）。"""
        from src.dialogue.lcd_decision import reset_lcd_requirement_facts
        from src.models.requirement import RequirementProfile

        profile = _exhibition_lcd_profile()
        reset_lcd_requirement_facts(profile)
        assert profile.purpose is None

        # 新这块屏是会议室 → 场景必须能写进去
        fresh = RequirementProfile.from_slots(
            {"purpose": "conference", "display_type": "LCD"},
            explicit_keys={"purpose", "display_type"},
        )
        profile = profile.merge(fresh)
        assert profile.purpose == "conference", profile.purpose


class TestLcdEntryResetsOnANewScreen:

    def _state(self, session_id, profile):
        return {
            "session_id": session_id,
            "current_message": "",
            "already_recommended": True,
            "requirement_profile": profile,
            "display_type_decision": {
                "display_type": "LCD",
                "status": "CONFIRMED",
                "source": "CUSTOMER",
                "locked": True,
            },
        }

    def test_new_screen_message_clears_the_previous_screens_facts(self):
        from src.agents.sales.nodes.requirement import _lcd_requirement_action
        from src.memory.store import memory

        session_id = "lcd-new-item-1"
        memory.clear(session_id)
        try:
            profile = _exhibition_lcd_profile()
            message = "我还需要一款屏幕，可有手写的会议室使用的"
            state = self._state(session_id, profile)

            _lcd_requirement_action(state, profile, message)

            # 走的是"新的一块屏"这条路
            assert state.get("lcd_new_item"), state.get("lcd_new_item")
            # 上一块的场景（exhibition）必须没了，换成新这块屏的场景（会议室）
            assert profile.purpose, "新的一块屏要能重新报出场景"
            assert "exhibition" not in str(profile.purpose).lower(), profile.purpose
            assert "meeting" in str(profile.purpose).lower(), profile.purpose
            # 上一块的拼接事实不许继承
            assert profile.lcd_splicing_layout in (None, ""), profile.lcd_splicing_layout
            assert profile.lcd_screen_count in (None, 0, ""), profile.lcd_screen_count
            assert profile.lcd_is_splicing is not True, profile.lcd_is_splicing
        finally:
            memory.clear(session_id)

    def test_a_normal_turn_keeps_the_current_screen_facts(self):
        """普通轮次（继续答同一块屏）不许清空需求。"""
        from src.agents.sales.nodes.requirement import _lcd_requirement_action
        from src.memory.store import memory

        session_id = "lcd-new-item-2"
        memory.clear(session_id)
        try:
            profile = _exhibition_lcd_profile()
            state = self._state(session_id, profile)

            _lcd_requirement_action(state, profile, "i need 3x3")

            assert not state.get("lcd_new_item")
            assert profile.lcd_size_inch == 65.0, "同一块屏的既有需求不能被清掉"
        finally:
            memory.clear(session_id)
