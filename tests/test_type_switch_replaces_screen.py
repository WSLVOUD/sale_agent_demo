"""回归：客户**改主意换产品大类**时，旧的那块屏要被替换掉，不能并列成 Screen 1。

客户口径（2026-10 实测）：

    客户: i need a lcd display → advertising and indoor → 75'' → touch
    AI  : DS-M-75 …
    客户: no , i change my mind , i need a led display      ← 改主意要 LED
    … 采集完 LED 需求后 …
    AI  : Screen 1 (indoor / advertising): DS-M-75                    ← ❌ 已放弃的 LCD
          Screen 2 (outdoor / advertising): TW11-OD-P3
    客户: i only need screen2

换产品大类**不是"再来一块屏"**，而是把原来那块换掉。两处配合才修好：

  1. ``MultiScreenManager._maybe_start_new_item``：reason 是 ``display_type_switch``
     时**替换**（清空条目重来），不做归档、不复制旧需求；
  2. ``memory.reset_requirement_state``：整体重置时也要清 ``project_items`` /
     ``active_item_index``，否则旧的 LCD 条目会一直留在多屏列表里。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _lcd_screen_profile():
    """上一块屏：室内广告 75 寸 LCD（已经描述清楚，算"一块屏"）。"""
    from src.models.requirement import RequirementProfile

    profile = RequirementProfile()
    profile.display_type = "LCD"
    profile.purpose = "advertising"
    profile.environment = "indoor"
    profile.lcd_size_inch = 75.0
    profile.lcd_touch_required = True
    profile.sources.update(
        {
            "display_type": "customer_explicit",
            "purpose": "explicit",
            "environment": "explicit",
            "lcd_size_inch": "explicit",
        }
    )
    return profile


class TestResetClearsMultiScreenItems:

    def test_reset_drops_project_items(self):
        from src.memory.store import memory

        session_id = "switch-items-reset"
        memory.clear(session_id)
        try:
            memory.set_project_items(
                session_id,
                [{"profile": {"display_type": "LCD", "lcd_size_inch": 75.0}, "model": "DS-M-75"}],
            )
            memory.set_active_item_index(session_id, 0)

            memory.reset_requirement_state(session_id)

            assert memory.get_project_items(session_id) == []
            assert memory.get_active_item_index(session_id) == 0
        finally:
            memory.clear(session_id)

    def test_reset_does_not_wipe_the_first_contact_flag(self):
        """整体重置需求 ≠ 清会话：首次接待标记不该被顺手清掉。"""
        from src.memory.store import memory

        session_id = "switch-items-reset-2"
        memory.clear(session_id)
        try:
            memory.mark_first_contact_done(session_id)
            memory.reset_requirement_state(session_id)
            assert memory.is_first_contact_done(session_id) is True
        finally:
            memory.clear(session_id)


class TestTypeSwitchReplacesInsteadOfArchiving:

    def _manager(self, profile):
        from src.rag.multi_screen import MultiScreenManager
        from src.memory.store import memory

        return MultiScreenManager(store=memory, profile_lookup=lambda _sid: profile)

    def test_change_of_mind_clears_the_old_screen(self):
        from src.memory.store import memory

        session_id = "switch-items-mgr-1"
        memory.clear(session_id)
        try:
            profile = _lcd_screen_profile()
            # 旧的 LCD 那块已经归档在项目里
            memory.set_project_items(
                session_id,
                [{"profile": profile.model_dump(), "model": "DS-M-75"}],
            )
            memory.set_active_item_index(session_id, 0)
            memory.mark_recommendation_done(
                session_id, [{"model": "DS-M-75"}]
            )

            manager = self._manager(profile)
            reason = manager._maybe_start_new_item(
                session_id, "no , i change my mind , i need a led display"
            )

            assert reason.startswith("display_type_switch"), reason
            # 旧的 LCD 条目必须没了 —— 不能再以 Screen 1 出现
            assert memory.get_project_items(session_id) == [], memory.get_project_items(session_id)
            assert memory.get_active_item_index(session_id) == 0
        finally:
            memory.clear(session_id)

    def test_a_genuine_second_screen_is_still_archived(self):
        """真正的"另一块屏"（同项目多屏）行为不能被这次修改弄坏。"""
        from src.memory.store import memory

        session_id = "switch-items-mgr-2"
        memory.clear(session_id)
        try:
            profile = _lcd_screen_profile()
            memory.set_project_items(
                session_id, [{"profile": profile.model_dump()}]
            )
            memory.set_active_item_index(session_id, 0)

            manager = self._manager(profile)
            reason = manager._maybe_start_new_item(
                session_id, "i also need a screen for the lobby entrance"
            )

            assert reason, "另一块屏应该被识别"
            assert not reason.startswith("display_type_switch"), reason
            # 归档：第 1 块屏保留，活跃下标前进到第 2 块
            items = memory.get_project_items(session_id)
            assert len(items) >= 2, items
            assert memory.get_active_item_index(session_id) == 1
        finally:
            memory.clear(session_id)
