"""品类由语境决定 + 锁定屏幕（客户口径 2026-09-30）。

客户原话：

    「不允许有关键词来触发 AI 的某个链路，需要让 AI 结合语境上下文，
      去选择品类，然后根据对应品类推进需求询问，然后锁定屏幕。」

这里守住四件事：

    1. 语境结论定品类（客户一句话里没有任何场景关键词也能定）
    2. 品类一旦定下就锁定 —— 关键词再也改不动它，只有"客户改口"（语境结论变化）才允许换
    3. 客户只是在回答参数（尺寸 / 室内外）时，沿用已锁定的品类，不退回 unknown、不问用途
    4. 屏幕锁定：需求指纹没变 → 复用同一个型号；需求变了 → 才允许重新选型
"""
import os
import sys
from types import SimpleNamespace

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import lcd_category_understanding as lcu  # noqa: E402
from src.dialogue.lcd_decision import (  # noqa: E402
    ADVERTISING,
    CONFERENCE_EDUCATION,
    MONITORING,
    UNKNOWN,
    category_is_locked,
    lcd_turn,
)
from src.memory.store import memory  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402


def _profile(**slots) -> RequirementProfile:
    base = {"display_type": "LCD"}
    base.update(slots)
    return RequirementProfile.from_slots(base, explicit_keys=set(base))


class TestCategoryComesFromContext:
    def test_context_decides_the_category_without_any_scene_keyword_merged(self):
        """合并自 7 条同类测试（瘦身；断言全部保留）。"""

        # ── test_context_decides_the_category_without_any_scene_keyword ──
        """客户说的是"想在墙上看画面"，一个场景关键词都没有 → 语境判定为监控。"""
        profile = _profile()
        _profile_out, action = lcd_turn(
            profile,
            "I want one big picture on the wall so the duty team can watch every feed",
            category_signal={"category": MONITORING, "confidence": 0.9},
        )

        assert action.lcd_category == MONITORING, action.lcd_category
        # 监控分支的第一个问题：室内还是室外
        assert action.question_slot == "environment", action.question_slot
        assert action.next_action != "ask_purpose"

        # ── test_category_is_locked_once_understood ──
        profile = _profile()
        lcd_turn(
            profile,
            "for our control room",
            category_signal={"category": MONITORING, "confidence": 0.9},
        )

        assert profile.lcd_category == MONITORING
        assert (profile.sources or {}).get("lcd_category") == "understanding"
        assert category_is_locked(profile) is True

        # ── test_keyword_hint_cannot_override_a_locked_category ──
        """品类锁定后，后面这句里出现"advertising"也不能把分支改掉。"""
        profile = _profile()
        lcd_turn(
            profile,
            "for our control room",
            category_signal={"category": MONITORING, "confidence": 0.9},
        )
        # 关键词兜底路径（没有语境结论）——模拟模型不可用时的降级
        _profile_out, action = lcd_turn(profile, "also maybe some advertising signage")

        assert action.lcd_category == MONITORING, action.lcd_category
        assert profile.lcd_category == MONITORING

        # ── test_understanding_can_change_the_category_when_customer_changes_mind ──
        profile = _profile()
        lcd_turn(
            profile,
            "for our control room",
            category_signal={"category": MONITORING, "confidence": 0.9},
        )
        _profile_out, action = lcd_turn(
            profile,
            "actually it is for the meeting room, we need a whiteboard",
            category_signal={"category": CONFERENCE_EDUCATION, "confidence": 0.95},
        )

        assert action.lcd_category == CONFERENCE_EDUCATION, action.lcd_category

        # ── test_parameter_answer_keeps_the_locked_category ──
        """客户只答参数（尺寸 / 室内外）时，不能因为这一句没提场景就退回 unknown。"""
        profile = _profile()
        lcd_turn(
            profile,
            "for our control room",
            category_signal={"category": MONITORING, "confidence": 0.9},
        )
        _profile_out, action = lcd_turn(
            profile,
            "65 inch indoor",
            category_signal={"category": UNKNOWN, "confidence": 0.2},
        )

        assert action.lcd_category == MONITORING, action.lcd_category
        assert action.question_slot != "purpose", "用途已经定了，不能再问用途"

        # ── test_keyword_fallback_still_works_when_model_is_unavailable ──
        """降级路径：没有语境结论时，关键词兜底仍然能选维度（不能把链路打死）。"""
        profile = _profile()
        _profile_out, action = lcd_turn(profile, "we need a video wall for the control room")

        assert action.lcd_category == MONITORING, action.lcd_category
        assert profile.lcd_category == MONITORING

        # ── test_unknown_stays_unknown_when_there_is_nothing_to_go_on ──
        profile = _profile()
        _profile_out, action = lcd_turn(profile, "I need an LCD.")

        assert action.lcd_category == UNKNOWN, action.lcd_category
        assert action.question_slot == "purpose"


class TestUnderstandingModule:
    def test_parses_a_valid_signal_merged(self):
        """合并自 3 条同类测试（瘦身；断言全部保留）。"""

        # ── test_parses_a_valid_signal ──
        parsed = lcu._parse(
            '{"category": "advertising", "confidence": 0.8, "scene": "shopping mall lobby",'
            ' "reason": "publishes ads to passers-by"}'
        )
        assert parsed["category"] == ADVERTISING
        assert parsed["confidence"] == pytest.approx(0.8)
        assert parsed["scene"] == "shopping mall lobby"

        # ── test_rejects_an_unknown_category_value ──
        assert lcu._parse('{"category": "splicing", "confidence": 0.9}') == {}

        # ── test_rejects_non_json ──
        assert lcu._parse("I think it is monitoring.") == {}

    def test_asks_the_llm_with_the_context(self, monkeypatch):
        captured = {}

        class _FakeLLM:
            def invoke(self, messages, *args, **kwargs):
                captured["messages"] = messages
                return SimpleNamespace(
                    content='{"category": "conference_education", "confidence": 0.7,'
                    ' "scene": "meeting room", "reason": "team meetings"}'
                )

        monkeypatch.setattr("src.core.llm.get_llm", lambda **kwargs: _FakeLLM())
        lcu.clear_cache()

        signal = lcu.understand_lcd_category(
            "we also need the whiteboard",
            session_id="ctx-1",
            conversation="客户：我们要一间会议室的屏\nAI：Do you need touch?",
            asked_question="Do you need touch?",
            profile=None,
        )

        assert signal["category"] == CONFERENCE_EDUCATION
        blob = "\n".join(str(getattr(m, "content", "")) for m in captured["messages"])
        assert "最近的对话" in blob and "Do you need touch?" in blob

    def test_returns_empty_when_the_model_fails(self, monkeypatch):
        def _boom(**kwargs):
            raise RuntimeError("no network")

        monkeypatch.setattr("src.core.llm.get_llm", _boom)
        lcu.clear_cache()

        assert (
            lcu.understand_lcd_category("hello", session_id="ctx-2") == {}
        ), "模型不可用时必须返回空信号，交给关键词兜底"


def _candidate(model: str, **meta):
    return {"metadata": dict(meta, model=model, display_type="LCD")}


class TestScreenLock:
    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch):
        # 选型节点只在本测试里用，不连模型：直接让表达层走模板兜底
        def _boom(*args, **kwargs):
            raise RuntimeError("no network in tests")

        monkeypatch.setattr(
            "src.agents.solution.nodes.recommend.get_llm", _boom, raising=False
        )
        memory.clear_all()
        yield
        memory.clear_all()

    def _raw_products(self):
        return [
            _candidate(
                "H6530LN-B",
                display_size_inch='65"',
                resolution="3840x2160",
                bazel_mm="3.5mm",
                brightness_nit=500,
                is_splicing=True,
            ),
            _candidate(
                "DS-W-65",
                display_size_inch='65"',
                resolution="1920x1080 / 3840x2160",
                brightness_nit=300,
                is_splicing=False,
            ),
        ]

    def _monitoring_profile(self, size=65.0):
        return _profile(
            lcd_category=MONITORING,
            lcd_is_splicing=True,
            lcd_splicing_layout="3x3",
            lcd_screen_count=9,
            lcd_bezel_mm=3.5,
            lcd_size_inch=size,
            lcd_resolution="4K",
        )

    def test_recommend_locks_the_screen_merged(self):
        """合并自 4 条同类测试（瘦身；断言全部保留）。"""

        # ── test_recommend_locks_the_screen ──
        from src.agents.solution.nodes.recommend import _recommend_lcd

        state = {"session_id": "lock-probe-1"}
        _recommend_lcd(dict(state), self._monitoring_profile(), self._raw_products())

        locked = memory.get_lcd_lock("lock-probe-1")
        assert locked.get("model") == "H6530LN-B", locked
        assert locked.get("category") == MONITORING
        assert locked.get("fingerprint", {}).get("lcd_size_inch") == 65.0

        # ── test_locked_screen_is_reused_even_if_candidate_order_changes ──
        from src.agents.solution.nodes.recommend import _recommend_lcd

        state = {"session_id": "lock-probe-2"}
        profile = self._monitoring_profile()
        products = self._raw_products()
        _recommend_lcd(dict(state), profile, products)

        shuffled = list(reversed(products))
        out = _recommend_lcd(dict(state), profile, shuffled)
        picked = (out.get("products") or [{}])[0].get("metadata", {}).get("model")
        assert picked == "H6530LN-B", picked

        # ── test_lock_is_dropped_when_the_requirement_changes ──
        from src.agents.solution.nodes.recommend import _recommend_lcd

        state = {"session_id": "lock-probe-3"}
        _recommend_lcd(dict(state), self._monitoring_profile(65.0), self._raw_products())
        locked_before = memory.get_lcd_lock("lock-probe-3")

        _recommend_lcd(dict(state), self._monitoring_profile(55.0), self._raw_products())
        locked_after = memory.get_lcd_lock("lock-probe-3")

        assert locked_before["fingerprint"] != locked_after["fingerprint"]
        assert locked_after["fingerprint"]["lcd_size_inch"] == 55.0

        # ── test_reset_requirement_state_clears_the_lock ──
        from src.agents.solution.nodes.recommend import _recommend_lcd

        state = {"session_id": "lock-probe-4"}
        _recommend_lcd(dict(state), self._monitoring_profile(), self._raw_products())
        assert memory.get_lcd_lock("lock-probe-4")

        memory.reset_requirement_state("lock-probe-4")
        assert memory.get_lcd_lock("lock-probe-4") == {}
