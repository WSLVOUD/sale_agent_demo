"""
《三套需求状态统一改造计划 v2.0》的回归测试（M1 / M2 / M3 / M5 / M7 / M11 / M12）。

覆盖计划第十二节的 9 个 Case，以及 M1「Profile 是唯一主状态」的核心断言。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.requirement import (  # noqa: E402
    CONFIRMED_SOURCES,
    RequirementProfile,
)
from src.rag.query_understanding import extract_slots, merge_slots  # noqa: E402
from src.rag.readiness import check_recommendation_ready  # noqa: E402


def _profile(message: str) -> RequirementProfile:
    slots = extract_slots(message)
    return RequirementProfile.from_slots(slots, explicit_keys=set(slots))


class TestCase1InsufficientRequirements:
    """Case 1：需求不足 → 不推荐、继续问。"""

    def test_bare_display_type_is_not_ready(self):
        decision = check_recommendation_ready(_profile("I need an LED display."))
        assert decision.ready is False
        assert decision.next_question
        assert set(decision.missing) >= {"environment", "purpose"}


class TestCase2Church:
    """Case 2：church → environment=indoor，来源 scenario_derived，不再问室内外。"""

    def test_church_settles_indoor(self):
        profile = _profile("We need a display for a church.")
        assert profile.environment == "indoor"
        assert profile.sources["environment"] == "scenario_derived"
        assert profile.status["environment"] == "confirmed"

        decision = check_recommendation_ready(profile)
        assert "environment" not in decision.missing
        assert not decision.ready  # 还缺安装方式 / 观看距离


class TestCase3Stadium:
    """Case 3：stadium → environment=outdoor（scenario_derived）。"""

    def test_stadium_settles_outdoor(self):
        profile = _profile("We need a screen for a stadium.")
        assert profile.environment == "outdoor"
        assert profile.sources["environment"] in CONFIRMED_SOURCES


class TestCase4InferredNeverOpensGate:
    """Case 4：算法估算（inferred）不能作为客户确认打开 Gate。"""

    def test_inferred_values_block(self):
        slots = extract_slots("indoor conference room, 5m viewing distance, fixed install")
        profile = RequirementProfile.from_slots(slots)  # 不传 explicit_keys → 全 inferred
        assert profile.sources.get("viewing_distance_m") == "inferred"
        decision = check_recommendation_ready(profile)
        assert decision.ready is False
        # purpose 只要"有来源"即可（它永远来自客户原话/LLM 对原话的抽取）；
        # 真正需要 provenance 的是环境、安装方式、观看距离这类工程条件。
        assert {"environment", "installation", "viewing_distance"} <= set(decision.missing)


class TestCase5DefaultDoesNotOpenGateAlone:
    """Case 5：系统默认值（default）可参与打分，但不能单独放行 Gate。"""

    def test_scene_default_installation_is_default_source(self):
        profile = _profile("indoor LED display for a conference room")
        assert profile.installation == "fixed"
        assert profile.sources["installation"] == "default"

        decision = check_recommendation_ready(profile)
        assert decision.ready is False
        assert "installation" in decision.missing

    def test_explicit_installation_still_wins(self):
        profile = _profile("indoor LED display for a conference room, fixed installation")
        assert profile.sources["installation"] == "explicit"


class TestCase6MultiTurnProfileAccumulates:
    """Case 6：多轮累积，不能因为重建而丢字段。"""

    def test_merge_keeps_previous_facts(self):
        profile = _profile("We need a display for a church.")
        profile = profile.merge(
            RequirementProfile.from_slots(
                extract_slots("fixed installation"),
                explicit_keys={"installation", "is_rental"},
            )
        )
        profile = profile.merge(
            RequirementProfile.from_slots(
                extract_slots("about 5m viewing distance"),
                explicit_keys={"viewing_distance_m"},
            )
        )

        assert profile.purpose == "church"
        assert profile.environment == "indoor"
        assert profile.installation == "fixed"
        assert profile.viewing_distance_m == pytest.approx(5.0)
        assert check_recommendation_ready(profile).ready is True

    def test_merge_slots_keeps_provenance_markers(self):
        base = extract_slots("church")
        assert base.get("_scenario_derived") == ["environment"]
        assert base.get("_default_slots") == ["installation"]

        merged = merge_slots(base, extract_slots("fixed installation"))
        # 客户后来说了固装 → default 标记被撤销
        assert "installation" not in (merged.get("_default_slots") or [])
        assert merged.get("installation") == "fixed"
        # 场景推出的环境标记保留
        assert merged.get("_scenario_derived") == ["environment"]


class TestCase8GateExceptionNeverRecommends:
    """Case 8：Gate 异常时必须安全降级 —— 不能因为 usage 有值就推荐。"""

    def test_gate_exception_blocks_solution(self, monkeypatch):
        import src.rag.readiness as readiness_mod
        import src.agents.sales.nodes.requirement as sales_req

        class _Response:
            content = '{"usage": "会议室", "additional_requirements": [], "ack": ""}'

        class _FakeChat:
            def __init__(self, *args, **kwargs):
                pass

            def invoke(self, *args, **kwargs):
                return _Response()

        monkeypatch.setattr(sales_req, "ChatOpenAI", _FakeChat)

        def _boom(*args, **kwargs):
            raise RuntimeError("gate exploded")

        monkeypatch.setattr(readiness_mod, "check_recommendation_ready", _boom)

        state = {
            "messages": [{"role": "user", "content": "会议室用 LED 屏"}],
            "current_message": "会议室用 LED 屏",
            "session_id": "gate-exception-session",
            "intent": "need_query",
            "next_action": "ask",
            "requirements": {"usage": "会议室", "location_type": "室内"},
            "additional_requirements": [],
            "should_generate_solution": False,
            "response": "",
        }
        result = sales_req.requirement_mining(state)

        assert result["should_generate_solution"] is False
        assert result["recommendation_gate"]["ready"] is False
        assert "gate_error" in result["recommendation_gate"]["reason"]


class TestCase9SolutionConsumesSalesProfile:
    """Case 9：Solution 直接消费 Sales 的 Profile，不再重新解析对话。"""

    def test_initial_state_uses_profile(self):
        from src.agents.solution.runner import SolutionAgentRunner

        profile = _profile("indoor LED display for a conference room, fixed, 5m")
        runner = SolutionAgentRunner.__new__(SolutionAgentRunner)  # 不初始化向量库
        runner.hybrid_search = None
        state = runner._build_initial_state(
            message="what about the cabinet size?",
            history=[{"role": "user", "content": "indoor LED display for a conference room"}],
            requirements=None,
            additional_requirements=None,
            profile=profile,
        )

        assert state["requirement_profile"] is profile
        assert state["requirement"]["indoor"] is True
        assert state["requirement"]["distance"] == "5米"
        assert state["requirement"]["display_type"] == "LED"


class TestLegacyAdapterIsOneWay:
    """M1：legacy requirements 只是 Profile 的投影（单向），不会反向污染 Profile。"""

    def test_adapter_projects_profile(self):
        from src.models.legacy_adapter import profile_to_legacy, rebuild_legacy_view

        profile = _profile("outdoor advertising screen, 20m viewing distance, fixed")
        legacy = profile_to_legacy(profile)
        assert legacy["usage"] == "advertising"
        assert legacy["location_type"] == "室外"
        assert legacy["outdoor"] is True
        assert legacy["distance"] == "20米"
        assert legacy.get("is_rental") is False

        # 旧字典里塞进脏值，重建后必须被 Profile 覆盖
        dirty = {"usage": "meeting room", "indoor": True, "size": "999米x999米"}
        rebuild_legacy_view(dirty, profile)
        assert dirty["usage"] == "advertising"
        assert dirty["indoor"] is False
        assert "size" not in dirty

    def test_size_hint_survives_projection(self):
        from src.models.legacy_adapter import profile_to_legacy

        profile = _profile("129,2cm")
        legacy = profile_to_legacy(profile)
        assert legacy["screen_size_hint_mm"] == pytest.approx(1292.0)
