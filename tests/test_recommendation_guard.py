"""
排查计划 v1.0 Phase 7 / 13 / 15：防绕过与推荐泄漏回归测试。

统一入口是 ``RecommendationService``：任何要推荐的路径都必须先过 Gate。
本测试直接验证"绕过 Gate 也推荐不了"，以及"Gate 未通过时不得泄漏型号"。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.recommendation_engine import RecommendationEngine  # noqa: E402
from src.rag.recommendation_service import RecommendationService  # noqa: E402


def _profile(slots: dict, confirmed: bool = True) -> RequirementProfile:
    """构造"客户明确说出"的档案；confirmed=False 模拟纯推断值。"""
    return RequirementProfile.from_slots(
        slots, explicit_keys=set(slots) if confirmed else set()
    )


@pytest.fixture(scope="module")
def service():
    return RecommendationService()


class TestRecommendationServiceGate:
    """排查计划 15.1：Gate 测试"""

    def test_indoor_scene_blocks(self, service):
        result = service.recommend(_profile({"environment": "indoor", "purpose": "conference"}))
        assert result["recommendation_status"] == "NEED_CLARIFICATION"
        assert result["recommendations"] == []

    def test_outdoor_scene_blocks(self, service):
        result = service.recommend(_profile({"environment": "outdoor", "purpose": "advertising"}))
        assert result["recommendation_status"] == "NEED_CLARIFICATION"

    def test_scene_installation_distance_partial_blocks(self, service):
        # 环境+场景+固装（缺视距）→ 不推荐
        result = service.recommend(_profile({
            "environment": "indoor", "purpose": "conference", "installation": "fixed",
        }))
        assert result["recommendation_status"] == "NEED_CLARIFICATION"
        assert "viewing_distance" in result["missing_fields"]

        # 环境+场景+视距（缺固装租赁）→ 不推荐
        result = service.recommend(_profile({
            "environment": "indoor", "purpose": "conference", "viewing_distance_m": 5,
        }))
        assert result["recommendation_status"] == "NEED_CLARIFICATION"
        assert "installation" in result["missing_fields"]

    def test_shortcuts_allow_recommendation(self, service):
        for slots in (
            {"pixel_pitch_mm": 2.5},
            {"brightness_min_nit": 600},
            {"model": "TW21-3216-P2.5"},
            {"series_id": "TW11-OD"},
        ):
            result = service.recommend(_profile(slots))
            assert result["recommendation_status"] == "RECOMMENDED", slots
            assert result["recommendations"]

    def test_full_combo_allows_recommendation(self, service):
        result = service.recommend(_profile({
            "environment": "indoor", "purpose": "conference",
            "installation": "fixed", "viewing_distance_m": 5,
        }))
        assert result["recommendation_status"] == "RECOMMENDED"
        assert result["recommendations"][0]["model"].startswith("TW")


class TestProvenanceNeverOpensGate:
    """排查计划 15.2：inferred 参数不能打开 Gate"""

    def test_inferred_installation_and_distance_block(self, service):
        result = service.recommend(_profile(
            {
                "environment": "indoor", "purpose": "conference",
                "installation": "fixed", "viewing_distance_m": 5,
            },
            confirmed=False,  # 全部是系统推断出来的，客户没明说
        ))
        assert result["recommendation_status"] == "NEED_CLARIFICATION"

    def test_obvious_scene_settles_environment(self):
        """客户说"会议室 / 教堂"这种一眼室内的场景 → 环境直接确定，不再追问室内外。

        （客户反馈：说了 church 还问"室内还是室外"很傻。旧策略要求环境必须由
        客户明说，现在只对"一眼能定"的场景放宽。）
        """
        from src.rag.query_understanding import extract_slots

        for message in ("会议室用", "church", "在展厅放一块屏", "户外广告牌"):
            slots = extract_slots(message)
            assert slots.get("environment") in ("indoor", "outdoor"), message
            assert "environment" not in (slots.get("_inferred_slots") or []), message
            profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
            # M2 四态：客户明说室内外 → explicit；场景直接判定 → scenario_derived
            assert profile.sources.get("environment") in ("explicit", "scenario_derived"), message
            assert profile.status["environment"] == "confirmed", message
            # 场景默认固装属于 default：参与打分，但不单独放行 Gate
            assert profile.sources.get("installation") == "default", message

    def test_ambiguous_scene_still_asks_environment(self):
        """舞台 / 演唱会 / 租赁室内外都可能 → 环境仍然必须问客户。"""
        from src.rag.query_understanding import extract_slots
        from src.rag.readiness import check_recommendation_ready

        for message in ("舞台演出用", "concert", "rental event"):
            slots = extract_slots(message)
            assert not slots.get("environment"), message
            profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
            assert "environment" in check_recommendation_ready(profile).missing, message


class TestEngineSecondGuard:
    """排查计划 15.3：引擎第二道保险"""

    def test_engine_rejects_incomplete_when_require_ready(self):
        engine = RecommendationEngine()
        result = engine.recommend(
            profile=_profile({"environment": "indoor", "purpose": "conference"}),
            require_ready=True,
        )
        assert result["recommendation_status"] == "NEED_CLARIFICATION"
        assert result["recommendations"] == []

    def test_engine_default_still_usable_for_direct_scoring(self):
        """直接调用引擎（未要求 ready）仍可用于打分，不破坏既有调用方"""
        engine = RecommendationEngine()
        result = engine.recommend(
            profile=_profile({
                "environment": "indoor", "purpose": "conference",
                "viewing_distance_m": 5,
            })
        )
        assert result.get("recommendation_status") != "NEED_CLARIFICATION"
        assert result["recommendations"]


class TestNoRecommendationLeak:
    """排查计划 15：Gate 未通过时不得泄漏型号"""

    def test_blocked_recommend_node_has_no_model(self):
        from src.agents.solution.nodes.recommend import recommend_node

        state = {
            "requirement": {},
            "requirement_profile": _profile({"environment": "indoor", "purpose": "conference"}),
            "products": [],
            "messages": [{"role": "user", "content": "会议室用"}],
            "current_message": "会议室用",
            "additional_requirements": [],
        }
        result = recommend_node(state)
        assert result["next_action"] == "clarify"
        assert result["products"] == []
        answer = result["recommendation"]
        assert "TW" not in answer.upper()
        assert not any(ch in answer for ch in ("P2.5", "P1.8", "P3.0"))
