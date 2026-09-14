"""
v2.0 Phase 7：Rerank 作为"证据排序 / 冲突检测"层的测试。

核心约束：
  - 不调用 LLM，结果确定
  - 顺序跟随 Recommendation Engine，LLM 不参与产品选择
  - 冲突证据直接丢弃，硬约束不可被突破
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.recommendation_engine import RecommendationEngine  # noqa: E402
from src.rag.rerank import rank_evidence, rerank_node  # noqa: E402


def _evidence(model, text, **metadata):
    meta = {"model": model, "product_category": "display"}
    meta.update(metadata)
    return {"id": model, "text": text, "metadata": meta}


OUTDOOR_PROFILE = RequirementProfile.from_slots({
    "environment": "outdoor", "purpose": "advertising",
    "installation": "fixed", "viewing_distance_m": 20,
})


class TestEvidenceRerank:

    def test_environment_conflict_is_dropped(self):
        items = [
            _evidence("TW21-3216-P2.5", "indoor meeting room LED", indoor=True, outdoor=False),
            _evidence("TW21-OD-P6", "outdoor advertising LED", indoor=False, outdoor=True),
        ]
        kept, dropped = rank_evidence(items, {}, OUTDOOR_PROFILE, limit=3)
        assert [item["metadata"]["model"] for item in kept] == ["TW21-OD-P6"]
        assert dropped and dropped[0]["reason"] == "环境与需求冲突"

    def test_non_display_evidence_is_dropped(self):
        items = [
            _evidence("", "floor stand bracket", product_category="mount"),
            _evidence("TW21-OD-P6", "outdoor LED", indoor=False, outdoor=True),
        ]
        kept, dropped = rank_evidence(items, {}, OUTDOOR_PROFILE, limit=3)
        assert len(kept) == 1
        assert any(item["reason"] == "非显示产品" for item in dropped)

    def test_order_follows_engine_selection(self):
        engine = RecommendationEngine()
        selection = engine.recommend(profile=OUTDOOR_PROFILE)
        order = [rec["model"] for rec in selection["recommendations"]]

        # 故意把证据顺序打乱，并且让被选中的型号排在后面
        items = [
            _evidence("TW21-OD-P8", "outdoor LED", indoor=False, outdoor=True),
            _evidence(order[0], "outdoor LED", indoor=False, outdoor=True),
            _evidence(order[1], "outdoor LED", indoor=False, outdoor=True),
        ]
        kept, _ = rank_evidence(items, selection, OUTDOOR_PROFILE, limit=3)
        assert [item["metadata"]["model"] for item in kept][:2] == [order[0], order[1]]

    def test_rerank_node_is_deterministic_and_llm_free(self, monkeypatch):
        import src.rag.rerank as rerank_module

        called = {"count": 0}

        class _Boom:
            def invoke(self, *args, **kwargs):
                called["count"] += 1
                raise AssertionError("rerank 不应调用 LLM")

        monkeypatch.setattr(
            rerank_module, "get_llm", lambda *a, **k: _Boom(), raising=False
        )
        monkeypatch.setattr(
            rerank_module, "ChatOpenAI", lambda *a, **k: _Boom(), raising=False
        )

        state = {
            "products": [
                _evidence("TW21-OD-P6", "outdoor LED", indoor=False, outdoor=True),
                _evidence("TW21-3216-P2.5", "indoor LED", indoor=True, outdoor=False),
            ],
            "recommendation_result": {},
            "requirement_profile": OUTDOOR_PROFILE,
        }
        first = rerank_node(dict(state))
        second = rerank_node(dict(state))
        assert called["count"] == 0
        assert [p["metadata"]["model"] for p in first["products"]] == \
               [p["metadata"]["model"] for p in second["products"]]

    def test_hard_constraints_survive_rerank(self):
        """Rerank 之后不允许残留环境冲突证据"""
        engine = RecommendationEngine()
        selection = engine.recommend(profile=OUTDOOR_PROFILE)
        items = [
            _evidence("TW21-3216-P2.5", "indoor LED", indoor=True, outdoor=False),
            _evidence("TW31-COB-P1.2H", "indoor COB LED", indoor=True, outdoor=False),
            _evidence("TW21-OD-P6", "outdoor LED", indoor=False, outdoor=True),
        ]
        out = rerank_node({
            "products": items,
            "recommendation_result": selection,
            "requirement_profile": OUTDOOR_PROFILE,
        })
        for product in out["products"]:
            assert product["metadata"]["outdoor"] is True
