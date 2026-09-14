"""
Recommendation Service —— 统一的推荐入口（v1.0 排查计划 Phase 6）。

职责：
    所有"要推荐产品"的调用都必须经过这里，而不是直接调用
    ``RecommendationEngine``。本服务强制先过 ``Recommendation Ready Gate``：

        Recommendation Ready Gate
                ├── FALSE → NEED_CLARIFICATION（不推荐，返回缺失字段）
                └── TRUE  → Recommendation Engine → RECOMMENDED

这样即使 Sales / Solution / Fast / Normal / Agent / API 某条路径忘了判 Gate，
只要它走本服务，也不会出现"需求不足就推荐"。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.models.requirement import RequirementProfile
from src.rag.readiness import check_recommendation_ready

logger = logging.getLogger(__name__)


class RecommendationService:
    """Gate → Engine 的统一推荐服务。"""

    def __init__(self, engine=None):
        if engine is None:
            from src.rag.recommendation_engine import RecommendationEngine

            engine = RecommendationEngine()
        self.engine = engine

    def recommend(
        self,
        profile: RequirementProfile,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """先过 Gate，再推荐。"""
        gate = check_recommendation_ready(profile)
        if not gate.ready:
            return {
                "recommendation_status": "NEED_CLARIFICATION",
                "missing_fields": gate.missing,
                "next_question": gate.next_question,
                "gate": gate.to_dict(),
                "recommendations": [],
                "candidate_count": 0,
                "rejected_count": 0,
                "violations": [],
            }

        result = self.engine.recommend(profile=profile, top_k=top_k, require_ready=True)
        result["recommendation_status"] = "RECOMMENDED"
        result["missing_fields"] = []
        result["gate"] = gate.to_dict()
        return result


__all__ = ["RecommendationService"]
