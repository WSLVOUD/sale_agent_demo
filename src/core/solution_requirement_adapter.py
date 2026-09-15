"""
Solution Agent Requirement Adapter - 适配器

职责：
1. 将 Solution Agent 的 requirement extraction 迁移到使用 RequirementExtractor
2. 避免重复 LLM extraction
3. 保持现有接口兼容
"""

import logging
from typing import Dict, Any, Optional
from src.core.requirement_extractor import get_requirement_extractor
from src.models.requirement import RequirementProfile
from src.models.legacy_adapter import profile_to_solution_requirement

logger = logging.getLogger(__name__)


class SolutionRequirementAdapter:
    """Solution Agent 需求理解适配器"""

    def __init__(self):
        self.extractor = get_requirement_extractor()

    def extract_from_message(
        self,
        message: str,
        previous_profile: Optional[RequirementProfile] = None,
        skip_understand: bool = False,
    ) -> Dict[str, Any]:
        """
        从消息中提取需求，返回兼容格式。

        Args:
            message: 客户消息
            previous_profile: 前一轮的 RequirementProfile
            skip_understand: 是否跳过理解（性能优化）

        Returns:
            {
                "profile": RequirementProfile,
                "requirement": dict (legacy format for solution),
                "info_sufficient": bool,
                "missing_info": list,
            }
        """
        # 如果 skip_understand=True，直接使用前一轮 profile
        if skip_understand and previous_profile:
            logger.info("Skip understand optimization: reusing previous profile")
            profile = previous_profile
        else:
            # 使用统一 Extractor
            profile = self.extractor.extract(message, previous_profile)

        # 转换为 Solution Agent 的 legacy 格式
        legacy_requirement = profile_to_solution_requirement(profile)

        # 判断信息是否充足
        info_sufficient = profile.is_recommendation_ready()
        missing_info = profile.missing_slots() if not info_sufficient else []

        return {
            "profile": profile,
            "requirement": legacy_requirement,
            "info_sufficient": info_sufficient,
            "missing_info": missing_info,
        }

    def build_retrieval_query(self, profile: RequirementProfile) -> str:
        """
        从 RequirementProfile 构建检索查询。

        Args:
            profile: 需求档案

        Returns:
            检索查询字符串
        """
        from src.rag.query_understanding import build_retrieval_query
        from src.models.legacy_adapter import profile_to_solution_requirement

        # 转换为 slots 格式
        slots = profile.to_slots()
        
        # 使用现有的 build_retrieval_query
        return build_retrieval_query(slots, fallback=f"{profile.display_type or 'LED'} display")

    def get_technical_parameters(self, profile: RequirementProfile) -> Dict[str, Any]:
        """
        从 RequirementProfile 获取技术参数（用于 parameter_inference）。

        Args:
            profile: 需求档案

        Returns:
            技术参数字典
        """
        return profile.to_facts()


# 全局单例
_adapter = None


def get_solution_requirement_adapter() -> SolutionRequirementAdapter:
    """获取 Solution Requirement Adapter 单例"""
    global _adapter
    if _adapter is None:
        _adapter = SolutionRequirementAdapter()
    return _adapter
