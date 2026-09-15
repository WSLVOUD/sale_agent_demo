"""
Sales Agent Requirement Adapter - 适配器

职责：
1. 将 Sales Agent 的 requirement_mining 迁移到使用 RequirementExtractor
2. 保持现有接口兼容
3. 逐步删除重复的场景分类和环境推断代码
"""

import logging
from typing import Dict, Any, Optional
from src.core.requirement_extractor import get_requirement_extractor
from src.models.requirement import RequirementProfile
from src.models.legacy_adapter import profile_to_legacy

logger = logging.getLogger(__name__)


class SalesRequirementAdapter:
    """Sales Agent 需求理解适配器"""

    def __init__(self):
        self.extractor = get_requirement_extractor()

    def extract_from_message(
        self,
        message: str,
        previous_profile: Optional[RequirementProfile] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        从消息中提取需求，返回兼容格式。

        Args:
            message: 客户消息
            previous_profile: 前一轮的 RequirementProfile
            session_id: 会话 ID（用于日志）

        Returns:
            {
                "profile": RequirementProfile,
                "requirements": dict (legacy format),
                "pending_question": str,
                "pending_slot": str,
                "should_generate_solution": bool,
            }
        """
        # 使用统一 Extractor
        profile = self.extractor.extract(message, previous_profile)

        # 转换为 legacy 格式（兼容现有代码）
        legacy_requirements = profile_to_legacy(profile)

        # 判断是否需要继续提问
        missing_slots = profile.missing_slots()
        pending_question = ""
        pending_slot = ""

        if missing_slots:
            # 取第一个缺失的槽位
            pending_slot = missing_slots[0]
            pending_question = self._generate_question_for_slot(pending_slot)
            logger.info(f"[{session_id}] Missing slot: {pending_slot}, asking: {pending_question}")

        # 判断是否可以推荐（Recommendation Ready Gate）
        should_recommend = profile.is_recommendation_ready()

        return {
            "profile": profile,
            "requirements": legacy_requirements,
            "pending_question": pending_question,
            "pending_slot": pending_slot,
            "should_generate_solution": should_recommend,
        }

    def _generate_question_for_slot(self, slot: str) -> str:
        """
        为缺失的槽位生成问题。

        TODO: 这部分应该与现有的 question_planner 集成
        """
        questions = {
            "display_type": "What type of display are you looking for? (LED/LCD/IFP)",
            "environment": "Will it be used indoors or outdoors?",
            "purpose": "What is the screen for? (conference, retail, advertising, etc.)",
            "installation": "Will this be a fixed installation or rental?",
            "viewing_distance_m": "What is the viewing distance?",
            "target_size": "What size screen do you need?",
        }
        return questions.get(slot, f"Could you provide the {slot}?")


# 全局单例
_adapter = None


def get_sales_requirement_adapter() -> SalesRequirementAdapter:
    """获取 Sales Requirement Adapter 单例"""
    global _adapter
    if _adapter is None:
        _adapter = SalesRequirementAdapter()
    return _adapter
