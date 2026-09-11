"""
规则引擎模块
判断是否有足够信息进行推荐
"""
from typing import Dict, Any, List, TypedDict
import logging

logger = logging.getLogger(__name__)


# Minimum required fields for a recommendation
REQUIRED_FIELDS = [
    ("indoor", "indoor or outdoor usage"),
    ("outdoor", "indoor or outdoor usage"),
    ("distance", "viewing distance"),
]


class RuleEngineState(TypedDict):
    """规则引擎状态"""
    requirement: Dict[str, Any]
    info_sufficient: bool
    missing_info: List[str]
    next_action: str


def check_missing_info(requirement: Dict[str, Any]) -> List[str]:
    """Check which required information is missing."""
    missing = []

    # Must have either indoor or outdoor
    if not requirement.get("indoor") and not requirement.get("outdoor"):
        missing.append("使用环境（室内/户外）")

    has_env = requirement.get("indoor") or requirement.get("outdoor")
    has_size = bool(requirement.get("size"))
    has_purpose = bool(requirement.get("purpose"))
    has_distance = bool(requirement.get("distance"))

    # Viewing distance is required when the usage environment is known.
    if not has_distance and has_env:
        missing.append("观看距离")
    elif not has_distance and not has_size and not has_purpose:
        missing.append("观看距离或尺寸或使用场景")

    return missing


def rule_engine_node(state: RuleEngineState) -> RuleEngineState:
    """
    Decide whether the current recommendation request has enough context.
    """
    requirement = state.get("requirement", {})

    missing_info = check_missing_info(requirement)
    info_sufficient = not missing_info

    logger.info(
        "Info sufficient: %s, missing=%s, requirement=%s",
        info_sufficient,
        missing_info,
        requirement,
    )

    return {
        **state,
        "info_sufficient": info_sufficient,
        "missing_info": missing_info,
        "next_action": "classify" if info_sufficient else "ask"
    }


def should_recommend(requirement: Dict[str, Any]) -> bool:
    """
    Quick check if we can make a recommendation.
    Used for deciding next action in the graph.
    """
    missing = check_missing_info(requirement)
    return len(missing) == 0
