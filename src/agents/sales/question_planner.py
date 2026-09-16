"""
Phase 7：Sales Agent 需求采集规划器。

目标（来自计划文档「十一、Phase 7」）：
    让 Sales Agent 先问清楚客户需求，再进入产品推荐；**一次只问一个高价值问题**。

采集顺序：
    1. 产品类型 → 2. 室内/室外 → 3. 使用场景 → 4. 固装/租赁
    → 5. 观看距离 → 6. 屏幕尺寸 → 7. 预算 → 8. 特殊要求

禁止事项：不要一上来就问点间距（P2.5）、亮度、刷新率 —— 这些应由
「使用场景 + 观看距离」通过 Phase 5 的规则推断出来。

本模块是"问什么"的唯一来源：requirement_mining 用它决定是否继续追问，
script_generator 用它生成追问话术（模板化，不调用 LLM）。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.models.requirement import RequirementProfile

logger = logging.getLogger(__name__)

# 规划器槽位名 → RequirementProfile 的规范槽位名（Unknown 状态机用这套命名）
_SLOT_ALIAS: Dict[str, str] = {
    "viewing_distance_m": "viewing_distance",
    "target_size": "size",
    "budget_level": "budget",
}

# 采集顺序（值为提问模板；'en' 与 'zh' 两套）
QUESTION_PLAN: tuple[tuple[str, str, str], ...] = (
    (
        "environment",
        "Will the display be installed indoors or outdoors?",
        "这块屏是装在室内还是室外？",
    ),
    (
        "purpose",
        "What will you mainly use it for — meeting room, classroom, retail, advertising, or something else?",
        "主要用在什么场景？会议室、教室、商场零售、广告，还是其他？",
    ),
    (
        "installation",
        "Is this a fixed installation, or do you need it for rental/events?",
        "是固定安装，还是租赁/活动用？",
    ),
    (
        "viewing_distance_m",
        "Roughly how far will viewers typically stand from the screen?",
        "观众通常离屏幕大概多远？",
    ),
    (
        "target_size",
        "Do you already have a target screen size (width x height)?",
        "有大概的目标尺寸吗（宽 x 高）？",
    ),
    (
        "budget_level",
        "Do you have a budget level in mind — entry, mid-range, or premium?",
        "预算大概在什么档位？入门、中端还是高端？",
    ),
)

# 这些槽位一旦缺失就不允许进入推荐（"不知道用在哪"无法选型）
BLOCKING_SLOTS = ("environment", "purpose")


def _slot_filled(profile: RequirementProfile, slot: str) -> bool:
    """该槽位是否"不用再问了"。

    Phase 17：客户已经明确表示不知道（问满两次 / 主动跳过）的字段 → 视为
    "不用再问"，直接跳到下一个，绝不重复追问同一项。
    """
    canonical = _SLOT_ALIAS.get(slot, slot)
    if profile.is_unknown(canonical):
        return True
    if slot == "target_size":
        return profile.has_target_size
    if slot == "viewing_distance_m":
        return profile.viewing_distance_m is not None or bool(profile.pixel_pitch_mm)
    value = getattr(profile, slot, None)
    return value not in (None, "", [], {})


def plan_next_question(
    profile: RequirementProfile,
    language: str = "en",
    seed: int = 0,
) -> Optional[Dict[str, Any]]:
    """返回下一个应该问的问题（没有缺失则返回 None）。

    问法来自 ``src/rag/readiness.py`` 的 ``QUESTION_VARIANTS``：
    同一槽位有多种自然说法，按 ``seed`` 轮换，问到的内容始终一致。
    """
    if profile is None:
        return None
    from src.rag.readiness import question_for
    from src.models.requirement import SLOT_PRIORITY

    for slot, question_en, question_zh in QUESTION_PLAN:
        if _slot_filled(profile, slot):
            continue
        question = question_for(slot, language, seed) or (
            question_zh if language == "zh" else question_en
        )
        return {
            "slot": slot,
            "question": question,
            "blocking": slot in BLOCKING_SLOTS,
            # Phase 17：字段优先级（HIGH → MEDIUM → LOW）
            "priority": SLOT_PRIORITY.get(_SLOT_ALIAS.get(slot, slot), "MEDIUM"),
            "missing": profile.missing_slots(),
        }
    return None


def should_ask_before_recommend(profile: RequirementProfile) -> bool:
    """是否必须先追问再推荐。

    判定规则集中在 ``src/rag/readiness.py``（v2.0 Recommendation Ready Gate）：
      - 客户已点名型号 / 系列，或明确给出点间距 / 亮度 → 直接推荐
      - 室内外 + 场景 + 安装方式 + 观看距离齐备 → 直接推荐
      - 否则先问一个最关键的问题
    """
    if profile is None:
        return True
    from src.rag.readiness import check_recommendation_ready

    return not check_recommendation_ready(profile).ready


def missing_slots(profile: RequirementProfile) -> List[str]:
    """按采集优先级返回缺失槽位。"""
    if profile is None:
        return [slot for slot, _, _ in QUESTION_PLAN]
    return [slot for slot, _, _ in QUESTION_PLAN if not _slot_filled(profile, slot)]


__all__ = [
    "BLOCKING_SLOTS",
    "QUESTION_PLAN",
    "missing_slots",
    "plan_next_question",
    "should_ask_before_recommend",
]
