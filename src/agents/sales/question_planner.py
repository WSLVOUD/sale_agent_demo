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
        # 客户口径：问场景时不举例，直接问问题
        "What will you mainly use it for?",
        "主要用在什么场景？",
    ),
    (
        "installation",
        "Is this a fixed installation, or do you need it for rental/events?",
        "是固定安装，还是租赁/活动用？",
    ),
    (
        "pixel_pitch_mm",
        "Do you have a pixel pitch in mind — for example P2.5, P3 or P5?",
        "您对点间距有要求吗？比如 P2.5、P3 或 P5。",
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


# ── v2.1：字段状态驱动的"要不要再问" ─────────────────────────────────────
# CONFIRMED / INFERRED / DELEGATED / DECLINED / DEFERRED → 一律不再问
# UNKNOWN → 还可以用"降低门槛"的问法再问一次
# MISSING → 正常问
_SKIP_STATES = frozenset({"CONFIRMED", "INFERRED", "DELEGATED", "DECLINED", "DEFERRED"})


def slot_state(profile: RequirementProfile, slot: str) -> str:
    """槽位当前状态（大写；见 RequirementProfile.field_decision）。"""
    canonical = _SLOT_ALIAS.get(slot, slot)
    try:
        return profile.field_decision(canonical)
    except Exception:  # pragma: no cover - 防御式
        return "MISSING"


def _slot_filled(profile: RequirementProfile, slot: str) -> bool:
    """该槽位是否"不用再问了"。

    v2.1：由字段状态决定 ——
      已确认 / 系统推断 / 客户授权 AI 决定 / 客户拒绝 / 已延后 → 不用再问；
      客户说不知道（UNKNOWN）→ 换成低门槛问法再问一次；
      从没问过（MISSING）→ 可以问。
    """
    state = slot_state(profile, slot)
    if state in _SKIP_STATES:
        return True
    if state == "UNKNOWN":
        return False
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

    ⚠️ 架构收口（Phase 2）：QuestionPlanner 只是**候选生成器**。
    这个函数现在等价于 ``candidate_questions(...)[0]``，保留只为兼容旧调用；
    最终问哪一项由 DialoguePolicy（``dialogue_action``）决定，
    Planner **不得**覆盖 Policy，也不得直接写 ``state["response"]``。

    问法来自 ``src/rag/readiness.py`` 的 ``QUESTION_VARIANTS``：
    同一槽位有多种自然说法，按 ``seed`` 轮换，问到的内容始终一致。
    """
    candidates = candidate_questions(profile, language=language, seed=seed)
    return candidates[0] if candidates else None


def candidate_questions(
    profile: RequirementProfile,
    language: str = "en",
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """本轮**候选**问题（按该问的先后排序；Planner 只提供候选，不做最终决定）。

    每个候选都带齐决策层需要的信息：slot / question / priority / blocking /
    state / easier / reason。DialoguePolicy 从中挑一个（并允许换成别的槽位）。
    """
    if profile is None:
        return []
    from src.rag.readiness import question_for
    from src.models.requirement import SLOT_PRIORITY

    candidates: List[Dict[str, Any]] = []

    # ── v2.5+++（计划 §8）：Environment Hard Priority ────────────────────
    # 室内外还没定（也没被明显场景推断出来）→ 第一项问题必须是 Indoor / Outdoor；
    # 已经确定 / 已推断 / 客户答过 → 跳过，剩下的问题再按原有顺序问。
    from src.dialogue.question_flow import environment_should_ask_now

    if environment_should_ask_now(profile):
        state = slot_state(profile, "environment")
        easier = state == "UNKNOWN" and profile.ask_count("environment") >= 1
        question = question_for("environment", language, seed, easier=easier)
        if question:
            candidates.append({
                "slot": "environment",
                "question": question,
                "blocking": True,
                "state": state,
                "easier": easier,
                "priority": SLOT_PRIORITY.get("environment", "HIGH"),
                "missing": profile.missing_slots(),
                "reason": "environment_hard_gate",
            })

    for slot, question_en, question_zh in QUESTION_PLAN:
        if _slot_filled(profile, slot):
            continue
        state = slot_state(profile, slot)
        # 客户说过"不知道" → 第二次用"降低门槛"的问法（给区间 / 二选一）
        easier = state == "UNKNOWN" and profile.ask_count(_SLOT_ALIAS.get(slot, slot)) >= 1
        question = question_for(slot, language, seed, easier=easier) or (
            question_zh if language == "zh" else question_en
        )
        candidates.append({
            "slot": slot,
            "question": question,
            "blocking": slot in BLOCKING_SLOTS,
            "state": state,
            "easier": easier,
            # Phase 17：字段优先级（HIGH → MEDIUM → LOW）
            "priority": SLOT_PRIORITY.get(_SLOT_ALIAS.get(slot, slot), "MEDIUM"),
            "missing": profile.missing_slots(),
            "reason": "question_plan",
        })
    return candidates


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
    """按采集优先级返回**还值得问**的槽位（已决策/已延后的不再列）。"""
    if profile is None:
        return [slot for slot, _, _ in QUESTION_PLAN]
    return [slot for slot, _, _ in QUESTION_PLAN if not _slot_filled(profile, slot)]


__all__ = [
    "BLOCKING_SLOTS",
    "QUESTION_PLAN",
    "candidate_questions",
    "missing_slots",
    "plan_next_question",
    "should_ask_before_recommend",
    "slot_state",
]
