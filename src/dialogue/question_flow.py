"""v2.4：提问顺序随机化 + "一轮走完再回头问硬性条件"。

客户口径（2026-09-20）：

  1. 提问顺序**随机**（不再按固定优先级一个一个来）；
  2. 客户说"不知道"或答非所问 / 没回答 → 这个问题**本轮不再重复问**，换下一个；
  3. 全部问题随机问完一遍之后，才回头问**硬性条件**里还缺的，
     并且要**说明为什么需要知道这个**；
  4. 硬性条件复问仍然遵守"每个字段最多两次接触"：第二次还拿不到 → DEFERRED，
     推荐照常降级（室内外拿不到 → BLOCKED）；
  5. 只有客户**明确要推荐**（recommend / 帮我推荐 / 选一款…）时，才跳过随机轮
     直接进"硬性条件复问 / 推荐"；光是把硬性条件凑齐不触发推荐。

随机做法：复用 `question_order.shuffled_slots()` —— 按会话 id 播种洗牌，
同一会话内顺序**稳定可复现**，不同会话之间顺序**各不相同**（顺序只有这一个出口）。
"""
from __future__ import annotations

from typing import Any, List, Optional

from .question_order import shuffled_slots
from .question_planner import QuestionPlan

# 参与提问的槽位（内容类型按客户口径不主动问，只记录）
ASK_POOL: tuple[str, ...] = (
    "environment",      # 室内外（硬性）
    "purpose",          # 使用场景（软）
    "installation",     # 固装 / 租赁（硬性）
    "price_preference", # 价位取向（软）
    "pixel_pitch",      # P 值（硬性）
    "viewing_distance", # 观看距离（硬性，用于反推 P 值）
    "size",             # 尺寸（硬性）
)

HARD_SLOTS: tuple[str, ...] = (
    "environment", "installation", "pixel_pitch", "viewing_distance", "size",
)

# 复问硬性条件时的"为什么需要知道"（英文，一句话）
HARD_WHY: dict[str, str] = {
    "environment": "indoor and outdoor screens use different cabinets and brightness",
    "installation": "fixed and rental cabinets are built differently",
    "pixel_pitch": "the pitch decides how sharp the image looks from where people sit",
    "viewing_distance": "the viewing distance tells me which pixel pitch is enough",
    "size": "the screen size decides the cabinet and module layout",
}

# 已经"有结论"的状态（不再问）
_SETTLED = ("CONFIRMED", "INFERRED", "DELEGATED", "DECLINED", "DEFERRED", "CONFLICT")
_ASK_ACTIONS = ("ask", "ask_easier", "ask_later")


def random_order(session_id: str, slots: tuple[str, ...] = ASK_POOL) -> List[str]:
    """会话内的随机顺序（稳定可复现，顺序实现见 question_order.py）。"""
    return shuffled_slots(slots, seed=session_id)


def _settled(profile: Any, slot: str) -> bool:
    if profile is None:
        return False
    try:
        return str(profile.field_decision(slot) or "") in _SETTLED
    except Exception:  # pragma: no cover - 防御式
        return False


def _actions(profile: Any) -> dict:
    """当前每个槽位的 Action（与 Gate 用同一套字段策略 + 跨槽位规则）。

    这样"该不该问"只有一份判断：点间距能由观看距离推导时不会被问、
    客户给了 P 值时不会再问观看距离……
    """
    from src.rag.field_policy import (
        RECOMMENDATION_SLOTS,
        apply_cross_slot_rules,
        field_action,
    )

    actions = {slot: field_action(profile, slot) for slot in RECOMMENDATION_SLOTS}
    try:
        apply_cross_slot_rules(profile, actions)
    except Exception:  # pragma: no cover - 防御式
        pass
    return actions


def _askable(profile: Any, slot: str, actions: dict) -> bool:
    """这个槽位现在属于"该问"吗（Gate 的 Action 说了算）。"""
    if slot not in actions:
        return True
    return str(actions.get(slot) or "") in _ASK_ACTIONS


def _asked(profile: Any, slot: str) -> int:
    if profile is None:
        return 0
    try:
        return int(profile.ask_count(slot) or 0)
    except Exception:  # pragma: no cover - 防御式
        return 0


def _max_asks() -> int:
    from src.models.requirement import MAX_ASKS_PER_SLOT

    return int(MAX_ASKS_PER_SLOT)


def pass1_pending(profile: Any, session_id: str = "") -> List[str]:
    """随机轮里"还没问过、也还没答案"的槽位。

    v2.5++++（计划 §8）：先按会话随机取顺序，再按**业务价值**排序 ——
    随机只在价值相同的槽位之间生效（不会再随机到不重要的字段）。
    """
    actions = _actions(profile)
    slots = [
        slot for slot in random_order(session_id)
        if _askable(profile, slot, actions)
        and not _settled(profile, slot)
        and _asked(profile, slot) == 0
    ]
    if len(slots) <= 1:
        return slots
    try:
        from .policy import rank_slots_by_value

        return rank_slots_by_value(slots, profile, session_id=session_id)
    except Exception:  # pragma: no cover - 防御式
        return slots


def pass1_complete(profile: Any) -> bool:
    """所有问题都随机问过一遍了（或者已有结论）。"""
    actions = _actions(profile)
    return not any(
        _askable(profile, slot, actions)
        and not _settled(profile, slot)
        and _asked(profile, slot) == 0
        for slot in ASK_POOL
    )


def hard_recap_pending(profile: Any, session_id: str = "") -> List[str]:
    """第一轮没拿到答案的**硬性条件**（还能再问一次的那些）。"""
    limit = _max_asks()
    actions = _actions(profile)
    return [
        slot for slot in random_order(session_id, HARD_SLOTS)
        if _askable(profile, slot, actions)
        and not _settled(profile, slot)
        and _asked(profile, slot) < limit
    ]


def why_for(slot: str) -> str:
    return HARD_WHY.get(str(slot or ""), "")


def environment_needs_asking(profile: Any) -> bool:
    """v2.5+++（计划 §8）：环境是否**还没定**、必须优先问 Indoor / Outdoor。

    · 客户明说过室内/户外        → 已定，不问
    · 明显场景可判定（教堂/会议室/户外广告…）→ 已定，不问（业务规则保留）
    · 客户已经答过一次且档案有值 → 已定，不问
    · 客户明确拒绝 / 已延后      → 不再问（交给 Gate 的 BLOCK / 降级）
    其余情况（MISSING / UNKNOWN）→ 必须作为**第一项**需求问题。
    """
    if profile is None:
        return False
    try:
        from src.rag.readiness import environment_settled

        if environment_settled(profile):
            return False
    except Exception:  # pragma: no cover - 防御式
        if getattr(profile, "environment", None):
            return False
    try:
        return not profile.is_exhausted("environment")
    except Exception:  # pragma: no cover - 防御式
        return True


def _environment_was_just_asked(profile: Any) -> bool:
    """上一轮刚问过环境、客户还没给出答案（答非所问 / 只报了别的需求）。

    客户口径（v2.4 起一直有效）：**答非所问 → 本轮不再重复问同一项，换下一个**；
    这一项留到"一轮走完后的硬性条件复问"再问（那时才用降门槛的问法）。
    实测 bug：客户回 "3*5"（没答室内外），系统紧接着又把"室内还是室外"问了一遍，
    还换成"That's okay, most installations are indoors…"，客户连看两次同一个问题。
    """
    if profile is None:
        return False
    try:
        if str(getattr(profile, "last_asked_slot", "") or "") != "environment":
            return False
        return int(profile.ask_count("environment") or 0) >= 1
    except Exception:  # pragma: no cover - 防御式
        return False


def environment_should_ask_now(profile: Any) -> bool:
    """环境这一轮该不该作为**第一问**出现。

    · 环境还没定（MISSING / UNKNOWN）      → 该问（硬性 Gate）
    · 但上一轮刚问过、客户还没答           → 这一轮先让位给别的问题
      （留到"一轮走完后的硬性条件复问"，那时才用降门槛的问法）
    """
    if not environment_needs_asking(profile):
        return False
    return not _environment_was_just_asked(profile)


def environment_gate_plan(
    profile: Any,
    *,
    language: str = "en",
    seed: int = 0,
) -> Optional["QuestionPlan"]:
    """环境未定时的那一句 Indoor / Outdoor 问题（否则 None）。"""
    if not environment_should_ask_now(profile):
        return None
    from src.rag.readiness import question_for

    easier = False
    try:
        easier = profile.ask_count("environment") >= 1
    except Exception:  # pragma: no cover - 防御式
        easier = False
    question = question_for("environment", language, seed, easier=easier)
    if not question:
        return None
    return QuestionPlan(
        slot="environment",
        question=question,
        action="ASK",
        easier=easier,
        reason="environment_hard_gate",
        why=why_for("environment"),
    )


def next_question_plan(
    profile: Any,
    *,
    session_id: str = "",
    language: str = "en",
    seed: int = 0,
    conversation: Any = None,
    customer_wants_recommendation: bool = False,
    exclude: Optional[set] = None,
) -> Optional[QuestionPlan]:
    """这一轮该问什么（返回 None 表示"该推荐了 / 交给 Gate"）。

    顺序：

        Phase 1（随机轮）：还没问过的槽位 → 按会话随机顺序问一个
        Phase 2（复问）：一轮走完后 → 只问缺的硬性条件（带"为什么"）
        客户明确要推荐 → 跳过 Phase 1，直接 Phase 2（齐了就返回 None = 推荐）
    """
    if profile is None:
        return None
    from src.rag.readiness import question_for

    skip = {str(item) for item in (exclude or set())}

    # 客户报了一个裸尺寸（"129,2cm"）但没说方向 → 交给 Gate 的专用确认问句，
    # 不参与随机轮（这是"回应客户刚说的话"，优先级最高）。
    if getattr(profile, "screen_size_hint_mm", None) and not getattr(
        profile, "has_target_size", False
    ):
        return None

    # 冲突 / 阻塞优先交给 Gate 的专用问句，不参与随机轮
    try:
        if (getattr(profile, "conflicts", None) or []) or str(
            profile.field_decision("environment")
        ) == "CONFLICT":
            return None
    except Exception:  # pragma: no cover - 防御式
        pass

    # ── v2.5+++（计划 §8）：Environment Hard Gate ─────────────────────────
    # 室内外还没定 → **第一项需求问题必须是它**（随机池只决定"其余问题"的顺序）。
    # 客户明确要推荐时也一样：先把最关键的硬条件问清楚，而不是随机挑一个。
    if "environment" not in skip:
        gate_plan = environment_gate_plan(profile, language=language, seed=seed)
        if gate_plan is not None:
            return gate_plan

    if not customer_wants_recommendation:
        # 会话状态里记着"本会话问过哪些"（即使档案被重建，也不会重复问同一项）
        asked_before = set(getattr(conversation, "asked_slots", []) or [])
        pending = [
            slot for slot in pass1_pending(profile, session_id)
            if slot not in skip and slot not in asked_before
        ]
        if pending:
            slot = pending[0]
            question = question_for(slot, language, seed, easier=False)
            if question:
                return QuestionPlan(
                    slot=slot, question=question, action="ASK", easier=False,
                    reason="random_pass",
                    why="",   # 第一轮只问问题本身（客户口径：复问硬性条件时才解释）
                )

    hard_pending = [
        slot for slot in hard_recap_pending(profile, session_id) if slot not in skip
    ]
    # 会话里问过、但档案里没记账的（档案被重建的情况）→ 也算"问过一次"，可进入复问
    asked_before = set(getattr(conversation, "asked_slots", []) or [])
    extra = [
        slot for slot in random_order(session_id, HARD_SLOTS)
        if slot in asked_before
        and slot not in hard_pending
        and slot not in skip
        and _askable(profile, slot, _actions(profile))
        and not _settled(profile, slot)
        and _asked(profile, slot) < _max_asks()
    ]
    hard_pending = hard_pending + extra
    # 避免"紧接着重复同一问"：上一轮刚问过的那一项排到最后（还有别的可问时）
    last_asked = str(getattr(profile, "last_asked_slot", "") or "")
    if len(hard_pending) > 1 and last_asked in hard_pending:
        hard_pending = [slot for slot in hard_pending if slot != last_asked] + [last_asked]
    if hard_pending:
        slot = hard_pending[0]
        easier = _asked(profile, slot) >= 1
        question = question_for(slot, language, seed + 3, easier=easier)
        if question:
            return QuestionPlan(
                slot=slot, question=question, action="ASK", easier=easier,
                reason="hard_condition_recap", why=why_for(slot),
            )
    return None


__all__ = [
    "ASK_POOL",
    "HARD_SLOTS",
    "HARD_WHY",
    "environment_gate_plan",
    "environment_needs_asking",
    "environment_should_ask_now",
    "hard_recap_pending",
    "next_question_plan",
    "pass1_complete",
    "pass1_pending",
    "random_order",
    "why_for",
]
