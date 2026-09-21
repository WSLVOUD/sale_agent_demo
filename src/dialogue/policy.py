"""v2.5++++（消息聚合与自然对话链路优化 · 第四/五/六阶段）：Dialogue Policy。

计划 §7：把"缺什么就问什么"换成

    客户行为（SpeechAct） + 需求状态 + 业务优先级 → DialogueAction

优先级（§7.2）：

    P0  客户明确问题          → 先回答客户（这一轮可以不追问）
    P1  客户纠正 / 冲突        → 先澄清
    P2  客户刚给的需求要确认    → 接住 + 更新状态
    P3  Gate 必须条件          → 问
    P4  业务上最有价值的下一问   → 问
    P5  普通补充信息           → 只回应

§9：不是每一轮回答客户后都必须继续问问题 —— `answer_only` 是合法的动作。
§8：下一问用"候选 → 过滤 → 按业务价值排序 → 同分才随机 → 只选一个"。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .speech_act import (
    CUSTOMER_QUESTION_ACTS,
    DELIVERY_QUESTION,
    MULTI_INTENT,
    PRICE_QUESTION,
    SpeechActResult,
    detect_speech_act,
)

# ── 动作（计划 §9.1）──────────────────────────────────────────────────────
ANSWER_ONLY = "answer_only"
ANSWER_THEN_ASK = "answer_then_ask"
ASK_ONLY = "ask_only"
CLARIFY_ONLY = "clarify_only"
RECOMMEND_ONLY = "recommend_only"
ACK_ONLY = "acknowledge_only"

# ── 优先级（计划 §7.2）────────────────────────────────────────────────────
P0_CUSTOMER_QUESTION = 0
P1_CORRECTION_CONFLICT = 1
P2_NEW_REQUIREMENT = 2
P3_GATE_REQUIRED = 3
P4_BEST_NEXT_QUESTION = 4
P5_EXTRA_INFO = 5

# 只回答、这一轮**不**追加需求问题的问句类型（计划 §9 / §16.4 / §16.5）：
# 客户问价格、问交期时，"回答"本身就是这一轮的正事，硬塞需求问题就是问卷腔。
ANSWER_ONLY_KINDS = frozenset({PRICE_QUESTION, DELIVERY_QUESTION})

PRIORITY_LABELS = {
    P0_CUSTOMER_QUESTION: "customer_question",
    P1_CORRECTION_CONFLICT: "correction_or_conflict",
    P2_NEW_REQUIREMENT: "requirement_needs_ack",
    P3_GATE_REQUIRED: "gate_required",
    P4_BEST_NEXT_QUESTION: "best_next_question",
    P5_EXTRA_INFO: "extra_info",
}


@dataclass
class DialogueAction:
    """这一轮"做什么"（不决定怎么说）。"""

    action: str = ASK_ONLY
    target_slot: str = ""
    priority: int = P5_EXTRA_INFO
    reason: str = ""
    question: str = ""
    speech_act: str = ""
    candidates: List[Tuple[str, float]] = field(default_factory=list)
    answer_kinds: List[str] = field(default_factory=list)

    @property
    def asks_question(self) -> bool:
        return self.action in (ASK_ONLY, ANSWER_THEN_ASK, CLARIFY_ONLY)

    @property
    def answers_customer(self) -> bool:
        return self.action in (ANSWER_ONLY, ANSWER_THEN_ASK)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "target_slot": self.target_slot,
            "priority": self.priority,
            "priority_label": PRIORITY_LABELS.get(self.priority, ""),
            "reason": self.reason,
            "question": self.question,
            "speech_act": self.speech_act,
            "answer_kinds": list(self.answer_kinds),
            "candidates": [list(item) for item in self.candidates],
        }


# ── §8：候选问题 → 过滤 → 排序 ────────────────────────────────────────────
def question_candidates(
    profile: Any,
    *,
    session_id: str = "",
) -> List[Tuple[str, float]]:
    """当前还值得问的候选槽位（按业务价值从高到低）。

    价值 = 硬性条件 / 工程必要 > 推荐优化 > 销售偏好；同价值的顺序由会话随机种子决定。
    """
    if profile is None:
        return []
    from .question_flow import ASK_POOL, environment_should_ask_now
    from .question_order import shuffled_slots
    from src.rag.field_policy import field_action

    order = list(shuffled_slots(ASK_POOL, seed=session_id or ""))
    scores: List[Tuple[str, float]] = []
    for slot in order:
        try:
            action = field_action(profile, slot)
        except Exception:  # pragma: no cover - 防御式
            continue
        if action not in ("ask", "ask_easier", "ask_later"):
            continue
        if slot == "environment":
            # 环境：没定才问（而且"刚问过没答"要让位）
            if not environment_should_ask_now(profile):
                continue
            value = 10.0
        else:
            value = _business_value(slot, action)
        scores.append((slot, value))
    # 价值降序；同价值保持（会话随机后的）顺序 → 只在同价值之间才有随机性
    return sorted(scores, key=lambda item: -item[1])


def _business_value(slot: str, action: str) -> float:
    """§8：按业务价值给分（越大越先问）。"""
    from .action import question_priority

    priority = question_priority(slot)
    base = {0: 6.0, 1: 5.0, 2: 4.0, 3: 3.0}.get(priority, 3.0)
    if action == "ask_easier":
        base -= 0.5      # 客户答过"不知道"的，降门槛问法排后一点
    if action == "ask_later":
        base -= 1.5      # 客户还没答过的延后项最后再问
    return base


def select_next_question(
    profile: Any,
    *,
    session_id: str = "",
) -> Optional[Tuple[str, str]]:
    """选**一个**下一问题（返回 ``(slot, question)``；没有则 None）。"""
    candidates = question_candidates(profile, session_id=session_id)
    if not candidates:
        return None
    from src.rag.readiness import question_for

    slot = candidates[0][0]
    easier = False
    try:
        easier = profile.ask_count(slot) >= 1
    except Exception:  # pragma: no cover - 防御式
        easier = False
    question = question_for(slot, "en", 0, easier=easier) or ""
    if not question:
        return None
    return slot, question


def rank_slots_by_value(
    slots: Sequence[str],
    profile: Any,
    *,
    session_id: str = "",
) -> List[str]:
    """§8：把候选槽位按**业务价值**排序（同价值保持传入的随机顺序）。

    `question_flow.pass1_pending()` 先按会话随机取顺序，再调用这里做价值排序 ——
    于是"随机"只在价值相同的槽位之间生效，不会再随机到不重要的字段。
    """
    if not slots:
        return []
    from src.rag.field_policy import field_action

    def value(slot: str) -> float:
        if slot == "environment":
            return 10.0 if _environment_open(profile) else 0.0
        try:
            action = field_action(profile, slot)
        except Exception:  # pragma: no cover - 防御式
            action = "ask"
        return _business_value(slot, action)

    return sorted(list(slots), key=lambda slot: -value(slot))


def _environment_open(profile: Any) -> bool:
    try:
        from .question_flow import environment_should_ask_now

        return environment_should_ask_now(profile)
    except Exception:  # pragma: no cover - 防御式
        return False


# ── §7：决定这一轮做什么 ──────────────────────────────────────────────────
def decide_action(
    speech_act: SpeechActResult,
    *,
    profile: Any = None,
    ready_to_recommend: bool = False,
    conflicts: Optional[Sequence[str]] = None,
    recommend_requested: bool = False,
    session_id: str = "",
    last_asked_slot: str = "",
) -> DialogueAction:
    """客户行为 + 需求状态 + 业务优先级 → 这一轮的动作。"""
    conflicts = list(conflicts or [])

    # P1：冲突 / 纠正 → 先澄清（客户刚纠正的事实必须优先级高）
    if conflicts or speech_act.speech_act == "CORRECTION":
        return DialogueAction(
            action=CLARIFY_ONLY,
            priority=P1_CORRECTION_CONFLICT,
            reason="correction_or_conflict",
            speech_act=speech_act.speech_act,
            answer_kinds=list(conflicts),
        )

    # P0：客户主动提问 → **先回答客户**（§9：不强制追加需求问题）
    if speech_act.is_customer_question:
        kind = speech_act.question_kind()
        if kind in ANSWER_ONLY_KINDS or ready_to_recommend:
            # 价格 / 交期：答完就结束这一轮（下一轮客户继续说需求再继续采集）
            return DialogueAction(
                action=ANSWER_ONLY,
                priority=P0_CUSTOMER_QUESTION,
                reason="customer_question_answer_only",
                speech_act=speech_act.speech_act,
                answer_kinds=[kind] if kind else [],
            )
        action, reason = ANSWER_ONLY, "customer_question_answer_only"
        question, target_slot = "", ""
        picked = (
            select_next_question(profile, session_id=session_id)
            if profile is not None else None
        )
        if picked:
            # 产品/规格类问题：只有"回答它确实缺某个硬性条件"时才追问（有理由的追问）；
            # 其它客户提问（公司信息 / 泛问）沿用既有口径：答完接着收集需求。
            if _needed_to_answer(kind, picked[0]) or kind not in ("PRODUCT_QUESTION",):
                action, target_slot, question = ANSWER_THEN_ASK, picked[0], picked[1]
                reason = "answer_then_ask_required_field"
        return DialogueAction(
            action=action,
            target_slot=target_slot,
            priority=P0_CUSTOMER_QUESTION,
            reason=reason,
            question=question,
            speech_act=speech_act.speech_act,
            answer_kinds=[kind] if kind else [],
        )

    # P1：客户明确要推荐 / Gate 已就绪 → 推荐
    if recommend_requested or ready_to_recommend:
        return DialogueAction(
            action=RECOMMEND_ONLY,
            priority=P3_GATE_REQUIRED,
            reason="recommend_requested" if recommend_requested else "gate_ready",
            speech_act=speech_act.speech_act,
        )

    # P2/P3/P4：客户答了需求 / 主动补充 → 接住，然后按价值问下一个
    if speech_act.speech_act in ("ANSWER_REQUIREMENT", "NEW_REQUIREMENT", MULTI_INTENT):
        picked = select_next_question(profile, session_id=session_id)
        if picked:
            slot, question = picked
            return DialogueAction(
                action=ASK_ONLY,
                target_slot=slot,
                priority=P3_GATE_REQUIRED if _is_hard(slot) else P4_BEST_NEXT_QUESTION,
                reason="requirement_answered_ask_next",
                question=question,
                speech_act=speech_act.speech_act,
            )
        return DialogueAction(
            action=ACK_ONLY,
            priority=P2_NEW_REQUIREMENT,
            reason="requirement_recorded_nothing_to_ask",
            speech_act=speech_act.speech_act,
        )

    # P5：其它（闲聊 / 确认）→ 只回应
    return DialogueAction(
        action=ACK_ONLY,
        priority=P5_EXTRA_INFO,
        reason="casual_or_confirmation",
        speech_act=speech_act.speech_act,
    )


def _needed_to_answer(kind: str, slot: str) -> bool:
    """客户问的那类问题，是否**必须**先知道这个槽位才能答准。"""
    if kind == "PRODUCT_QUESTION":
        # 问"能不能做到/多大分辨率"时，尺寸/点间距/环境是回答的前提
        return slot in ("size", "pixel_pitch", "environment")
    return False


def _is_hard(slot: str) -> bool:
    try:
        from src.rag.field_policy import policy_for

        return bool(policy_for(slot).hard_condition)
    except Exception:  # pragma: no cover - 防御式
        return False


def should_append_requirement_question(message: str, profile: Any = None) -> bool:
    """这一轮回答客户之后，**该不该**再追一个需求问题。

    客户口径（§9 / §16.4 / §16.5）：
      · 客户问价格 / 交期 / 公司信息 → 先答完，**不硬塞**需求问题；
      · 客户问产品规格、而回答它确实缺硬性条件 → 可以有理由地追一个。
    """
    try:
        speech_act = detect_speech_act(
            message,
            profile=profile,
            last_asked_slot=str(getattr(profile, "last_asked_slot", "") or ""),
        )
        kind = speech_act.question_kind()
        # 价格 / 交期：这一轮只回答（§16.4 / §16.5）
        if kind in ANSWER_ONLY_KINDS:
            return False
        # 产品/规格类问题：只有"回答它确实缺某个硬性条件"才有理由追问
        if kind == "PRODUCT_QUESTION" and profile is not None:
            picked = select_next_question(profile)
            if picked and _needed_to_answer(kind, picked[0]):
                return True
            return False
        # 其它（公司信息 / 泛问 / 异议 / 不是提问）→ 维持既有口径：答完继续采集
        return True
    except Exception:  # pragma: no cover - 防御式
        return True


def decide_speech_policy(
    message: str,
    *,
    profile: Any = None,
    ready_to_recommend: bool = False,
    conflicts: Optional[Sequence[str]] = None,
    recommend_requested: bool = False,
    session_id: str = "",
) -> Dict[str, Any]:
    """一站式：识别 SpeechAct + 决定 DialogueAction（返回两个 dict，便于落日志/落 state）。"""
    speech = detect_speech_act(
        message,
        profile=profile,
        last_asked_slot=str(getattr(profile, "last_asked_slot", "") or ""),
    )
    action = decide_action(
        speech,
        profile=profile,
        ready_to_recommend=ready_to_recommend,
        conflicts=conflicts,
        recommend_requested=recommend_requested,
        session_id=session_id,
    )
    return {"speech_act": speech.to_dict(), "dialogue_action": action.to_dict()}


__all__ = [
    "ACK_ONLY",
    "ANSWER_ONLY",
    "ANSWER_THEN_ASK",
    "ASK_ONLY",
    "CLARIFY_ONLY",
    "DialogueAction",
    "P0_CUSTOMER_QUESTION",
    "P1_CORRECTION_CONFLICT",
    "P2_NEW_REQUIREMENT",
    "P3_GATE_REQUIRED",
    "P4_BEST_NEXT_QUESTION",
    "P5_EXTRA_INFO",
    "PRIORITY_LABELS",
    "RECOMMEND_ONLY",
    "decide_action",
    "decide_speech_policy",
    "question_candidates",
    "rank_slots_by_value",
    "select_next_question",
    "should_append_requirement_question",
]
