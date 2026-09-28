"""客户口径（2026-09-21 → 2026-09-22 收紧 → 2026-09-22 改判定依据）。

规则原文：

    · 不要让 AI 出现连续询问的情况；
    · 客户说了与需求无关的话时才"承接/闲聊"，然后最多三条就回到需求询问；
    · 硬性条件也可以跳转（每个问题没得到答案都可以跳），
      只有在最后要推荐产品时才必须满足硬性条件。

两次收紧后的**最终口径**：

    「承接三条改成承接一条，第二条必须回归有需求提问的那句话」
    「客户再说的是需求有关的话题，AI 就不要再寒暄承接了……如果客户聊得是
      无关的才承接闲聊，再客户说第二条闲聊消息的时候，承接客户的这句话，
      在同一条消息询问 AI 需求」（同一轮里的多条消息按**一次**算；
      **不要用关键词触发，让 AI 在语境里感知**）

所以判定依据**不是**"客户有没有回答上一问"，而是"客户这句话跟需求有没有关系"：

    ① 与需求有关（给了参数 / 答了上一问 / 主动聊需求 / 问业务问题）
        → 立刻恢复正常节奏：**接住 + 追问缺的那一项**（不做"只承接"）
        → 承接计数清零；缺的那一项由重复提问闸门保证不会重复问
    ② 与需求无关（闲聊 / 寒暄 / 题外话）
        → 这一轮只"承接"（正文里不许留问句）
        → 连续承接最多 1 条；**第 2 条闲聊必须"接住 + 提问"写在同一句话里**
    ③ 客户回到需求话题 → 计数清零，回到 ①

"与需求有没有关系"由销售节点在语境里判断（LLM 语义理解 + 规则解析，
见 ``src/agents/sales/nodes/requirement.py`` 的 ``offtopic_turn``），
这里只消费那个判定结果，**不设置任何关键词**。

配置（.env，可选）：

    LED_RAG_MAX_ACK_STREAK=1   # 连续承接上限（默认 1）
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

# 客户口径（2026-09-22）：每一次闲聊最长 1 条 —— 承接一条，第二条必须回归需求提问
MAX_ACK_STREAK = 1

# 计划 v2.9.4 / 客户口径（2026-09-23）：确认 / 纠正这两种 SpeechAct 都**不是闲聊** ——
# 客户回了 "ok"（确认我们的判断）之后就该继续推进，不能被"承接额度"压成一句寒暄
# （实测：客户确认 LED 后只收到 "Good to know."，一整个 turn 没有推进）。
REQUIREMENT_RELATED_SPEECH_ACTS = frozenset({"CONFIRMATION", "CORRECTION"})


def configured_ack_streak_limit() -> int:
    """连续承接上限（默认 1，可用 ``LED_RAG_MAX_ACK_STREAK`` 覆盖）。"""
    try:
        raw = os.environ.get("LED_RAG_MAX_ACK_STREAK")
        if raw is None or not str(raw).strip():
            return MAX_ACK_STREAK
        return max(0, int(str(raw).strip()))
    except Exception:  # pragma: no cover - 防御式
        return MAX_ACK_STREAK


@dataclass
class ContinuationDecision:
    """这一轮要不要把"需求问题"压下去（改成承接）。"""

    suppress_question: bool = False
    reason: str = ""
    next_streak: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suppress_question": self.suppress_question,
            "reason": self.reason,
            "next_streak": self.next_streak,
        }


def decide_continuation(
    *,
    has_question: bool,
    off_topic: bool,
    ack_streak: int = 0,
    max_ack_streak: Optional[int] = None,
    type_gate_question: bool = False,
) -> ContinuationDecision:
    """这一轮是"承接（不问）"还是"可以问需求"。纯函数，方便单测。

    Args:
        has_question: 本轮计划里是否有要问客户的需求问题
        off_topic: 客户这一句是否**与需求无关**（闲聊 / 题外话）
            —— 由销售节点在语境里判断（LLM + 规则），不是关键词匹配
        ack_streak: 已经连续承接了几条（不含本轮；按 turn 计，
            同一轮里客户连发多条消息只算一次）
        max_ack_streak: 承接上限（默认读 ``LED_RAG_MAX_ACK_STREAK``，缺省 1）
        type_gate_question: 本轮要问的是不是"LED 还是 LCD"（计划 §五 第一优先级）。
            类型没确认之前，这个问题**不是**"申请细节"，不受承接额度约束 ——
            线上实测（2026-09-23）：它被承接额度压掉后，客户只收到一段
            LED 需求话术，既没推进，也没人问他到底要哪种屏。
    """
    streak = max(0, int(ack_streak or 0))
    limit = max(
        0,
        int(
            max_ack_streak
            if max_ack_streak is not None
            else configured_ack_streak_limit()
        ),
    )

    if not off_topic:
        # ① 客户说的是需求相关的话（哪怕是答了别的一项 / 问业务问题）→
        #    不做"只承接"，直接接住 + 追问缺的那一项
        return ContinuationDecision(False, "requirement_related", 0)
    if type_gate_question:
        # ①.5 类型未确认 → "LED 还是 LCD"照问，且不消耗承接额度
        return ContinuationDecision(False, "type_gate_first", streak)
    if not has_question:
        # ② 闲聊，且本轮没有可问的需求问题 → 只承接（照样占额度）
        return ContinuationDecision(False, "no_question_this_turn", streak + 1)
    if streak < limit:
        # ② 第一条闲聊 → 只承接，不问问题
        return ContinuationDecision(True, "customer_off_topic_chat_first", streak + 1)
    # ② 第二条闲聊 → 必须"接住 + 提问"写在同一句话里（承接额度用完）
    return ContinuationDecision(False, "ack_budget_exhausted", 0)


def apply_continuation_budget(
    result: Dict[str, Any],
    *,
    conversation: Any = None,
    question_slot: str,
    questions: Any = None,
    profile_slots: Any = (),
    speech_act: Any = None,
) -> "tuple[str, list]":
    """把"承接额度"结论落进 result（从 Orchestrator 迁入，计划 §二/§五）。

    客户口径（2026-09-21 → 2026-09-22 最终版）：不要连续提问 ——

      · 与需求有关（给参数 / 答别的一项 / 问业务问题）→ 直接"接住 + 追问缺项"，
        绝不做"只承接"（实测 bug：客户答 "maybe 5m"，AI 只寒暄一句就停了）；
      · 与需求无关（闲聊）→ 只承接 1 条；第 2 条闲聊必须"接住 + 提问"写在同一条
        消息里（客户口径："第二条……在同一条消息询问需求"）。

    判定由销售节点在语境里给出（``result["offtopic_turn"]``，LLM + 规则，不看关键词）；
    同一轮里客户连发多条消息按**一次**算（一个 turn 只决策一次）。

    Returns:
        ``(question_slot, questions)``（被承接额度压掉时问句清空）
    """
    import logging

    logger = logging.getLogger(__name__)

    data = result if isinstance(result, dict) else {}
    ack_streak = int(getattr(conversation, "ack_streak", 0) or 0)
    ack_limit = configured_ack_streak_limit()
    off_topic = bool(data.get("offtopic_turn"))
    act = speech_act if speech_act is not None else (data.get("speech_act") or {})
    # 判定依据是销售节点给的 SpeechAct（语境判定），不是关键词。
    if isinstance(act, dict):
        act_name = str(act.get("speech_act") or act.get("act") or "").upper()
    else:
        act_name = str(getattr(act, "speech_act", "") or "").upper()
    if act_name in REQUIREMENT_RELATED_SPEECH_ACTS:
        off_topic = False
    # 档案里有值 = 客户确实给过这一项 → 问题登记簿同步成 ANSWERED，并把"档案里有值"
    # 也算成"客户答过"（实测：客户答"只在意质量"，档案已记 price_preference=quality，
    # 但登记簿还是 ASKED → 误判成没回答）。
    if conversation is not None:
        for slot in profile_slots or ():
            conversation.note_answered(str(slot))
    has_question = bool(question_slot and data.get("pending_question"))
    decision = decide_continuation(
        has_question=has_question,
        off_topic=off_topic,
        ack_streak=ack_streak,
        max_ack_streak=ack_limit,
        # 计划 v2.9.4 修复："LED 还是 LCD"不是需求细节问题（计划 §五 第一优先级），
        # 不受承接额度约束 —— 否则类型没定就被"先承接一句"压掉，客户看不到任何推进。
        type_gate_question=str(question_slot or "") == "display_type",
    )
    data["continuation"] = decision.to_dict()
    data["ack_streak"] = decision.next_streak
    if conversation is not None:
        conversation.ack_streak = decision.next_streak
    if decision.suppress_question and has_question:
        logger.info(
            "[Continuation] 客户说的是与需求无关的话 → 本轮先承接、不问问题"
            "（承接 %d/%d，下一轮接住 + 提问）",
            decision.next_streak, ack_limit,
        )
        data["suppressed_question"] = {
            "slot": question_slot,
            "question": str(data.get("pending_question") or ""),
            "ack_streak": decision.next_streak,
        }
        data["pending_question"] = ""
        data["pending_slot"] = ""
        return "", []
    return question_slot, list(questions or [])


__all__ = [
    "ContinuationDecision",
    "MAX_ACK_STREAK",
    "REQUIREMENT_RELATED_SPEECH_ACTS",
    "apply_continuation_budget",
    "configured_ack_streak_limit",
    "decide_continuation",
]
