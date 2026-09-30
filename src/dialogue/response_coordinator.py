"""v2.3 §13：把 Orchestrator 里的"回复组装"抽出来。

Orchestrator 只负责流程编排；这里负责"这一轮最终对客户说什么"：

    业务回复（Sales / Solution 已经生成好的那句话）
        + 售后口径回答（客户问到才加）
        + 图片识别结果核对（带图的那一轮）

抽出来之后，Orchestrator 不再直接碰这些业务细节，也方便单测。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TurnPlan:
    """这一轮"想说什么 + 问哪一项"的最终结论（从 Orchestrator 迁入）。"""

    text: str = ""
    questions: list = field(default_factory=list)
    action: str = ""
    question_slot: str = ""
    conversation_before: dict = field(default_factory=dict)
    extras: list = field(default_factory=list)
    density: str = ""


class ResponseCoordinator:
    """回复的最后两道加工（顺序固定：售后口径 → 图片核对）。"""

    def __init__(self, store: Any = None, profile_lookup: Any = None):
        self.store = store
        self.profile_lookup = profile_lookup
        self.last_guard: Any = None

    # ── 售后口径 ────────────────────────────────────────────────────────
    def attach_service_faq(
        self, response: str, message: str, *, already_answered: bool = False
    ) -> str:
        """客户问到售后口径时**必须回答**（不主动提，尤其质保）。

        ``already_answered=True``：销售那一轮已经按标准口径答过了（由 LLM 自己组织
        成一段话）→ 这里**不再前置**标准口径，只做"矛盾清洗"，避免同一段里说两遍、
        甚至出现"我们不做现场安装 / 我们提供现场安装"这种自相矛盾。
        """
        try:
            from src.rag.reply_composer import reply_language
            from src.rag.service_faq import (
                detect_service_faq,
                sanitize_service_reply,
                service_faq_reply,
            )

            answer = service_faq_reply(message, language=reply_language(message))
            # 与标准口径矛盾的句子（实测："Yes, installation is included."）先清掉
            response = sanitize_service_reply(response, detect_service_faq(message))
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("[ServiceFAQ] failed: %s", exc)
            return response
        if already_answered:
            return response
        if not answer or answer in str(response or ""):
            return response
        return f"{answer} {response}".strip() if response else answer

    # ── 图片核对 ────────────────────────────────────────────────────────
    def attach_vision_confirmation(self, response: str, session_id: str, message: str) -> str:
        """带图的那一轮：先把"图片里看到什么"跟客户核一遍，再继续问需求。"""
        try:
            profile = self._profile(session_id)
            if profile is None or not getattr(profile, "vision_confirmation_pending", None):
                return response
            # LCD / IFP（《LCD_IFP 整改计划》§二十二）：核对要结合上下文，
            # 客户已经说过的部分不再重复问（"I've matched the image with what you described…"）。
            if str(getattr(profile, "display_type", "") or "").upper() in ("LCD", "IFP"):
                from .image_confirmation import prompt_from_profile
                from src.rag.reply_composer import reply_language

                lcd_sentence = prompt_from_profile(
                    profile, language=reply_language(message)
                )
                if lcd_sentence:
                    text = str(response or "")
                    if lcd_sentence in text:
                        return response
                    return f"{lcd_sentence} {text}".strip() if text else lcd_sentence
            from src.rag.reply_composer import (
                reply_language,
                vision_confirmation_items,
                vision_confirmation_sentence,
            )

            language = reply_language(message)
            items = [item for item in vision_confirmation_items(profile, language) if item]
            text = str(response or "")
            if items and all(item in text for item in items):
                return response
            sentence = vision_confirmation_sentence(profile, language, 0)
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("[Vision] confirmation sentence failed: %s", exc)
            return response
        if not sentence or sentence in str(response or ""):
            return response
        return f"{sentence} {response}".strip() if response else sentence

    def finalize(
        self,
        response: str,
        *,
        session_id: str = "",
        message: str = "",
        questions: Any = None,
        language: str = "en",
        service_faq_answered: str = "",
        require_question: bool = False,
    ) -> str:
        """固定顺序：售后口径 → 图片核对 → **FinalResponseGuard 收口**。

        v2.5+++（计划 §3）：Guard 是客户可见内容的**唯一最后一道**，
        保证一轮最多一个问题（多余的进下一轮）并把内部术语清掉。
        """
        text = self.attach_service_faq(
            str(response or ""), message, already_answered=bool(service_faq_answered)
        )
        text = self.attach_vision_confirmation(text, session_id, message)
        guarded = self._guard().finalize(
            text, questions=questions, language=language
        )
        self.last_guard = guarded
        text = guarded.text
        # 客户口径（2026-09-28 实测）：**必问的那一问不能丢**。
        # 实测：LCD 入口那一轮 LLM 只写了"Let's go with LCD display…"，问题被吞掉，
        # 客户等不到下一问（Sales / DialoguePolicy 都已经定好"这一轮要问什么"）。
        # Guard 只负责"最多一个问题"，不补问题 —— 这里补上（仅当整段没有任何问句）。
        # 注意：只有"这一轮本来就是问问题"（ask_only / answer_then_ask）才补，
        # 推荐轮（RECOMMEND）里残留的 pending_question 不许被塞进交付话术。
        if require_question:
            text = self._ensure_required_question(text, questions)
        return text

    @staticmethod
    def _ensure_required_question(text: str, questions: Any) -> str:
        """这一轮有"必问项"但正文里一个问句都没有 → 把那一问补回去。"""
        required = ""
        for item in questions or []:
            candidate = item.get("text") if isinstance(item, dict) else str(item or "")
            candidate = str(candidate or "").strip()
            if candidate:
                required = candidate
                break
        if not required:
            return text
        if "?" in str(text) or "？" in str(text):
            return text
        body = str(text or "").strip()
        if not body:
            # 正文为空说明 Guard 有意清空了它（下游还有 result["response"] 兜底），
            # 这时不要凭空造一句问句出来。
            return text
        return f"{body} {required}".strip()

    # ── 最终收口（FinalResponseGuard）────────────────────────────────────
    def prepare_final(
        self,
        result: "Dict[str, Any]",
        *,
        session_id: str,
        message: str,
        turn_context: Optional[Dict[str, Any]] = None,
    ) -> TurnPlan:
        """v2.7 Phase 7~14 的"决策段"（计划《二次瘦身计划》§二/§五）。

        顺序与迁移前完全一致：

            覆盖率 → 重复提问闸门 → Policy 对齐 → 产品类型闸门
                → Momentum → 唯一 Action → 承接上下文 → 承接额度

        全部落进 ``result`` 供审计留痕，返回本轮的 ``TurnPlan`` 供收口使用。
        """
        from .answer_coverage import compute_answer_coverage
        from .continuation_budget import apply_continuation_budget
        from .conversation_state import (
            conversation_snapshot,
            get_conversation_state,
            last_answer_match_object,
        )
        from .duplicate_firewall import FIREWALL, apply_question_firewall
        from .final_response import question_candidates_from_result
        from .momentum import compute_momentum
        from .natural_continuation import build_natural_continuation
        from .policy import ranked_question_slots
        from .product_type_router import align_question_with_type_gate
        from .profile_slots import profile_slot_map
        from .response_density import MINIMAL
        from .action_bridge import dialogue_action_label
        from .turn_action import (
            align_question_with_policy,
            decide_turn_action,
            next_candidate_slot,
            select_turn_action,
        )

        turn_context = dict(turn_context or result.get("turn_context") or {})
        questions = question_candidates_from_result(result)
        # §28：审计要记"做决定之前"的对话状态，所以先拍一张快照
        conversation_before = conversation_snapshot(session_id)
        # 计划 §4.4/§4.5：候选 Action → 唯一 Action（其余记进 discarded_actions）
        selected_action, discarded_actions = select_turn_action(result)
        action = (
            selected_action.action
            if selected_action is not None
            else dialogue_action_label(result)
        )
        result["discarded_actions"] = [item.to_dict() for item in discarded_actions]
        if selected_action is not None:
            result["selected_action"] = selected_action.to_dict()
            result["action_candidates"] = [
                item.to_dict() for item in ([selected_action] + list(discarded_actions))
            ]
        question_slot = str(result.get("pending_slot") or "")
        extras = [str(item) for item in (result.get("extra_messages") or []) if item]
        result.setdefault("customer_input", str(message or ""))

        # ══ v2.7 Phase 7~11：覆盖率 → 闸门 → 唯一 Action → 承接上下文 ══════
        previous_question = turn_context.get("previous_question") or {}
        previous_slot = str(previous_question.get("question_slot") or "")
        newly_filled = [str(item) for item in (result.get("newly_filled_slots") or [])]
        coverage = compute_answer_coverage(
            asked_slot=previous_slot,
            match=last_answer_match_object(session_id),
            newly_filled_slots=newly_filled,
        )
        result["answer_coverage"] = coverage.to_dict()

        # ── LCD / IFP 与 LED **在这条链路上彻底隔离**（计划 §二十四/§二十九）──
        # 实测（2026-09-28）：LCD 决策层已经算出 ask_purpose，但收口层的 LED
        # DialoguePolicy 把问题改回 environment / size(width x height) / installation，
        # 客户看到的还是 LED 问句（"Policy=ASK(environment) 与销售层准备的问句(purpose)
        # 不一致 → 以 Policy 为准"）。所以 LCD 会话**不跑** LED 的重复提问闸门与
        # Policy 对齐：下一问只认 LCD 决策层（lcd_action）。
        lcd_turn_active = bool(result.get("lcd_action")) or str(
            result.get("product_domain") or ""
        ).upper() in ("LCD", "IFP")
        if lcd_turn_active:
            duplicate_check = "lcd_chain"
            result["duplicate_check"] = duplicate_check
            questions = question_candidates_from_result(result)
            logger.info(
                "[LCD] 跳过 LED 的重复提问闸门 / Policy 对齐 —— 下一问由 LCD 决策层决定"
                "（slot=%s）",
                question_slot or "-",
            )
        else:
            question_slot, duplicate_check = apply_question_firewall(
                result,
                session_id=session_id,
                coverage=coverage,
                previous_question=previous_question,
                newly_filled=newly_filled,
                ranked_slots=ranked_question_slots(
                    self._profile(session_id), session_id=session_id
                ),
                firewall=FIREWALL,
                next_candidate=next_candidate_slot,
            )
            if question_slot:
                result["pending_slot"] = question_slot
            result["duplicate_check"] = duplicate_check
            questions = question_candidates_from_result(result)

            # Phase 1（计划 §16.2）+ 计划 v2.9.3 §五/§六：问哪一项由对话层收口
            question_slot, questions = align_question_with_policy(
                result,
                selected_action=selected_action,
                question_slot=question_slot,
                duplicate_check=duplicate_check,
            )
        question_slot, questions = align_question_with_type_gate(
            result,
            question_slot=question_slot,
            sales_asked_type=bool(result.get("product_type_gate")),
        )

        conversation = get_conversation_state(session_id)
        momentum = compute_momentum(
            newly_filled_slots=newly_filled,
            answered_slots=coverage.answered_slots,
            last_question_slot=str(getattr(conversation, "last_question_slot", "") or ""),
        )
        result["momentum"] = momentum.to_dict()
        speech_act = result.get("speech_act") or {}
        customer_question = bool(
            speech_act.get("customer_questions") or speech_act.get("is_customer_question")
        )
        question_kind = str(speech_act.get("question_kind") or "")
        turn_action = decide_turn_action(
            customer_question=customer_question,
            question_kind=question_kind,
            ready_to_recommend=bool((result.get("recommendation_gate") or {}).get("ready")),
            recommend_requested=bool(result.get("recommend_requested")),
            conflicts=(result.get("requirements") or {}).get("conflicts"),
            newly_filled_slots=newly_filled,
            missing_slots=result.get("missing_slots") or [],
            # LCD 会话的候选问题来自 LCD 决策层（不许把 LED 槽位塞进偏好记录）
            question_candidates=(
                list((result.get("lcd_action") or {}).get("missing_fields") or [])
                if lcd_turn_active
                else ranked_question_slots(self._profile(session_id), session_id=session_id)
            ),
            momentum_slot=momentum.slot,
            previous_question_slot=previous_slot,
            blocked_slot=str(previous_question.get("question_slot") or "")
            if duplicate_check.startswith("duplicate_question")
            else "",
            hard_gate_slot="environment"
            if result.get("pending_slot") == "environment"
            else "",
        )
        result["turn_action"] = turn_action.to_dict()
        # §20/§21：Question Planner/Flow 产出候选，Dialogue Policy 决定"这一轮做不做、
        # 做的优先级"；**具体问哪一项仍以已落地的候选为准** —— 实测教训：事后改槽位
        # 会和已经组好的正文脱节（日志里 last_question 与 last_response 不一致）。
        result["policy_preferred_slot"] = str(getattr(turn_action, "target_slot", "") or "")
        # §14：Conversation State 的"当前话题"（momentum）只用于对话规划
        if conversation is not None:
            conversation.current_topic = momentum.slot
        continuation = build_natural_continuation(
            customer_message=message,
            speech_act=speech_act,
            newly_filled_slots=newly_filled,
            momentum=momentum,
            next_required_slot=question_slot,
            customer_question=customer_question,
            question_kind=question_kind,
            conflicts=(result.get("requirements") or {}).get("conflicts"),
            has_recommendation=bool(result.get("products")),
            is_first_contact=bool(result.get("first_contact_messages")),
        )
        result["response_density"] = continuation.density
        result["natural_continuation"] = continuation.to_dict()

        # 客户口径（2026-09-21 → 2026-09-22 最终版）：不要连续提问 ——
        # 判定与额度都在对话层（continuation_budget），这里只落地结论。
        question_slot, questions = apply_continuation_budget(
            result,
            conversation=conversation,
            question_slot=question_slot,
            questions=questions,
            profile_slots=profile_slot_map(self._profile(session_id)),
            # LCD 决策层算出的下一问不受 LED 时代的"承接额度"约束
            protect_question=lcd_turn_active,
        )
        if (
            continuation.density == MINIMAL
            and str(result.get("response_mode") or "") != "FALLBACK"
        ):
            result["response_mode"] = "MINIMAL"

        return TurnPlan(
            text=str(result.get("response") or ""),
            questions=list(questions or []),
            action=action,
            question_slot=question_slot,
            conversation_before=conversation_before,
            extras=extras,
            density=continuation.density,
        )

    def postprocess_final(self, text: str, *, result: Any = None, density: str = "") -> str:
        """Guard 之后的正文加工（从 Orchestrator 迁入，计划《二次瘦身计划》§五）。

        顺序固定（与迁移前一致）：

          ① 承接轮：正文里不许再留下问句（只接住客户的话）；
          ② 去掉机械确认开头（"Got it / Thanks / Based on that…"）——
             只管模板拼出来的句子；LLM 自己写的开场不再被回头清洗；
          ③ 型号闸门：没在交付推荐，就不许出现具体型号
             （"需求没收齐就先报型号"是客户实测抱怨过的）。
        """
        from .natural_continuation import continuation_only_text
        from .natural_response import strip_mechanical_phrases
        from .response_density import DETAILED

        data = result if isinstance(result, dict) else {}
        out = str(text or "")
        if data.get("suppressed_question"):
            out = continuation_only_text(
                text=out,
                acknowledgement=str(data.get("acknowledgement") or ""),
                customer_input=str(data.get("customer_input") or ""),
            )
        out, removed_mechanical = strip_mechanical_phrases(
            out,
            allow=(
                str(density or "") == DETAILED
                or str(data.get("response_source") or "") == "llm"
                or bool(data.get("suppressed_question"))
            ),
        )
        if removed_mechanical:
            data["mechanical_phrases_removed"] = removed_mechanical

        try:
            delivering = bool(data.get("products")) or bool(
                (data.get("recommendation_gate") or {}).get("ready")
            )
            if out and not delivering:
                from src.rag.model_guard import strip_model_mentions

                cleaned, removed_models = strip_model_mentions(
                    out, allow=data.get("allowed_models") or ()
                )
                if removed_models:
                    logger.info(
                        "[ModelGate] 需求未就绪 → 清掉提前报出的型号：%s", removed_models
                    )
                    data["model_mentions_stripped"] = removed_models
                    out = cleaned or out
            elif out:
                # 客户口径（2026-09-28）：交付推荐时不要"怎么选"的收尾评论
                # （"Both options … so the choice comes down to …"）—— 话术少点，
                # 也不要复述客户已经说过的需求；带排布数字的句子一律保留。
                from src.utils.text import (
                    strip_fact_free_commentary,
                    strip_requirement_echo_prefix,
                )

                out = strip_requirement_echo_prefix(strip_fact_free_commentary(out))
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("[ModelGate] 型号闸门失败：%s", exc)
        return out

    def guard_extras(
        self,
        response: str,
        extras: Any,
        *,
        questions: Any = None,
    ):
        """附加气泡（extra_messages）也过一遍 Guard：一轮最多一个问题。"""
        return self._guard().guard_extras(response, list(extras or []), questions=questions)

    def _guard(self):
        guard = getattr(self, "_guard_instance", None)
        if guard is None:
            from .final_guard import FinalResponseGuard

            guard = FinalResponseGuard()
            self._guard_instance = guard
        return guard

    # ── 图片核对句（v2.3.1 从 Orchestrator 迁入）────────────────────────────
    def vision_confirmation_sentence(self, session_id: str, message: str) -> str:
        """图片识别结果的"跟客户核对"那一句（没有待确认字段时返回空串）。"""
        if not session_id:
            return ""
        try:
            profile = self._profile(session_id)
            if profile is None or not getattr(profile, "vision_confirmation_pending", None):
                return ""
            if str(getattr(profile, "display_type", "") or "").upper() in ("LCD", "IFP"):
                from .image_confirmation import prompt_from_profile
                from src.rag.reply_composer import reply_language

                lcd_sentence = prompt_from_profile(
                    profile, language=reply_language(message)
                )
                if lcd_sentence:
                    return lcd_sentence
            from src.rag.reply_composer import reply_language, vision_confirmation_sentence

            return vision_confirmation_sentence(profile, reply_language(message), 0)
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("[Vision] confirmation sentence failed: %s", exc)
            return ""

    # ── 答复 + 继续追问（v2.3.1 从 Orchestrator 迁入）───────────────────────
    def compose_with_requirement_question(
        self,
        *,
        answer: str,
        sales_result: Any,
        message: str,
        seed: int = 0,
        session_id: str = "",
    ) -> str:
        """把"答复客户"与"继续追问需求"合成一句自然的销售回复。

        需求还没问清时（Ready Gate 未放行），客户的问题照样要答 —— 但答完必须
        把还缺的那个关键问题接上，否则销售就变成"只会问问题"或"答完就断线"。
        """
        sales_result = sales_result or {}
        question = str(sales_result.get("pending_question") or "")
        if not question:
            return answer
        # v2.5++++（计划 §9 / §16.4 / §16.5）：Dialogue Policy 说"这一轮只回答"时，
        # 不追加需求问题 —— 客户问价格 / 交期，答完就是答完，别硬塞问卷题。
        try:
            from .policy import should_append_requirement_question

            profile = self._profile(session_id)
            if not should_append_requirement_question(message, profile):
                logger.info(
                    "[DialoguePolicy] answer_only：本轮回答客户后不再追加需求问题"
                )
                return answer
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("[DialoguePolicy] 判定失败，按原口径追加：%s", exc)
        try:
            from src.rag.reply_composer import compose_requirement_reply, reply_language

            return compose_requirement_reply(
                answer=answer,
                question=question,
                slot=str(sales_result.get("pending_slot") or ""),
                message=message,
                language=reply_language(message),
                seed=seed,
                requirement=sales_result.get("requirements") or {},
                include_ack=not sales_result.get("requirements_reset", False),
                llm_ack=str(sales_result.get("acknowledgement") or ""),
                # 带图的那一轮：图片识别结果要跟客户核对（而且不能一边核对一边又问同一项）
                vision_confirmation=self.vision_confirmation_sentence(session_id, message),
            )
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("Compose requirement reply failed: %s", exc)
            return answer or question

    # ── 内部 ────────────────────────────────────────────────────────────
    def _profile(self, session_id: str) -> Optional[Any]:
        if not session_id:
            return None
        if callable(self.profile_lookup):
            try:
                return self.profile_lookup(session_id)
            except Exception as exc:  # pragma: no cover - 防御式
                logger.warning("[ResponseCoordinator] profile_lookup failed: %s", exc)
                return None
        try:
            if self.store is not None and hasattr(self.store, "get_requirement_profile"):
                return self.store.get_requirement_profile(session_id)
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("[ResponseCoordinator] profile lookup failed: %s", exc)
        return None


__all__ = ["ResponseCoordinator"]
