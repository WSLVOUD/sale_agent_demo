"""Solution 需求节点 —— 只读 RequirementProfile，不再自己理解需求。

计划 update_v2.9.x 第三/四/五/七/九阶段（Solution 里的重复需求理解全部退出）：

    · ``understand_node`` 只做 Adapter：读 RequirementProfile → 用确定性 Gate
      给出 ``info_sufficient`` / ``missing_info`` → 返回；没有档案就标记
      REQUIREMENT_NOT_READY，**不再**自己造需求；
    · 已删除：``infer_display_type``（IFP 否则 LED 的默认判断）、
      ``_build_requirement_prompt`` / ``_parse_requirement_response``
      以及 Solution 内部那条 LLM 需求提取链；
    · 工程参数推断统一由 ``rag.parameter_inference`` 负责（本文件只调用它）；
    · 产品类型（LED/LCD/IFP）只由 ProductTypeRouter 决定。
"""
import logging
import re
from typing import Any, Dict, List, Optional

from ..state import SolutionState

logger = logging.getLogger(__name__)


def _extract_keywords(text: str) -> List[str]:
    """Extract search keywords from text."""
    keywords = []
    keyword_patterns = [
        r"演唱会|音乐会|演出",
        r"体育|足球|篮球|田径",
        r"广告|户外广告|路牌",
        r"会议|会议室|培训",
        r"教室|教育|课堂",
        r"租赁|快装|拆卸",
        r"室内|户外|室外",
    ]
    for pattern in keyword_patterns:
        if re.search(pattern, text):
            keywords.append(re.search(pattern, text).group())
    return keywords


def understand_node(state: SolutionState) -> SolutionState:
    """纯 Adapter（计划第三/四阶段）：Solution 不再理解需求，只读档案。

    以前这里还会：① 复用/调用统一抽取器重新抽一遍；② fallback 到 LLM 解析 JSON；
    ③ 把结果 merge 进旧 requirement 字典 —— 等于第二套需求理解。现在全部去掉。
    """
    from ....models.requirement import RequirementProfile
    from ....rag.readiness import check_recommendation_ready

    messages = state.get("messages") or []
    all_text = " ".join(
        (m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")) or ""
        for m in messages
    )
    keywords = _extract_keywords(all_text)

    profile = state.get("requirement_profile")
    if not isinstance(profile, RequirementProfile):
        logger.warning(
            "understand: 没有 RequirementProfile → REQUIREMENT_NOT_READY（不再自己重建需求）"
        )
        return {
            **state,
            "requirement_not_ready": True,
            "info_sufficient": False,
            "missing_info": [],
            "search_keywords": keywords,
        }

    decision = check_recommendation_ready(profile)
    logger.info(
        "understand: 只读档案 → ready=%s missing=%s", decision.ready, decision.missing
    )
    return {
        **state,
        "requirement_not_ready": False,
        "info_sufficient": bool(decision.ready),
        "missing_info": list(decision.missing),
        "search_keywords": keywords,
    }


def infer_parameters_node(state: SolutionState) -> SolutionState:
    """Phase 5：技术参数推断统一委托给 ``parameter_inference`` 的权威规则表。

    旧实现里这里还有一份独立的点间距/亮度/租赁规则（与
    ``parameter_inference`` 的规则冲突），现已删除 —— 工程推断只有一个入口。
    """
    from ....rag.parameter_inference import parameter_inference_node

    return parameter_inference_node(state)


def clarify_node(state: SolutionState) -> SolutionState:
    """Ask the user to clarify missing requirements."""
    # ── v2.2：追问只有一个出处 —— 确定性 Gate（字段决策状态） ────────────────
    # 历史遗留分支会用 LLM 给的 missing_info 拼问句，于是出现实测日志里那句
    # "To recommend the right products for you, could you tell me: distance?"
    # ——把内部槽位名直接抛给客户。这里先做确定性判断：
    #   1) 有 RequirementProfile → 一律用 Gate 的问句；
    #   2) 确定性 Gate 已经 ready（例如 reflect 之后又走到这里）→ 本轮不问。
    # 只有完全没有 profile 时才会落到下面的旧逻辑。
    from ....models.requirement import RequirementProfile
    from ....rag.readiness import (
        check_recommendation_ready,
        first_missing_slot,
        question_for,
    )

    _profile = state.get("requirement_profile")
    if isinstance(_profile, RequirementProfile):
        _decision = check_recommendation_ready(_profile)
        if _decision.ready:
            logger.info("Clarify: deterministic gate is ready -> no question this turn")
            return {
                **state,
                "pending_question": "",
                "next_action": "end",
                "waiting_for_clarification": False,
            }
        _question = (
            _decision.next_question
            or str(state.get("pending_question") or "").strip()
            or question_for(first_missing_slot(_decision.missing) or "", "en", 0)
            or ""
        )
        if _question:
            logger.info("Clarify: asking via deterministic gate: %s", _question)
            return {
                **state,
                "pending_question": _question,
                "recommendation": _question,
                "next_action": "end",
                "waiting_for_clarification": True,
            }
    # ── v2.0 Phase 4：Recommendation Ready Gate 已给出明确追问 → 直接使用 ──
    # 旧启发式（按面积/人数反推视距等）保留在下面，但 Gate 判定为"未就绪"时
    # 以 Gate 的问题为准，避免两套逻辑互相覆盖。
    gate = state.get("recommendation_gate") or {}
    gate_question = state.get("pending_question")
    if gate.get("ready") is False and gate_question:
        logger.info("Clarify: 使用 Recommendation Ready Gate 的追问: %s", gate_question)
        return {
            **state,
            "pending_question": gate_question,
            "recommendation": gate_question,
            "next_action": "end",
            "waiting_for_clarification": True,
        }

    missing = state.get("missing_info", [])
    
    if not missing:
        return {**state, "next_action": "understand"}
    
    # 旧分支最后一道保险：槽位名必须翻译成人话，绝不允许把内部字段名抛给客户
    # （实测日志："... could you tell me: distance?"）
    from ....rag.readiness import human_label, question_for

    raw_slot = str(missing[0]).strip()
    slot_alias = {
        "distance": "viewing_distance",
        "viewing_distance_m": "viewing_distance",
        "width": "width",
        "height": "height",
        "size": "size",
        "pitch": "pixel_pitch",
    }.get(raw_slot.lower(), raw_slot)
    question = question_for(slot_alias, "en", 0) or (
        "To recommend the right products for you, could you tell me: "
        f"{human_label(raw_slot) or 'a bit more about your setup'}?"
    )
    
    return {
        **state,
        "pending_question": question,
        "recommendation": question,
        "next_action": "end",
        "waiting_for_clarification": True,
    }


def product_question_node(state: SolutionState) -> SolutionState:
    """Handle product-specific questions."""
    # Delegate to others node with product context
    from .others import others_node
    return others_node(state)


def recommendation_gate_node(state: SolutionState) -> SolutionState:
    """v2.0 Phase 4：Recommendation Ready Gate。

    把"进入 Solution Agent"与"允许执行产品推荐"分开：
      - ready → next_action="retrieve"，继续走检索 / 选型 / 计算 / 表达
      - 未 ready → next_action="clarify"，本轮只问一个关键问题
    """
    from ....models.requirement import RequirementProfile
    from ....rag.readiness import check_recommendation_ready

    profile = state.get("requirement_profile")
    if not isinstance(profile, RequirementProfile):
        # 【关键】只信任"客户原话"的确定性解析结果。
        # 旧实现直接 `from_legacy(state["requirement"])`，会把 Solution Agent
        # 的 LLM 自行补出的 indoor/outdoor/distance 当成客户确认的事实，
        # 导致"只知道室内外 + 场景"也能打开 Gate。现在：
        #   - 确定性解析用户消息 → confirmed
        #   - LLM 提取的 purpose（场景原话）→ 作为 confirmed 补进来
        #   - LLM 提取的其它工程参数 → 一律忽略
        from ....rag.query_understanding import extract_slots, merge_slots

        confirmed_slots: Dict[str, Any] = {}
        for msg in state.get("messages", []) or []:
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", "")
            if role in ("user", "human"):
                content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
                if content:
                    confirmed_slots = merge_slots(
                        confirmed_slots, extract_slots(str(content))
                    )

        profile = RequirementProfile.from_slots(
            confirmed_slots, explicit_keys=set(confirmed_slots)
        )
        logger.debug("\n%s", profile.describe())

    decision = check_recommendation_ready(
        profile, variant_seed=len(state.get("messages", []) or [])
    )
    logger.info(
        "RecommendationGate: ready=%s missing=%s reason=%s",
        decision.ready, decision.missing, decision.reason,
    )

    updates: Dict[str, Any] = {
        "requirement_profile": profile,
        "recommendation_gate": decision.to_dict(),
    }
    if decision.ready:
        updates["next_action"] = "retrieve"
        updates["pending_question"] = ""
    else:
        updates["next_action"] = "clarify"
        updates["pending_question"] = decision.next_question or ""
        # 兼容旧 clarify 启发式使用的 missing_info 结构
        updates["missing_info"] = [
            {
                "environment": "whether it is indoors or outdoors",
                "purpose": "the application scenario",
                "installation_or_distance": "fixed installation or rental, and viewing distance",
                "display_type": "the display type",
                "width": "the screen width",
                "height": "the screen height",
            }.get(slot, slot)
            for slot in decision.missing
        ]
        updates["waiting_for_clarification"] = True
    return {**state, **updates}


def detect_intent(message: str, state: SolutionState = None) -> str:
    """Detect the user's intent based on the message content.

    Phase 5：统一入口 —— 规则优先，规则判不出时再交给 LLM。
    返回 "recommendation" / "product_question" / "conversation" / "others"。
    """
    from ....rag.parameter_inference import detect_intent as _detect_intent

    return _detect_intent(message, use_llm=True)
