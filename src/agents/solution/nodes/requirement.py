"""Requirement extraction and understanding nodes."""
import json
import logging
import re
from typing import Dict, Any, List, Optional

from ..state import SolutionState
from ....core.llm import get_llm
from ....utils.ifp_intent import has_ifp_intent

logger = logging.getLogger(__name__)

def infer_display_type(requirement: Dict[str, Any], messages: List) -> Optional[str]:
    """Infer display type (LED/IFP) from requirement and conversation context.
    
    Returns "IFP" if the user seems to want an interactive flat panel (conference room
    with handwriting/interaction needs), otherwise "LED".
    """
    purpose = requirement.get("purpose", "")
    user_text = " ".join(
        msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        for msg in (messages or [])
        if (isinstance(msg, dict) and msg.get("role") in ("user", "human")
            ) or (not isinstance(msg, dict) and getattr(msg, "type", "") in ("user", "human"))
    )
    
    if has_ifp_intent(requirement, user_text=user_text):
        logger.info("Inferred display_type=IFP from IFP intent")
        return "IFP"
    
    # Default to LED for non-IFP scenarios
    logger.info("Inferred display_type=LED (default)")
    return "LED"


def _normalize_history(messages: List) -> List[Dict[str, str]]:
    """Normalize messages to role/content dicts."""
    result = []
    for msg in messages:
        if isinstance(msg, dict):
            result.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        else:
            result.append({"role": getattr(msg, "type", "user"), "content": getattr(msg, "content", "")})
    return result


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


def _build_requirement_prompt(requirement: Dict[str, Any], conversation: str) -> str:
    """Build the requirement extraction prompt."""
    current_req = ""
    if requirement:
        req_parts = [f"{k}: {v}" for k, v in requirement.items() if v]
        current_req = "\n".join(req_parts)

    return f"""从对话中提取或更新 LED/LCD 显示产品需求：

当前需求：
{current_req or '无'}

对话历史：
{conversation}

要求：
1. 只提取用户明确说过的内容，不要猜测
2. 布尔字段（如 indoor/outdoor）用 true/false/null
3. 返回标准 JSON 格式：
{{
    "indoor": true/false/null,
    "outdoor": true/false/null,
    "distance": "可视距离描述",
    "purpose": "使用场景",
    "size": "尺寸要求",
    "brightness": "亮度要求",
    "resolution": "分辨率要求",
    "display_type": "LED/LCD/BOTH/null",
    "info_sufficient": true/false,
    "missing_info": ["缺失的关键项"]
}}

只返回 JSON，不要其他内容。"""


def _parse_requirement_response(response: str) -> Dict[str, Any]:
    """Parse the LLM response into a requirement dict."""
    try:
        # Strip markdown if present
        content = response.strip()
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse requirement JSON: {e}")
        return {"info_sufficient": False, "missing_info": ["需求提取失败"]}


def understand_node(state: SolutionState) -> SolutionState:
    """Understand and extract requirements from the conversation.

    Phase 2 优化：如果上游（Sales Agent）已经传入了充分的 requirements，
    且 state 中标记 ``requirements_skip_understand`` 为 True，则**跳过 LLM**，
    直接复用已提取的需求，避免重复理解同一份需求。
    """
    messages = _normalize_history(state.get("messages", []))
    current_req = state.get("requirement", {}) or {}

    # Phase 2: 跳过的判定
    # 1) 由 Runner 显式标记（外部认为 requirements 已足够）
    if state.get("requirements_skip_understand"):
        logger.info(
            "understand: SKIPPED (state flag set; reusing %d fields from Sales Agent)",
            len(current_req),
        )
        return {
            **state,
            "info_sufficient": True,
            "missing_info": [],
            "search_keywords": _extract_keywords(" ".join(
                m.get("content", "") for m in messages
            )),
        }
    # 2) 当前已有充分字段（indoor/outdoor + usage/purpose + display_type）
    key_fields = ("usage", "purpose", "indoor", "outdoor", "display_type")
    filled = sum(1 for k in key_fields if current_req.get(k))
    if filled >= 3:
        logger.info(
            "understand: SKIPPED (requirements already have %d/5 key fields)",
            filled,
        )
        return {
            **state,
            "info_sufficient": True,
            "missing_info": [],
            "search_keywords": _extract_keywords(" ".join(
                m.get("content", "") for m in messages
            )),
        }

    # 3) Phase 4/13：复用统一 RequirementExtractor 的同一轮语义结果
    #    Sales Agent 已经把这条消息的语义理解缓存下来了 → 这里不再重复调 LLM。
    try:
        from ....core.requirement_extractor import get_requirement_extractor
        from ....models.legacy_adapter import profile_to_solution_requirement
        from ....rag.readiness import check_recommendation_ready

        extractor = get_requirement_extractor()
        if extractor._cached_semantic(state.get("current_message", "")) is not None:
            profile = extractor.extract(state.get("current_message", ""), use_llm=False)
            if profile.purpose or profile.environment or profile.display_type:
                decision = check_recommendation_ready(profile)
                logger.info(
                    "understand: 复用统一 Extractor 的语义结果（不再调 LLM）missing=%s",
                    decision.missing,
                )
                return {
                    **state,
                    "requirement": profile_to_solution_requirement(profile),
                    "requirement_profile": profile,
                    "info_sufficient": decision.ready,
                    "missing_info": list(decision.missing),
                    "search_keywords": _extract_keywords(
                        " ".join(m.get("content", "") for m in messages)
                    ),
                }
    except Exception as exc:  # pragma: no cover - 防御式
        logger.warning("understand: extractor reuse failed, fallback to LLM: %s", exc)

    # Build conversation context
    conversation_parts = []
    for msg in messages[-6:]:  # Last 6 messages
        role = msg.get("role", "user")
        content = msg.get("content", "")[:200]
        conversation_parts.append(f"{role}: {content}")
    conversation = "\n".join(conversation_parts)

    prompt = _build_requirement_prompt(state.get("requirement", {}), conversation)

    try:
        llm = get_llm(temperature=0.1)
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        parsed = _parse_requirement_response(content)

        logger.info(f"Understand: LLM parsed={parsed}, current_req before merge={state.get('requirement', {})}")

        # Merge with existing requirement (preserve existing values if LLM returns empty/null)
        current_req = state.get("requirement", {})
        new_req = {**current_req}
        for key, value in parsed.items():
            if key not in ("info_sufficient", "missing_info"):
                # Only overwrite if LLM returns a non-empty value
                if value is not None and value != "" and value != []:
                    new_req[key] = value
                # If LLM returns None/empty but we already have a value, keep existing
                elif key not in current_req or not current_req.get(key):
                    # Only set to None/empty if not already present
                    if value is None or value == "":
                        if key not in new_req:
                            new_req[key] = value

        # 【不再从容纳人数反推视距】这类"根据场景推测参数"是导致过早推荐/选错点间距的根因。
        # 视距、固装租赁等工程参数只能来自客户原话的确定性解析；没有就由 Gate 询问。
        all_text = " ".join([m.get("content", "") for m in messages])

        # Update state
        updates = {
            "requirement": new_req,
            "info_sufficient": parsed.get("info_sufficient", False),
            "missing_info": parsed.get("missing_info", []),
        }

        # Extract search keywords
        updates["search_keywords"] = _extract_keywords(all_text)

        logger.info(f"Understand: requirement={new_req}, info_sufficient={parsed.get('info_sufficient')}")
        return {**state, **updates}

    except Exception as e:
        logger.error(f"Understand node error: {e}")
        return {**state, "info_sufficient": False, "missing_info": ["需求理解出错"]}


def infer_parameters_node(state: SolutionState) -> SolutionState:
    """Phase 5：技术参数推断统一委托给 ``parameter_inference`` 的权威规则表。

    旧实现里这里还有一份独立的点间距/亮度/租赁规则（与
    ``parameter_inference`` 的规则冲突），现已删除 —— 工程推断只有一个入口。
    """
    from ....rag.parameter_inference import parameter_inference_node

    return parameter_inference_node(state)


def clarify_node(state: SolutionState) -> SolutionState:
    """Ask the user to clarify missing requirements."""
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
    requirement = state.get("requirement", {})
    purpose = requirement.get("purpose", "")
    inferred_size = state.get("inferred_screen_size", "")
    distance = requirement.get("distance", "")
    
    logger.info(f"Clarify: missing={missing}, inferred_size={inferred_size}, distance={distance}")
    
    # 如果有 inferred_screen_size，自动用它填充 size，无需再问用户
    size_missing = any(
        kw in m.lower() for m in missing for kw in ["尺寸", "size", "屏幕", "大小"]
    )
    if size_missing and inferred_size:
        logger.info(f"Using inferred size '{inferred_size}' for missing '尺寸'")
        new_requirement = {**requirement, "size": inferred_size}
        new_missing = [
            m for m in missing
            if not any(kw in m.lower() for kw in ["尺寸", "size", "屏幕", "大小"])
        ]
        state = {**state, "requirement": new_requirement, "missing_info": new_missing}
        missing = new_missing

    # 如果已经有 distance，自动从 missing 中移除"可视距离"询问
    distance_missing = any(
        kw in m.lower() for m in missing for kw in ["可视距离", "视距", "距离", "多远"]
    )
    if distance_missing and distance:
        logger.info(f"Using existing distance '{distance}' for missing '可视距离'")
        new_missing = [
            m for m in missing
            if not any(kw in m.lower() for kw in ["可视距离", "视距", "距离", "多远"])
        ]
        state = {**state, "missing_info": new_missing}
        missing = new_missing

    # 如果有亮度缺失，根据室内/室外推断，无需再问用户
    brightness_missing = any(
        kw in m.lower() for m in missing for kw in ["亮度", "brightness", "nit", "尼特"]
    )
    if brightness_missing:
        inferred_brightness = None
        if requirement.get("indoor"):
            inferred_brightness = "300-500nit（室内标准亮度）"
            logger.info("Inferred brightness for indoor: 300-500nit")
        elif requirement.get("outdoor"):
            inferred_brightness = "4000-6000nit（户外高亮）"
            logger.info("Inferred brightness for outdoor: 4000-6000nit")
        if inferred_brightness:
            new_requirement = {**requirement, "brightness": inferred_brightness}
            new_missing = [
                m for m in missing
                if not any(kw in m.lower() for kw in ["亮度", "brightness", "nit", "尼特"])
            ]
            state = {**state, "requirement": new_requirement, "missing_info": new_missing}
            missing = new_missing

    # 如果有屏幕类型缺失，自动推断（会议室+手写→IFP，其他→LED）
    display_type_missing = any(
        kw in m.lower() for m in missing for kw in ["屏幕类型", "display_type", "屏类型", "led", "lcd", "ifp"]
    )
    if display_type_missing and not requirement.get("display_type"):
        inferred_dt = infer_display_type(requirement, state.get("messages", []))
        if inferred_dt:
            logger.info(f"Using inferred display_type='{inferred_dt}' for missing '屏幕类型'")
            new_requirement = {**requirement, "display_type": inferred_dt}
            new_missing = [
                m for m in missing
                if not any(kw in m.lower() for kw in ["屏幕类型", "display_type", "屏类型", "led", "lcd", "ifp"])
            ]
            state = {**state, "requirement": new_requirement, "missing_info": new_missing}
            missing = new_missing

    # 如果有分辨率缺失，从面积/距离自动推断，无需再问用户
    # 推理链：面积 → 视距 → 点间距 → 分辨率等级
    resolution_missing = any(
        kw in m.lower() for m in missing for kw in ["分辨率", "resolution", "清晰度", "像素"]
    )
    if resolution_missing:
        inferred_resolution = None
        # 优先用已有的点间距范围推断分辨率等级
        pitch_max = state.get("inferred_pixel_pitch_max_mm") or state.get("inferred_pixel_pitch_min_mm")
        size_val = requirement.get("size") or inferred_size
        dist_val = distance

        if pitch_max:
            # 根据点间距推断分辨率等级
            if pitch_max <= 1.0:
                inferred_resolution = "4K（超高清）"
            elif pitch_max <= 1.8:
                inferred_resolution = "2K-4K"
            elif pitch_max <= 3.0:
                inferred_resolution = "1080P-2K"
            elif pitch_max <= 5.0:
                inferred_resolution = "720P-1080P"
            else:
                inferred_resolution = "SD（标清）"
            logger.info(f"Inferred resolution '{inferred_resolution}' from pitch_max={pitch_max}")
        elif dist_val and size_val:
            # 根据视距推断点间距，再推断分辨率
            numbers = re.findall(r'\d+\.?\d*', dist_val)
            if numbers:
                dist = float(numbers[0])
                if dist > 100:
                    dist /= 1000
                # 点间距 ≈ 视距/1000（毫米）
                pitch_est = dist / 1000 * 1000  # mm
                if pitch_est < 1.5:
                    inferred_resolution = "4K（超高清）"
                elif pitch_est < 2.5:
                    inferred_resolution = "2K-4K"
                elif pitch_est < 4.0:
                    inferred_resolution = "1080P-2K"
                else:
                    inferred_resolution = "720P-1080P"
                logger.info(f"Inferred resolution '{inferred_resolution}' from distance={dist}m")

        if inferred_resolution:
            new_requirement = {**requirement, "resolution": inferred_resolution}
            new_missing = [
                m for m in missing
                if not any(kw in m.lower() for kw in ["分辨率", "resolution", "清晰度", "像素"])
            ]
            state = {**state, "requirement": new_requirement, "missing_info": new_missing}
            missing = new_missing
    
    # Special handling for conference room scenarios
    # 条件：会议室场景 且 缺少 distance（无论 size 有没有值）
    conference_keywords = ["会议", "培训", "教室", "报告"]
    is_conference = any(kw in purpose for kw in conference_keywords)

    if is_conference and not distance:
        # 【关键修复】先检查是否有 IFP 意图（会议室+手写/触控/白板等）
        # 如果有，直接锁定 IFP，跳过面积/视距询问，直接进入推荐
        user_text_for_ifp = " ".join(
            msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            for msg in state.get("messages", [])
            if (isinstance(msg, dict) and msg.get("role") in ("user", "human"))
            or (not isinstance(msg, dict) and getattr(msg, "type", "") in ("user", "human"))
        )
        if has_ifp_intent(requirement, user_text=user_text_for_ifp):
            logger.info("Conference room + IFP intent detected → locking display_type=IFP, skipping distance/size clarification")
            new_requirement = {**requirement, "display_type": "IFP"}
            new_missing = [
                m for m in missing
                if not any(kw in m.lower() for kw in [
                    "尺寸", "size", "屏幕", "大小", "面积", "视距", "距离", "可视距离",
                    "屏幕类型", "display_type", "屏类型",
                ])
            ]
            return {
                **state,
                "requirement": new_requirement,
                "missing_info": new_missing,
                "next_action": "understand",
            }

        # 尝试从消息中提取面积（平米）
        area_sqm = None
        last_message = state.get("messages", [])[-1] if state.get("messages") else {}
        msg_text = last_message.get("content", "") if isinstance(last_message, dict) else ""
        
        # 优先用 requirement 中已有的 size（可能是数字平米）
        size_str = requirement.get("size") or ""
        area_match = re.search(r'(\d+\.?\d*)\s*(?:平|平方米|㎡|平方)', size_str)
        if area_match:
            area_sqm = float(area_match.group(1))
            logger.info(f"Detected area from requirement: {area_sqm} sqm")
        else:
            # 从消息中再尝试提取一次
            area_match = re.search(r'(\d+\.?\d*)\s*(?:平|平方米|㎡|平方)', msg_text)
            if area_match:
                area_sqm = float(area_match.group(1))
                logger.info(f"Detected area from message: {area_sqm} sqm")
        
        # 如果有面积，从面积推算视距和尺寸
        if area_sqm and area_sqm > 0:
            # 粗略估算：假设会议室近似正方形，视距 ≈ √(面积) 的 1-1.5 倍
            # 3平米 → ~2m, 10平米 → ~3.5m, 20平米 → ~5m, 30平米 → ~6m
            if area_sqm <= 5:
                inferred_dist = "约2米"
                inferred_size = "65-75英寸"
            elif area_sqm <= 15:
                inferred_dist = "约3-4米"
                inferred_size = "75-86英寸"
            elif area_sqm <= 25:
                inferred_dist = "约4-5米"
                inferred_size = "86-98英寸"
            else:
                inferred_dist = "约5米以上"
                inferred_size = "98英寸以上"
            
            logger.info(f"From area {area_sqm}sqm → distance: {inferred_dist}, size: {inferred_size}")
            new_requirement = {**requirement, "distance": inferred_dist}
            # 如果原来没有 size，用推断值填充
            if not requirement.get("size"):
                new_requirement["size"] = inferred_size
            new_missing = [
                m for m in missing
                if not any(kw in m.lower() for kw in ["尺寸", "size", "屏幕", "大小", "面积", "视距", "距离"])
            ]
            return {
                **state,
                "requirement": new_requirement,
                "missing_info": new_missing,
                "next_action": "understand",
            }
        
        # 没有面积也没有视距 → 直接问用户
        question = "会议室大概有多大面积？或者最远的人离屏幕大概几米？这样好帮您选合适的点间距和尺寸。"
        return {
            **state,
            "pending_question": question,
            "recommendation": question,
            "next_action": "end",
            "waiting_for_clarification": True,
        }
    
    if not missing:
        return {**state, "next_action": "understand"}
    
    question = f"To recommend the right products for you, could you tell me: {missing[0]}?"
    
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

        legacy = RequirementProfile.from_legacy(state.get("requirement") or {})
        if legacy.purpose and not profile.purpose:
            # 场景(purpose)是客户自由描述的客观事实，允许来自 LLM 提取；
            # 但其它工程参数一概不采用。
            profile.purpose = legacy.purpose
            profile.sources["purpose"] = "confirmed"

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
