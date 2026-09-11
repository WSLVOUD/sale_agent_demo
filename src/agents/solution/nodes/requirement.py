"""Requirement extraction and understanding nodes."""
import json
import logging
import re
from typing import Dict, Any, List, Optional

from ..state import SolutionState
from ....core.llm import get_llm
from ....utils.ifp_intent import has_ifp_intent

logger = logging.getLogger(__name__)

# 中文数字转换表
_CHINESE_DIGITS = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "百": 100, "千": 1000,
}


def _chinese_to_number(text: str) -> int:
    """将中文数字转换为整数"""
    if not text:
        return 0
    result = 0
    temp = 0
    for char in text:
        if char in _CHINESE_DIGITS:
            val = _CHINESE_DIGITS[char]
            if val >= 100:
                result = (result or 1) * val
                temp = 0
            else:
                temp = temp * 10 + val
    result += temp
    return result


def _parse_capacity_people(text: str) -> int:
    """从文本中解析容纳人数"""
    patterns = [
        r'(?:大约|大概|约|差不多)?\s*能容纳\s*([零一二两三四五六七八九十百千\d]+)\s*个人?',
        r'([零一二两三四五六七八九十百千\d]+)\s*个人?\s*(?:的|左右|上下)',
        r'([零一二两三四五六七八九十百千]+)\s*人\s*(?:左右|上下)?',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            # Try Arabic digits first
            digit_match = re.search(r'(\d+)', match.group(1))
            if digit_match:
                return int(digit_match.group(1))
            # Then try Chinese numerals
            return _chinese_to_number(match.group(1))
    return 0


def _infer_size_from_capacity(text: str, requirement: Dict[str, Any]) -> Dict[str, Any]:
    """从容纳人数推断尺寸和视距"""
    # Treat both None and empty string as "missing" for inference
    size_val = requirement.get("size") or ""
    distance_val = requirement.get("distance") or ""
    if size_val and distance_val:
        return requirement
    
    inferred = dict(requirement)
    people = _parse_capacity_people(text)
    
    if people > 0:
        # 会议室/教室按每人1.5平米估算，加上走道余量
        area = people * 1.5 + 5
        if not inferred.get("size"):
            inferred["size"] = f"约{area:.0f}平米"
            logger.info(f"Inferred size '{inferred['size']}' from capacity: {people}人")
        
        # 观看距离推断
        if not inferred.get("distance"):
            if people <= 10:
                inferred["distance"] = "2-4米"
            elif people <= 20:
                inferred["distance"] = "3-5米"
            elif people <= 50:
                inferred["distance"] = "4-7米"
            else:
                inferred["distance"] = "5-8米"
            logger.info(f"Inferred distance '{inferred['distance']}' from capacity: {people}人")
    
    return inferred


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

        # 规则推理补充：从容纳人数等推断尺寸和视距
        all_text = " ".join([m.get("content", "") for m in messages])
        new_req = _infer_size_from_capacity(all_text, new_req)

        logger.info(f"Understand: after _infer_size_from_capacity, new_req={new_req}")

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
    """Infer technical parameters from natural language requirements."""
    requirement = state.get("requirement", {})

    # Normalize indoor/outdoor to string values for consistent vector store matching
    # (The vector store stores these as strings "True"/"False")
    indoor = requirement.get("indoor")
    outdoor = requirement.get("outdoor")
    if indoor is True or indoor == "True":
        requirement = {**requirement, "indoor": "True", "outdoor": "False"}
    elif outdoor is True or outdoor == "True":
        requirement = {**requirement, "indoor": "False", "outdoor": "True"}
    else:
        requirement = {**requirement, "indoor": "False", "outdoor": "False"}
    
    # Phase 11: Fix - use proper boolean check (strings "True"/"False" are truthy!)
    def _bool(val):
        return str(val).lower() == "true"

    # Brightness inference
    brightness_min = None
    brightness_max = None
    
    if _bool(requirement.get("outdoor")) and not _bool(requirement.get("indoor")):
        brightness_min = 4500  # Outdoor requires high brightness
    elif _bool(requirement.get("indoor")) and not _bool(requirement.get("outdoor")):
        brightness_max = 800  # Indoor should not be too bright
    
    # Pixel pitch and screen size inference based on viewing distance
    pitch_min = None
    pitch_max = None
    inferred_screen_size = None
    
    distance_text = requirement.get("distance", "")
    if distance_text:
        # Extract numbers from distance text (handle "4m", "4米", "4米视距" etc.)
        numbers = re.findall(r'\d+\.?\d*', distance_text)
        logger.info(f"Infer parameters: distance_text={distance_text}, numbers={numbers}")
        if numbers:
            # Get the first meaningful number (distance in meters)
            dist = float(numbers[0])
            # If number is > 100, assume it's in mm, convert to meters
            if dist > 100:
                dist = dist / 1000
            logger.info(f"Infer parameters: dist={dist}m")
            
            # Rough inference: pixel pitch should be ~1/1000 of viewing distance
            if dist < 3:
                pitch_max = 2.0  # Very close viewing needs fine pitch
            elif dist < 10:
                pitch_max = 4.0
            elif dist < 30:
                pitch_max = 10.0
            else:
                pitch_max = 16.0
            
            # Infer screen size based on viewing distance
            # Formula: optimal viewing distance ≈ 2-3x screen diagonal
            # For 4m distance: screen diagonal ≈ 1.3m - 2m (50" - 80")
            # Common LCD sizes: 55", 65", 75", 86", 98", 110"
            # For 4m: recommended 75" or 86"
            # For 3m: recommended 65" or 75"
            # For 5m+: recommended 86" or larger
            if dist <= 3:
                inferred_screen_size = "65-75英寸"  # 65-75 inch
            elif dist <= 4:
                inferred_screen_size = "75-86英寸"  # 75-86 inch
            elif dist <= 6:
                inferred_screen_size = "86-98英寸"  # 86-98 inch
            else:
                inferred_screen_size = "98英寸以上"  # 98+ inch
            logger.info(f"Infer parameters: inferred_screen_size={inferred_screen_size}")
    
    # Rental inference
    purpose = requirement.get("purpose", "")
    is_rental = None
    if any(kw in purpose for kw in ["租赁", "租用", "临时", "活动", "演出", "演唱会"]):
        is_rental = True

    return {
        **state,
        "inferred_brightness_min_nit": brightness_min,
        "inferred_brightness_max_nit": brightness_max,
        "inferred_pixel_pitch_min_mm": pitch_min,
        "inferred_pixel_pitch_max_mm": pitch_max,
        "inferred_screen_size": inferred_screen_size,
        "inferred_is_rental": is_rental,
    }


def clarify_node(state: SolutionState) -> SolutionState:
    """Ask the user to clarify missing requirements."""
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


def detect_intent(message: str, state: SolutionState = None) -> str:
    """Detect the user's intent based on the message content.
    
    Returns intent type:
    - "recommendation": user is describing needs or asking for product recommendations
    - "product_question": user is asking about specific product features/specs
    - "conversation": casual conversation
    - "others": other questions outside the standard flow
    """
    from src.core.llm import get_llm
    
    intent_prompt = f"""Analyze the user message and return the intent type:
- recommendation: user is describing needs or asking for product recommendations ("I need a meeting room display", "help me pick one", "need a rental screen")
- product_question: user is asking about specific product specs, features, capabilities, model differences
- conversation: greetings, casual chat, etc.
- others: company info, business hours, warranty policy, business process questions, etc.

User message: {message}

Return the intent type only, nothing else."""
    
    try:
        llm = get_llm(temperature=0)
        response = llm.invoke(intent_prompt)
        intent = response.content.strip().lower()
        
        # Validate intent
        valid_intents = ["recommendation", "product_question", "conversation", "others"]
        if intent not in valid_intents:
            return "others"
        return intent
    except Exception as e:
        logger.warning(f"detect_intent error: {e}")
        return "others"
