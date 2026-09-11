"""script_generator node - generate the sales response text."""
import logging
import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import SalesState
from ....core.llm import get_llm

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


def _parse_number(text: str) -> int:
    """解析字符串中的数字，支持中文和阿拉伯数字"""
    digit_match = re.search(r'(\d+)', text)
    if digit_match:
        return int(digit_match.group(1))
    cn_match = re.search(r'[一二两三四五六七八九十百千]+', text)
    if cn_match:
        return _chinese_to_number(cn_match.group())
    return 0


GREETING_PROMPT = """You are a sales advisor for LED and LCD display products, speaking with a customer face-to-face.

Requirements:
1. Simple greeting, one sentence
2. Friendly tone
3. Naturally ask about their needs
4. Plain text only, no markdown

Output the reply directly:"""


# Requirement question generator — only asks about key missing fields
REQUIREMENT_QUESTIONS = {
    "usage": "What kind of scenario will this be used in? Meeting room, classroom, retail store, advertising, etc.?",
    # Note: area, viewing distance, brightness, and resolution can be inferred — no need to ask the user
}


def _get_next_question(requirements: dict) -> str:
    """Return the next most important question based on current requirements."""
    # Priority order
    priority = ["usage", "location_type", "size", "viewing_distance", "brightness", "resolution"]

    # Infer missing fields from known info
    inferred = {}
    if requirements.get("location_type"):
        loc = requirements.get("location_type", "")
        if loc in ("indoor", "室内") and "brightness" not in requirements:
            inferred["brightness"] = True
        if loc in ("outdoor", "室外") and "brightness" not in requirements:
            inferred["brightness"] = True

    for key in priority:
        if key not in requirements or not requirements.get(key):
            return REQUIREMENT_QUESTIONS.get(key, "What else do you need for this scenario?")
    
    return "还有什么其他要求吗？"


def _infer_from_message(message: str, requirements: dict) -> dict:
    """从用户消息中推断并补充需求。"""
    text = message.lower()
    inferred = dict(requirements)
    
    # Infer indoor / outdoor
    if "indoor" not in requirements and "outdoor" not in requirements:
        indoor_keywords = ["indoor", "indoors", "office", "meeting room", "classroom", "retail", "store", "shop", "exhibition"]
        outdoor_keywords = ["outdoor", "outdoors", "open air", "facade", "plaza", "square"]

        for kw in indoor_keywords:
            if kw in text:
                inferred["indoor"] = True
                inferred["location_type"] = "indoor"
                break
        for kw in outdoor_keywords:
            if kw in text:
                inferred["outdoor"] = True
                inferred["location_type"] = "outdoor"
                break

    # Infer screen type
    if not requirements.get("display_type"):
        # IFP keywords
        ifp_keywords = ["touch", "interactive", "whiteboard", "annotation", "IFP", "interactive"]
        # LED keywords
        led_keywords = ["stage", "concert", "sports", "curtain", "rental", "advertising"]
        # LCD keywords
        lcd_keywords = ["digital signage", "signage", "kiosk"]

        if any(kw in text for kw in ifp_keywords):
            inferred["display_type"] = "IFP"
        elif any(kw in text for kw in led_keywords):
            inferred["display_type"] = "LED"
        elif any(kw in text for kw in lcd_keywords):
            inferred["display_type"] = "LCD"

    # Infer area / size
    if "size" not in requirements or not requirements.get("size"):
        import re
        # Match "X sqm"
        area_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:sqm|square meters?|平方米|平米)', text)
        if area_match:
            inferred["size"] = f"{area_match.group(1)} sqm"

        # Match "3m x 4m"
        size_match = re.search(r'(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*m', text)
        if size_match:
            inferred["size"] = f"{size_match.group(1)}m x {size_match.group(2)}m"

        # Match viewing distance "about Xm"
        dist_match = re.search(r'(?:about|approx|around)?\s*(\d+(?:\.\d+)?)\s*m', text)
        if dist_match and "viewing_distance" not in requirements:
            inferred["viewing_distance"] = f"{dist_match.group(1)}m"

        # Infer area from occupancy (meeting room / classroom: ~2.5 sqm per person)
        patterns = [
            r'(?:about|approx|around)?\s*(?:holds?|accommodates?)\s*([零一二两三四五六七八九十百千\d]+)\s*(?:people|persons?|people)',
            r'([零一二两三四五六七八九十百千\d]+)\s*(?:people|persons?|people)\s*(?:room|classroom|meeting)',
            r'(?:holds?|accommodates?)\s*([零一二两三四五六七八九十百千\d]+)\s*(?:people|persons?)',
        ]
        people = 0
        for pattern in patterns:
            occupancy_match = re.search(pattern, text)
            if occupancy_match:
                people = _parse_number(occupancy_match.group(1))
                if people > 0:
                    break

        if people > 0:
            area = people * 2.5 + 5
            inferred["size"] = f"approx {area:.0f} sqm"
            if "viewing_distance" not in requirements:
                if people <= 10:
                    inferred["viewing_distance"] = "2-4m"
                elif people <= 20:
                    inferred["viewing_distance"] = "3-5m"
                else:
                    inferred["viewing_distance"] = "4-6m"

    # Infer resolution from viewing distance
    if "resolution" not in requirements or not requirements.get("resolution"):
        dist_val = inferred.get("viewing_distance") or requirements.get("viewing_distance", "")
        if dist_val:
            import re
            numbers = re.findall(r'\d+\.?\d*', dist_val)
            if numbers:
                dist = float(numbers[0])
                if '-' in dist_val or '~' in dist_val:
                    all_nums = re.findall(r'\d+\.?\d*', dist_val)
                    if len(all_nums) >= 2:
                        dist = (float(all_nums[0]) + float(all_nums[1])) / 2
                if dist > 100:
                    dist /= 1000
                if dist <= 3:
                    inferred["resolution"] = "1080P-2K (close-range high clarity)"
                elif dist <= 5:
                    inferred["resolution"] = "1080P (standard clarity)"
                else:
                    inferred["resolution"] = "720P-1080P (mid-to-far distance standard)"

    return inferred


def _strip_markdown(text: str) -> str:
    """Remove common markdown artifacts so the chat reply is plain text."""
    if not text:
        return text
    cleaned = text.replace("**", "").replace("__", "")
    cleaned = re.sub(r"`+([^`]*?)`+", r"\1", cleaned)
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-*+]\s+", "", cleaned)
    return cleaned.strip()


def script_generator(state: SalesState) -> SalesState:
    """Produce the final user-facing response."""
    # Check if router has already processed
    if state.get("solutions") and state.get("response"):
        logger.info("Router has generated response with solutions, keeping it")
        state["next_action"] = "trigger_solution"  # Keep trigger signal for orchestrator
        return state
    
    intent = state["intent"]
    
    # 检查是否需要抑制问候语（首次接待刚完成后）
    suppress_greeting = state.get("suppress_greeting", False)
    
    response_text = ""
    
    logger.info(f"script_generator: intent={intent}, should_generate_solution={state.get('should_generate_solution')}, requirements={state.get('requirements')}, suppress_greeting={suppress_greeting}")
    
    # Greeting - 如果首次接待刚完成则跳过
    if intent == "greeting":
        if suppress_greeting:
            # 首次接待已完成，不需要再次问候，也不要再问姓名（首次接待已问过）
            # 直接询问场景用途，进入需求挖掘流程
            response_text = "What display scenario are you looking into? Meeting room, classroom, retail, advertising...?"
            state["next_action"] = "ask"
            logger.info("Greeting suppressed (first contact completed), asking about scenario directly")
        else:
            prompt = SystemMessage(content=GREETING_PROMPT)
            response = get_llm(temperature=0.3).invoke([prompt, HumanMessage(content=state["current_message"])])
            response_text = response.content.strip()
            state["next_action"] = "ask"
    
    # Need query
    elif intent == "need_query":
        requirements = state.get("requirements", {})
        
        # ===== 提前推理：所有能从已有信息推断的字段 =====
        # 先从当前消息中推断缺失的需求
        current_msg = state.get("current_message", "")
        requirements = _infer_from_message(current_msg, requirements)
        
        # Infer usage if LLM didn't extract it
        if not requirements.get("usage"):
            usage_keywords = {
                "meeting": "meeting room",
                "classroom": "classroom",
                "training": "classroom",
                "retail": "retail",
                "store": "retail",
                "exhibition": "exhibition",
                "hospital": "healthcare",
                "monitoring": "monitoring & control",
                "stage": "stage performance",
                "concert": "concert",
                "sports": "sports venue",
                "advertising": "advertising",
                "curtain": "architectural curtain",
                "rental": "rental events",
            }
            for kw, usage in usage_keywords.items():
                if kw in current_msg:
                    requirements["usage"] = usage
                    break
        
        state["requirements"] = requirements  # 更新state中的requirements
        
        # ===== 关键判断：何时触发推荐 =====
        # 只要有使用场景(usage)，就可以触发推荐
        # 其他参数(面积/亮度/分辨率)AI可以自行推断
        should_trigger = state.get("should_generate_solution", False) or bool(requirements.get("usage"))
        
        logger.info(f"need_query: should_trigger={should_trigger}, requirements={requirements}")
        
        if requirements.get("usage"):
            # Infer location_type if missing
            if not requirements.get("location_type"):
                usage_text = requirements.get("usage", "").lower()
                indoor_usage = ["meeting room", "classroom", "office", "exhibition", "retail", "store", "monitoring", "control"]
                outdoor_usage = ["outdoor", "advertising", "curtain", "sports", "arena"]

                if any(u in usage_text for u in indoor_usage):
                    requirements["location_type"] = "indoor"
                    requirements["indoor"] = True
                elif any(u in usage_text for u in outdoor_usage):
                    requirements["location_type"] = "outdoor"
                    requirements["outdoor"] = True

            # Infer size/viewing distance if missing (AI can infer from context, no need to ask user)
            if not requirements.get("size") and not requirements.get("viewing_distance"):
                usage = requirements.get("usage", "")
                if "meeting" in usage:
                    pass  # Already handled in _infer_from_message
                elif "classroom" in usage or "training" in usage:
                    if not requirements.get("size"):
                        requirements["size"] = "approx 50 sqm (seats 20-30)"
                    if not requirements.get("viewing_distance"):
                        requirements["viewing_distance"] = "3-5m"

            # Trigger recommendation — usage is sufficient
            state["next_action"] = "trigger_solution"
            state["should_generate_solution"] = True
            logger.info(f"Key requirements collected (usage={requirements.get('usage')}), triggering product recommendation")
        else:
            # No usage yet — ask for it
            response_text = "What kind of scenario will this be used in? Meeting room, classroom, exhibition, advertising, etc.?"
            state["next_action"] = "ask"
            state["response"] = _strip_markdown(response_text)

    # Objection / Industry
    elif intent in ["objection", "industry"]:
        should_trigger = state.get("should_generate_solution", False)
        requirements = state.get("requirements", {})
        if should_trigger or requirements.get("usage"):
            if not state.get("response"):
                state["response"] = "Sure, let me find the right products for you..."
            state["next_action"] = "trigger_solution"
            logger.info("Industry intent with sufficient requirements")
        else:
            sales_search = state.get("sales_search")
            if sales_search:
                query = state["current_message"]
                docs = sales_search.similarity_search(query, k=2)
                reference_content = "\n\n".join([d.page_content for d in docs[:1]])

                prompt = SystemMessage(content=f"""You are a sales advisor for LED and LCD display products, speaking with a customer face-to-face.

Reference talking points (for reference only):
{reference_content}

Customer says: {query}

Requirements:
1. Conversational and natural, like chatting with a friend
2. Short, 1-2 sentences
3. Plain text only, no markdown

Reply directly:""")

                response = get_llm(temperature=0.3).invoke([prompt])
                response_text = response.content.strip()
            else:
                response_text = "Got it, let me learn more about your needs."
            state["next_action"] = "ask"

    # Product question
    elif intent == "product_question":
        # Check if requirements are sufficient to trigger solution
        if state.get("should_generate_solution", False):
            response_text = "Sure, let me find the right products for you..."
            state["next_action"] = "trigger_solution"
            logger.info("product_question with sufficient requirements, triggering solution")
        else:
            response_text = "Sure."
            state["next_action"] = "product_question"

    # Others
    elif intent == "others":
        # Check if requirements are sufficient to trigger solution
        if state.get("should_generate_solution", False):
            response_text = "Sure, let me find the right products for you..."
            state["next_action"] = "trigger_solution"
            logger.info("others with sufficient requirements, triggering solution")
        else:
            response_text = "Sure."
            state["next_action"] = "others"

    # Closing
    elif intent == "closing":
        response_text = "Alright, I'll put together a detailed proposal and quote for you as soon as possible. Feel free to ask if you have any other questions."
        state["next_action"] = "end"

    else:
        response_text = "What display scenario are you looking into?"
        state["next_action"] = "ask"
    
    # Only set response if it hasn't been set by router
    if not state.get("response"):
        state["response"] = _strip_markdown(response_text)
    
    logger.info(f"Generated response for intent={intent}, next_action={state['next_action']}")
    return state
