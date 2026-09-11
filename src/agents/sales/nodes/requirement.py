"""requirement_mining node - progressive requirement mining."""
import logging
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import SalesState
from ....config import config
from ....utils.ifp_intent import has_ifp_intent, user_messages_text

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
    # 先尝试阿拉伯数字
    digit_match = re.search(r'(\d+)', text)
    if digit_match:
        return int(digit_match.group(1))
    # 再尝试中文数字
    cn_match = re.search(r'[一二两三四五六七八九十百千]+', text)
    if cn_match:
        return _chinese_to_number(cn_match.group())
    return 0


REQUIRED_KEYS = ["usage"]  # 最少只需要知道使用场景
OPTIONAL_KEYS = ["location_type", "viewing_distance", "size", "brightness", "resolution"]

ADDITIONAL_CONTEXT_KEYS = ["pixel_pitch", "rental", "quick_install", "indoor", "outdoor"]

# 场景分类 - 用于检测上下文是否发生重大变化
SCENE_CATEGORIES = {
    "meeting": ["会议室", "会议", "教室", "培训", "教学", "学校", "课堂"],
    "church": ["教堂", "礼拜", "宗教", "礼拜堂"],
    "stage": ["舞台", "演唱会", "演出", "表演", "剧场"],
    "outdoor": ["室外", "户外", "露天", "外墙", "广场", "体育场"],
}

# 上下文相关字段 - 场景变化时应清除
CONTEXT_DEPENDENT_FIELDS = {
    "meeting": ["display_type"],  # 会议室可能需要 IFP
    "church": ["display_type", "size"],  # 教堂不需要手写屏，尺寸需要重新确定
    "stage": ["display_type", "size"],
    "outdoor": ["brightness"],
}


def _get_scene_category(usage: str) -> str:
    """根据 usage 返回场景分类"""
    if not usage:
        return "unknown"
    for category, keywords in SCENE_CATEGORIES.items():
        if any(kw in usage for kw in keywords):
            return category
    return "unknown"


def _should_clear_field_on_scene_change(old_category: str, new_category: str, field: str) -> bool:
    """判断场景变化时是否应清除某字段"""
    if old_category == new_category:
        return False
    # 场景类别发生变化
    if new_category in ["church", "stage", "outdoor"] and field == "display_type":
        return True  # 非会议室场景不需要 IFP
    if old_category == "meeting" and new_category != "meeting" and field == "size":
        return True  # 离开会议室场景时，之前的面积可能不适用
    return False


def _rule_based_inference(message: str, requirements: dict) -> dict:
    """基于规则从消息中推断需求，作为LLM提取的补充。"""
    text = message
    inferred = dict(requirements)
    
    # 推断 location_type / indoor / outdoor
    if not requirements.get("location_type"):
        indoor_keywords = ["室内", "户内", "会议室", "教室", "办公室", "展厅", "商场", "店铺", "监控室", "指挥中心"]
        outdoor_keywords = ["室外", "户外", "露天", "门口", "外墙", "广场", "体育场", "赛场"]
        
        for kw in indoor_keywords:
            if kw in text:
                inferred["location_type"] = "室内"
                inferred["indoor"] = True
                break
        for kw in outdoor_keywords:
            if kw in text:
                inferred["location_type"] = "室外"
                inferred["outdoor"] = True
                break
    
    # 推断 size / viewing_distance
    if not requirements.get("size"):
        # 匹配面积：X平米、X平方米、X平方
        area_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:平米|平方米|平方)', text)
        if area_match:
            inferred["size"] = f"{area_match.group(1)}平米"
    
    if not requirements.get("viewing_distance"):
        # 匹配具体尺寸：3米x4米、3x4米
        size_match = re.search(r'(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*米', text)
        if size_match:
            inferred["size"] = f"{size_match.group(1)}米x{size_match.group(2)}米"
        
        # 匹配距离：大约3米、大概5米、3-5米
        dist_match = re.search(r'(?:大约|大概|约|差不多)?\s*(\d+(?:\.\d+)?)(?:\s*[-~]\s*(\d+(?:\.\d+)?))?\s*米', text)
        if dist_match:
            if dist_match.group(2):
                inferred["viewing_distance"] = f"{dist_match.group(1)}-{dist_match.group(2)}米"
            else:
                inferred["viewing_distance"] = f"{dist_match.group(1)}米"
    
    # 推断 display_type（手写/触摸/交互 → IFP）
    if not requirements.get("display_type"):
        ifp_keywords = ["手写", "触摸", "触控", "交互", "白板", "会议一体机", "IFP", "interactive"]
        lcd_keywords = ["广告", "标牌", "信息发布", "电视墙"]
        led_keywords = ["舞台", "演唱会", "体育", "幕墙", "租赁"]
        
        if any(kw in text for kw in ifp_keywords):
            inferred["display_type"] = "IFP"
        elif any(kw in text for kw in led_keywords):
            inferred["display_type"] = "LED"
    
    # 从容纳人数推断 size（会议室/教室：每人约2-3平米）
    if not requirements.get("size") and not requirements.get("viewing_distance"):
        # 匹配"能容纳X人"、"X个人的"、"大概能容纳X人"等
        patterns = [
            r'(?:大约|大概|约|差不多)?\s*能容纳\s*([零一二两三四五六七八九十百千\d]+)\s*个人?',
            r'([零一二两三四五六七八九十百千\d]+)\s*个人?\s*(?:的|左右|左右)',
            r'能容纳\s*([零一二两三四五六七八九十百千\d]+)\s*个人?',
        ]
        people = 0  # 初始化默认值
        for pattern in patterns:
            occupancy_match = re.search(pattern, text)
            if occupancy_match:
                people = _parse_number(occupancy_match.group(1))
                if people > 0:
                    break
        
        if people > 0:
            # 会议室/教室按每人2.5平米估算，加上走道余量
            area = people * 2.5 + 5  # 额外5平米作为走道和空间余量
            inferred["size"] = f"约{area:.0f}平米"
            # 观看距离：10人以内的会议室，距离约2-4米
            if people <= 10:
                inferred["viewing_distance"] = "2-4米"
            elif people <= 20:
                inferred["viewing_distance"] = "3-5米"
            else:
                inferred["viewing_distance"] = "4-6米"
    
    return inferred


def _missing_requirements(requirements: dict) -> list:
    """Return a priority-ordered list of unmet requirement keys."""
    missing = [k for k in REQUIRED_KEYS if not requirements.get(k)]
    if len(missing) >= 2:
        return missing
    missing += [k for k in OPTIONAL_KEYS if not requirements.get(k)]
    return missing


def should_trigger_solution(requirements: dict) -> bool:
    """Trigger Solution Agent when all REQUIRED_KEYS are filled."""
    has_all_required = all(requirements.get(k) for k in REQUIRED_KEYS)
    return has_all_required


def requirement_mining(state: SalesState) -> SalesState:
    """Mine requirements from the conversation and decide next action."""
    llm = ChatOpenAI(
        model=config.MODEL_NAME,
        temperature=0,
        api_key=config.DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )
    
    # Build conversation context (last 3 messages)
    conversation = "\n".join([
        f"{getattr(msg, 'type', 'unknown')}: {getattr(msg, 'content', str(msg))}" 
        for msg in state["messages"][-3:]
    ])
    
    prompt = SystemMessage(content="""从对话中提取 LED 和 LCD 显示产品的需求信息，返回JSON格式：
{
  "location_type": "室内" | "室外" | null,
  "usage": "用户的原话场景" | null,
  "viewing_distance": "可视距离描述，如'3米'、'10-20米'、'远距离'" | null,
  "size": "尺寸描述" | null,
  "brightness": "亮度要求" | null,
  "resolution": "分辨率要求" | null,
  "display_type": "LED" | "LCD" | null,
  "additional_requirements": ["用户提到的其他特殊需求，如'需要租赁'、'要快装快拆'、'高刷新率'、'无缝拼接'等"]
}

**核心原则：只提取用户明确说过的字段，不要猜测或推断。**

**usage 提取规则**：直接用用户说的场景原词，不要做任何映射转换。例如：
- 用户说"演唱会" → usage="演唱会"
- 用户说"护理用" → usage="护理"
- 用户说"护理场景" → usage="护理"

**location_type 提取规则**：
- 用户说"室内"、"户内"、"室内使用" → "室内"
- 用户说"室外"、"户外"、"外面"、"露天"、"户外使用"、"室外使用" → "室外"
- 用户说"会议室用"、"办公室" → "室内"

**size 提取规则**：
- 用户说"3平米"、"3平方米"、"5平方"、"10平方"等面积 → "X平米"（保留原数字）
- 用户说"3米x4米"、"2x3米"等具体尺寸 → "X米xY米"
- 用户说"大屏"、"大尺寸" → "大尺寸"

如果用户没有提到某个字段，该字段必须为 null，绝对不要凭空生成值。

**额外需求 (additional_requirements) 提取规则：**
1. 收集用户明确提到的所有特殊需求，不在上述必填/选填字段中的
2. 常见额外需求示例：租赁、快装快拆、防水、高刷新率、无缝、低功耗、移动安装、弧形、异形等
3. 如果用户没有提到任何额外需求，返回空列表 []

只返回JSON，不要其他内容。已知信息保留，新信息补充。""")
    
    current_msg = HumanMessage(content=f"已有需求：{state['requirements']}\n\n当前对话：\n{conversation}")
    response = llm.invoke([prompt, current_msg])
    
    # 保存合并前的旧状态，用于上下文变化检测
    pre_merge_requirements = dict(state["requirements"])
    
    # Parse extracted requirements
    import json
    try:
        extracted = json.loads(response.content.strip())
        # Merge with existing requirements (new values override)
        for k, v in extracted.items():
            if k == "additional_requirements":
                if not isinstance(state.get("additional_requirements"), list):
                    state["additional_requirements"] = []
                if isinstance(v, list):
                    for item in v:
                        if item and item not in state["additional_requirements"]:
                            state["additional_requirements"].append(item)
            elif v:
                state["requirements"][k] = v
    except json.JSONDecodeError:
        logger.warning("Failed to parse requirements JSON")
    
    # 规则推断补充：即使LLM提取失败，也能从消息中推断基本需求
    current_msg_text = state.get("current_message", "")
    state["requirements"] = _rule_based_inference(current_msg_text, state["requirements"])

    # 补充推断 usage 字段（_rule_based_inference 不推断 usage）
    if not state["requirements"].get("usage"):
        _USAGE_KEYWORDS = [
            ("会议室", "会议室"), ("会议", "会议室"), ("教室", "教室"), ("培训", "教室"),
            ("商场", "商场"), ("店铺", "商业零售"), ("零售", "商业零售"),
            ("展厅", "展厅"), ("展览", "展厅"), ("医院", "医疗"),
            ("监控", "监控指挥"), ("指挥", "监控指挥"),
            ("舞台", "舞台演出"), ("演出", "舞台演出"), ("演唱会", "演唱会"),
            ("体育", "体育场馆"), ("赛场", "体育场馆"),
            ("广告", "广告传媒"), ("幕墙", "建筑幕墙"), ("租赁", "租赁活动"),
        ]
        for kw, usage_val in _USAGE_KEYWORDS:
            if kw in current_msg_text:
                state["requirements"]["usage"] = usage_val
                logger.info(f"Inferred usage='{usage_val}' from keyword '{kw}'")
                break

    # 上下文变化检测：当 usage 从一种场景类别变为另一种时，清除上下文相关字段
    old_usage = pre_merge_requirements.get("usage", "")
    old_category = _get_scene_category(old_usage)
    new_usage = state["requirements"].get("usage", "")
    new_category = _get_scene_category(new_usage)
    
    if old_category != new_category and old_category != "unknown":
        logger.info(f"Context change detected: {old_category} → {new_category}, checking fields to clear")
        fields_to_clear = []
        for old_scene, fields in CONTEXT_DEPENDENT_FIELDS.items():
            if _should_clear_field_on_scene_change(old_category, new_category, fields[0] if fields else ""):
                fields_to_clear.extend(fields)
        # 去重
        fields_to_clear = list(set(fields_to_clear))
        for field in fields_to_clear:
            if field in state["requirements"]:
                logger.info(f"Clearing stale field '{field}' from previous context ({old_category})")
                state["requirements"].pop(field)
        # 重新检查触发条件
        state["should_generate_solution"] = should_trigger_solution(state["requirements"])

    # IFP safety: only keep IFP in meeting room context
    # Context-aware: check current usage, not just message keywords
    customer_text = user_messages_text(state.get("messages", []))
    display_type = str(state.get("requirements", {}).get("display_type", "")).strip().upper()
    current_usage = state.get("requirements", {}).get("usage", "")
    current_category = _get_scene_category(current_usage)
    
    if display_type == "IFP":
        # Only keep IFP if user explicitly wants interaction in meeting room context
        ifp_trigger_keywords = ["手写", "书写", "触摸", "触控", "交互", "白板", "批注", "会议一体机"]
        
        has_ifp_word = any(kw in customer_text for kw in ifp_trigger_keywords)
        
        # Keep IFP only in meeting room context with explicit interaction need
        if current_category == "meeting" and has_ifp_word:
            logger.info(f"IFP kept: meeting room context with interaction need")
        else:
            # Not in meeting room context or no explicit interaction need - remove
            logger.info(f"Removed IFP: current_category={current_category}, has_ifp_word={has_ifp_word}")
            state["requirements"].pop("display_type", None)

    # Check trigger condition
    state["should_generate_solution"] = should_trigger_solution(state["requirements"])

    # Compute missing requirements
    missing = _missing_requirements(state["requirements"])
    state["required_met"] = len(missing) == 0 or len([k for k in REQUIRED_KEYS if state["requirements"].get(k)]) == len(REQUIRED_KEYS)
    state["required_missing"] = missing[:2]

    logger.info(f"requirement_mining: usage={state['requirements'].get('usage')}, should_generate_solution={state['should_generate_solution']}, requirements={state['requirements']}")

    # Intent-based routing override
    intent = state.get("intent", "")
    
    # For greeting: block solution UNLESS suppress_greeting is True (first contact just completed)
    # After first contact, if requirements are sufficient, allow solution even with greeting intent
    if intent == "greeting":
        if state.get("suppress_greeting") and state.get("should_generate_solution"):
            # First contact just completed and requirements are sufficient — allow solution
            logger.info("Greeting intent but suppress_greeting=True and requirements sufficient — allowing solution")
        else:
            state["should_generate_solution"] = False
            logger.info(f"Intent 'greeting' — forcing should_generate_solution=False (suppress_greeting={state.get('suppress_greeting')})")
    
    # Closing always blocks solution
    elif intent == "closing":
        state["should_generate_solution"] = False
        logger.info(f"Intent 'closing' — forcing should_generate_solution=False")
    
    # For product_question/others: preserve their routing UNLESS requirements are sufficient
    elif intent in ("product_question", "others"):
        if state["should_generate_solution"]:
            # 关键修复：覆盖路由，不走 product_question/others 的流程
            state["next_action"] = "router"  # 强制走 router 触发推荐
            logger.info("Overriding %s routing — requirements sufficient, will trigger solution", intent)
        else:
            logger.info("Preserving %s next_action through requirement_mining", intent)
            state["should_generate_solution"] = False
    else:
        logger.info(f"Requirements: {state['requirements']}, should_trigger: {state['should_generate_solution']}, required_met: {state['required_met']}")

    return state
