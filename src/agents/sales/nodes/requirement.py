"""requirement_mining node - progressive requirement mining."""
import logging
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import SalesState
from ....config import config
from ....models.legacy_adapter import rebuild_legacy_view
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


ADDITIONAL_CONTEXT_KEYS = ["pixel_pitch", "rental", "quick_install", "indoor", "outdoor"]

# 场景切换时需要清理的字段：legacy 字段名 → RequirementProfile 字段名
_LEGACY_TO_PROFILE_FIELDS: dict = {
    "display_type": ("display_type",),
    "size": ("target_width_m", "target_height_m", "screen_size_hint_mm"),
    "brightness": ("brightness_min_nit", "brightness_max_nit"),
}

# 场景分类 - 用于检测上下文是否发生重大变化
# 注意：M1 之后 requirements 是 RequirementProfile 的投影，usage 可能是规范化 token
# （conference / church / advertising …），所以这里两套写法都要认。
SCENE_CATEGORIES = {
    "meeting": [
        "会议室", "会议", "教室", "培训", "教学", "学校", "课堂",
        "conference", "classroom", "office", "control_room", "hotel",
        "restaurant", "airport", "exhibition", "showroom", "museum", "hall",
        "retail", "hospital", "bank",
    ],
    "church": ["教堂", "礼拜", "宗教", "礼拜堂", "church"],
    "stage": ["舞台", "演唱会", "演出", "表演", "剧场", "stage", "concert"],
    "outdoor": ["室外", "户外", "露天", "外墙", "广场", "体育场", "advertising", "stadium"],
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
    """基于规则从消息中推断需求，作为LLM提取的补充。

    重要（v2.0 Ready Gate）：本函数分两类输出 ——
      - **客户明说的**（"5米"、"20平米"、"3米x4米"）：写入 requirements，来源视为 confirmed
      - **规则估算的**（从容纳人数反推面积/视距）：写入 requirements 的同时记入
        ``_inferred_slots``，来源标记为 inferred —— **不得用来打开推荐 Gate**，
        否则会出现"只知道室内外+场景就直接推荐"的错误行为。
    """
    text = message
    inferred = dict(requirements)
    
    # 推断 location_type / indoor / outdoor —— 只认客户明确说出的室内外关键词，
    # 不从"会议室/展厅/商场"等场景词推断环境（环境必须由客户明说，见计划 v1.0 阶段 5.1）
    if not requirements.get("location_type"):
        indoor_keywords = ["室内", "户内", "indoor", "indoors"]
        outdoor_keywords = ["室外", "户外", "露天", "outdoor", "outdoors"]
        
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
    
    return inferred


# ── M1：Profile ↔ 旧 requirements 的适配层（实现在 src/models/legacy_adapter.py）──
# 规则：旧字段只允许由 Profile 生成，绝不允许反过来修改 Profile。


# 客户在"问问题"的标志（问号 / 疑问词 / 业务咨询词）
_INTERROGATIVE_RE = re.compile(
    r"[?？]|"
    r"\b(?:what|how|which|why|when|where|who|does|do you|do u|do ya|have you got|"
    r"can you|could you|would you|is there|are there|tell me|"
    r"price|cost|spec|specs|specification|warranty|quote|quotation)\b|"
    r"多少|什么|怎么|哪|几|价格|报价|参数|规格|保修",
    re.IGNORECASE,
)

# 客户"报了一个需求值"的槽位（有值 = 他在回答需求，不是在提问）
_ANSWER_VALUE_SLOTS = (
    "environment",
    "installation",
    "viewing_distance_m",
    "distance",
    "target_width_mm",
    "target_height_mm",
    "screen_size_hint_mm",
)

# 场景关键词（用于判断"本轮这句话"有没有在说使用场景）
_USAGE_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("会议室", "会议室"), ("会议", "会议室"), ("教室", "教室"), ("培训", "教室"),
    ("商场", "商场"), ("店铺", "商业零售"), ("零售", "商业零售"),
    ("展厅", "展厅"), ("展览", "展厅"), ("医院", "医疗"),
    ("监控", "监控指挥"), ("指挥", "监控指挥"),
    ("舞台", "舞台演出"), ("演出", "舞台演出"), ("演唱会", "演唱会"),
    ("体育", "体育场馆"), ("赛场", "体育场馆"),
    ("广告", "广告传媒"), ("幕墙", "建筑幕墙"), ("租赁", "租赁活动"),
)

# 客户明确要求"再推荐 / 报价 / 下单"（这种即便已经推荐过也应重新给方案）
_EXPLICIT_RECO_REQUEST_RE = re.compile(
    r"推荐|帮我选|再选|换一款|换个型号|其他型号|别的型号|还有别的|其他方案|报价|报价单|价格表|下单|采购|"
    r"\b(?:recommend|suggest|quote|quotation|proposal|price list|another model|other options?|"
    r"alternative|come back with|proceed)\b",
    re.IGNORECASE,
)

# 判断"是否已经推荐过"时看的需求事实字段
# 客户"这一轮在说需求"的槽位（用于区分"回答问题"还是"陈述需求"）
_TURN_REQUIREMENT_KEYS = (
    "environment",
    "installation",
    "viewing_distance_m",
    "distance",
    "target_width_mm",
    "target_height_mm",
    "screen_size_hint_mm",
    "size",
    "pixel_pitch_mm",
    "purpose",
    "content_type",
    "price_preference",
    "budget_level",
)


_FACT_FIELDS = (
    "display_type",
    "environment",
    "purpose",
    "installation",
    "viewing_distance_m",
    "target_width_m",
    "target_height_m",
    "pixel_pitch_mm",
    "brightness_min_nit",
    "brightness_max_nit",
    "budget_level",
)


def _is_question_message(message: str) -> bool:
    """客户是不是在提问（与 Solution 侧共用同一份规则，见 query_understanding）。"""
    from ....rag.query_understanding import looks_like_question

    return looks_like_question(message)


_ACK_PROMPT = """你是 LED 显示屏产品的销售，正在微信上和客户聊天。
客户刚说的这句话**与产品需求无关**（寒暄、闲聊、感叹、题外话、随口一提等）。
请只回应这句话本身，像真人销售一样自然接一句（最多两句，口语化）。

硬性规则：
1. 只"接住"这句话：表示听到了 / 表示理解 / 顺着他的话头轻轻应一句。
2. **绝对不要**回答任何知识性、技术性问题；不要解释任何概念；不要给参数、型号、价格、方案或建议；
   不要输出任何我们没有依据的信息。
3. **不要提问**（系统会另外接一个需求问题，避免一句话里出现两个问题）。
4. 不要复述客户整句话；不要客套话堆砌；不要说"作为AI / 作为助手"。
5. 每次换一种说法，不要固定句式，可以让语气自然一点。
6. 语言要求：{language_rule}
7. 直接输出这一句回应，不要 JSON、不要引号、不要解释。"""

# 客户这句话是不是"我们该正面回答的"（产品/规格/价格/交期/公司信息）
_PRODUCT_TERMS_RE = re.compile(
    r"\bTW\s*\d{2}\b|\bP\s*\d(?:\.\d+)?\b|\b(?:led|lcd|ifp|cob|gob|hdr|ip6[56])\b|"
    r"pixel\s*pitch|brightness|refresh|resolution|\bnit\b|specs?\b|models?\b|series\b|"
    r"warrant(?:y|ies)|guarantee|certificat(?:e|ion)|"
    r"显示屏|屏幕|大屏|单色屏|点间距|亮度|刷新率|分辨率|型号|规格|参数|防水|像素|屏体|屏",
    re.IGNORECASE,
)


# ── 中文业务问题补充词表 ────────────────────────────────────────────────
# 【修复】客户用中文问"能不能定制 / 交付日期多久 / 有质保吗 / 有代理商吗"这类
# 业务问题时，上面的词表经常匹配不上，于是被当成"与需求无关的话"：
# 只回一句 "Got it, …" 接话，客户真正的问题**没有任何回答**
# （实测日志："我能定制产品吗"、"你们的交付日期是多久"）。
_ZH_BUSINESS_TERMS_RE = re.compile(
    r"定制|订制|定做|订做|改尺寸|加工|OEM|ODM|"
    r"质保|保修|售后|维修|认证|证书|检测报告|"
    r"代理|经销商|分销|工厂|厂家|"
    r"付款|定金|订金|预付|发票|运费|发货|到货|交付|交货|工期|生产周期|"
    r"亮度|分辨率|刷新|点间距|安装方式|"
    r"能(?:不)?能做|可以(?:不)?可以|支持吗|有.{0,4}吗",
    re.IGNORECASE,
)


def _is_product_or_business_question(message: str) -> bool:
    """客户这句话是不是"我们该正面回答的问题"。

    True = 产品/规格/价格/交期/公司信息（走正常回答路径）；
    False = 与业务无关的闲聊、题外话（哪怕是问句，也只"接住"再继续问需求）。
    """
    text = str(message or "")
    if _PRODUCT_TERMS_RE.search(text) or _ZH_BUSINESS_TERMS_RE.search(text):
        return True
    try:
        from ....rag.company_info import is_company_question
        from ....rag.delivery_info import is_delivery_question
        from ....rag.reply_composer import is_price_question
    except Exception:  # pragma: no cover - 防御式
        return False
    try:
        return bool(
            is_company_question(text) or is_delivery_question(text) or is_price_question(text)
        )
    except Exception:  # pragma: no cover - 防御式
        return False


def _ack_temperature() -> float:
    from ....config import config as _config

    try:
        return float(getattr(_config, "ACK_TEMPERATURE", 0.7))
    except (TypeError, ValueError):  # pragma: no cover - 防御式
        return 0.7


# ── 接话（ack）写法补充规则 ──────────────────────────────────────────────
# 客户口径（2026-09-18）：客户每说完一件事，AI 只会 "Got it / Understood"，
# 太死板。要求：顺着客户刚说的内容说、每次换一种说法、不许用烂开头，
# 但仍**只准一句**，不许提问（问需求由系统另起一句，并且两句话要自然连成一段）。
_ACK_STYLE_RULES = """

【接话补充规则 · 客户口径】
1. **禁止**用这些已经用烂的开头：Got it / Okay / OK / Understood / Sure / Thanks for that /
   好的 / 收到 / 了解 / 明白。第一句就直接顺着客户的话说。
2. **顺着客户这句话的具体内容说**：他提到场地就说场地、提到距离就说距离、
   提到用途 / 担忧 / 问题就接那个点；不要只回一句空泛的"收到 / 明白了"。
3. **每次换一种说法**：不能和下面这些最近已经发出去的接话雷同（开头、句式都要换）。
最近已发出的接话：
{recent}
"""


def _recent_ack_hints(state: SalesState, limit: int = 3) -> str:
    """最近几轮已经发给客户的话（供 ack 参考，避免重复）。"""
    lines: list[str] = []
    for item in reversed(state.get("messages") or []):
        if isinstance(item, dict):
            role = str(item.get("role") or item.get("type") or "")
            content = str(item.get("content") or "")
        else:  # pragma: no cover - LangChain 消息对象
            role = str(getattr(item, "type", "") or "")
            content = str(getattr(item, "content", "") or "")
        if role not in ("assistant", "ai"):
            continue
        text = " ".join(content.split())
        if not text:
            continue
        lines.append(f"- {text[:160]}")
        if len(lines) >= limit:
            break
    return "\n".join(reversed(lines)) or "（暂无）"


def _generate_offtopic_ack(
    message: str,
    *,
    requirement: str = "",
    language: str = "en",
) -> str:
    """客户说了与需求无关的话时，用较高温度生成一句自然的"接住"话术。

    与需求抽取分开调用：抽取仍是 temperature=0（稳定、不编参数），
    这里用 ``config.ACK_TEMPERATURE``（默认 0.7）让措辞发散、不重复；
    并且严格禁止回答知识性问题（见 _ACK_PROMPT）。
    """
    try:
        from ....rag.query_understanding import response_language_rule
        from ....rag.reply_composer import _clean_llm_ack
    except Exception:  # pragma: no cover - 防御式
        return ""
    try:
        llm = ChatOpenAI(
            model=config.MODEL_NAME,
            temperature=_ack_temperature(),
            api_key=config.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
        )
        system = SystemMessage(
            content=_ACK_PROMPT.format(language_rule=response_language_rule(language))
        )
        human = HumanMessage(
            content=(
                f"客户这句话：{message}\n"
                f"（已知需求，仅供判断语气，不要复述）：{requirement or '{}'}"
            )
        )
        response = llm.invoke([system, human])
        text = response.content if hasattr(response, "content") else str(response)
        cleaned = _clean_llm_ack(text, language)
        # 模型没按要求给口语回应（返回了 JSON / 代码块之类）→ 宁可不加这一句，
        # 也不能把内部结构或无关内容发出去。
        if not cleaned or "{" in cleaned or "}" in cleaned or cleaned.startswith(("[", "(")):
            logger.info("Off-topic ack dropped (not a natural sentence): %r", cleaned[:60])
            return ""
        return cleaned
    except Exception as exc:
        logger.warning("Off-topic ack generation failed: %s", exc)
        return ""


def _snapshot_facts(profile) -> dict:
    return {field: getattr(profile, field, None) for field in _FACT_FIELDS}


def _facts_added(before: dict, profile) -> bool:
    """这一轮是否**新增/更新**了需求事实（用于决定要不要重新推荐）。"""
    for field in _FACT_FIELDS:
        new_value = getattr(profile, field, None)
        if new_value in (None, "", [], {}):
            continue
        if new_value != before.get(field):
            return True
    return False


def _message_role_content(msg) -> tuple[str, str]:
    if isinstance(msg, dict):
        role = str(msg.get("role") or msg.get("type") or "")
        content = str(msg.get("content") or "")
        return role, content
    return str(getattr(msg, "type", "") or getattr(msg, "role", "")), str(
        getattr(msg, "content", "") or ""
    )


def _looks_like_requirement_answer(message: str, slots: dict) -> bool:
    """客户这句话是不是在"报需求"，而不是在"问问题"。

    实测 bug：客户回了一句 "129,2cm"，被 classify 判成 others →
    直接走了 Solution 的自由问答，把 indoor/outdoor 型号一股脑倒了出来。
    报需求的话必须留在需求采集流程里。
    """
    text = str(message or "").strip()
    if not text or _INTERROGATIVE_RE.search(text):
        return False
    return any(slots.get(key) for key in _ANSWER_VALUE_SLOTS)


def requirement_mining(state: SalesState) -> SalesState:
    """Mine requirements from the conversation and decide next action."""
    # 本轮"回应客户这句话"的口语回应（由下面的 LLM 抽取一并生成；失败则为空）
    state["acknowledgement"] = ""
    llm = ChatOpenAI(
        model=config.MODEL_NAME,
        temperature=0,
        api_key=config.DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )
    
    # Build conversation context (last 3 messages)
    # 需求重置（客户换产品 / 换项目）时只看本轮这句话：
    # 否则上一轮的"教堂 / 室内"会被 LLM 当成当前需求再提取回来，等于没清空。
    if state.get("requirements_reset"):
        conversation = f"user: {state.get('current_message', '')}"
    else:
        parts = []
        for msg in state["messages"][-6:]:
            role, content = _message_role_content(msg)
            if not content:
                continue
            label = "user" if role in ("user", "human") else "assistant"
            parts.append(f"{label}: {content}")
        conversation = "\n".join(parts[-6:])
    
    # 回应语言必须跟系统策略一致：默认策略是"永远英文"，但下面的提示词是中文写的，
    # 模型很容易顺手用中文回一句（实测出现过
    # "Sello 你好，很高兴认识你。Will it be an indoor or outdoor setup?"）。
    try:
        from ....config import config as _config

        _policy = str(getattr(_config, "RESPONSE_LANGUAGE_POLICY", "en") or "en").lower()
    except Exception:  # pragma: no cover - 防御式
        _policy = "en"
    _ack_language_rule = (
        "ack 必须使用客户所用的语言。"
        if _policy == "auto"
        else "ack 必须**只用英文**（即使客户用中文或其它语言，也必须用英文回应），绝对不要中英混排。"
    )

    prompt = SystemMessage(content="""你是需求采集助手，同时是一位正在跟客户面对面沟通的销售顾问。
请从对话中提取信息，**并给出一句自然的口语回应**，返回 JSON：
{
  "usage": "用户原话描述的使用场景，如'会议室'、'演唱会'、'展厅'；没有则 null",
  "purpose": "使用场景的**标准 token**（只能从下面这张表里选，不能自造）：retail/advertising/conference/classroom/stadium/concert/stage/wedding/church/museum/showroom/airport/bank/hotel/restaurant/office/hospital/exhibition/hall/rental/control_room/other；没有则 null",
  "environment": "indoor / outdoor / semi_outdoor —— **只有客户这句话里明确说了室内外才填**（比如 open-air、on the facade、露天、室内），否则必须 null",
  "environment_evidence": "填 environment 时，必须给出客户原话里的**原样片段**（例如 'open-air advertising'）；没填 environment 时给空字符串",
  "environment_implied_by_scene": "如果**客户原话里的场景本身**就决定了室内外（例如 church / classroom / football stadium / roadside billboard / shopping mall / 医院大厅），填 indoor 或 outdoor；只要室内外都可能（stage / concert / rental / wedding / 泛泛的 exhibition / 说不清），一律 null",
  "environment_implied_evidence": "填 environment_implied_by_scene 时，给出客户原话里的**原样片段**（例如 'for our church'）；没填时给空字符串",
  "installation": "fixed / rental —— **只有客户明确说了安装方式才填**（permanent、fixed、rental、temporary、租赁），否则 null",
  "installation_evidence": "填 installation 时给出客户原话里的原样片段（例如 'rental'）；没填时给空字符串",
  "additional_requirements": ["用户明确提出的、无法归入场景的特殊要求，如'租赁'、'防水'、'高刷新率'"],
  "ack": "一句话自然回应客户这句话本身；没有可回应的内容时返回空字符串"
}

ack 的写法（很重要，销售不能只会追问）：
1. **只回应客户这句话本身**：自我介绍 → 问候并称呼对方名字；说明单位 / 项目 / 身份
   → 表示理解；客户提问 → 简短回应（不清楚的就说明会帮忙确认）；客户要报价 / 规格 /
   资料 → 表示"确认几个关键点后马上准备"。
2. 像真人销售顾问说话，一句话（最多两句），不要客套话堆砌，不要复述客户整句话。
3. **禁止**编造任何技术参数、型号、价格、交期；不要说"根据资料/数据库/检索"。
4. **禁止**在 ack 里提问（系统会另外接一个需求问题），也不要给推荐结论。
5. 没有可回应内容（比如客户只回了一个词的数字）时，ack 返回 ""。
6. 语言要求：""" + _ack_language_rule + """

提取规则：
1. **purpose 是语义归一化**：客户用各种说法（shopping center / commercial complex /
   football venue / corporate boardroom / 商场 / 教堂…）你都要映射到标准 token；
   但**不要**因此推断客户没说的其它参数。
2. **environment / installation 只能来自客户原话**，并且必须给出原样证据片段；
   没有明确说就返回 null —— 绝对不要根据场景替客户猜（例如"会议室"不能推断 indoor，
   除非客户自己说了 indoor）。
3. **绝对不要**补全这些工程参数：观看距离、屏幕尺寸、亮度、分辨率、点间距 ——
   它们由系统的确定性解析器从客户原话提取。
4. usage 只写用户实际描述的场景原词，不要映射、不要扩展（映射由 purpose 负责）。
5. 没有对应信息就返回 null 或 []，禁止编造。
6. environment_implied_by_scene 是"**场景本身就决定室内外**"的判断：必须基于客户原话里的
   场景词，不能凭空猜；舞台 / 演唱会 / 租赁 / 婚礼 / 展会这类室内外都可能的一律 null。
6. 只返回 JSON，不要任何解释。""")
    
    current_msg = HumanMessage(content=f"已有需求：{state['requirements']}\n\n当前对话：\n{conversation}")
    # 【客户口径】把"接话要有变化"的规则追加进提示词（含最近已发出的接话，
    # 避免每轮都回 "Got it / Understood"）。仍然只准一句、不许提问。
    prompt = SystemMessage(
        content=str(prompt.content) + _ACK_STYLE_RULES.format(recent=_recent_ack_hints(state))
    )
    response = llm.invoke([prompt, current_msg])
    
    # 保存合并前的旧状态，用于上下文变化检测
    pre_merge_requirements = dict(state["requirements"])
    
    # Parse extracted requirements
    import json
    semantic_payload: dict = {}
    extracted: dict = {}
    try:
        extracted = json.loads(response.content.strip())
        # 只信任 LLM 的 usage 与 additional_requirements。
        # 其余工程参数（viewing_distance/size/brightness/resolution/location_type/display_type）
        # 一律忽略 —— 由确定性解析器负责，防止 AI 根据场景自行补全参数导致过早推荐。
        for k, v in extracted.items():
            if k == "additional_requirements":
                if not isinstance(state.get("additional_requirements"), list):
                    state["additional_requirements"] = []
                if isinstance(v, list):
                    for item in v:
                        if item and item not in state["additional_requirements"]:
                            state["additional_requirements"].append(item)
            elif k in ("ack", "acknowledgement"):
                # 由 LLM 生成的"接住客户这句话"的口语回应。
                # 只用于回复措辞，不参与任何 Gate 判定，且做安全清洗。
                ack = " ".join(str(v or "").split())
                if ack.endswith(("?", "？")) or any(mark in ack for mark in ("?", "？")):
                    # 追问由系统另外拼接，避免一句话里出现两个问题
                    logger.info("Dropping LLM ack containing a question: %r", ack[:80])
                    ack = ""
                if len(ack) > 240:
                    ack = ack[:240].rstrip()
                state["acknowledgement"] = ack
            elif k == "usage" and v:
                state["requirements"][k] = v

        # ── Phase 3/13：把这次 LLM 调用的语义结果交给统一 Extractor ─────────
        # purpose（标准 token）+ 带原话证据的 environment / installation。
        # 这样同一轮只做一次 LLM 语义理解，Sales 与 Solution 共用结果。
        for key in ("purpose", "environment", "installation", "display_type"):
            if extracted.get(key) not in (None, "", []):
                semantic_payload[key] = extracted[key]
        for key in ("environment_evidence", "installation_evidence"):
            if extracted.get(key):
                semantic_payload[key] = extracted[key]
        # 场景本身就决定室内外（AI 判断，必须带客户原话证据）：
        # 客户口径 —— 明显的室内/室外场景不要再问客户"室内还是室外"
        for key in ("environment_implied_by_scene", "environment_implied_evidence"):
            if extracted.get(key):
                semantic_payload[key] = extracted[key]
    except json.JSONDecodeError:
        logger.warning("Failed to parse requirements JSON")
    
    # 规则推断补充：即使LLM提取失败，也能从消息中推断基本需求
    current_msg_text = state.get("current_message", "")
    state["requirements"] = _rule_based_inference(current_msg_text, state["requirements"])

    # ── 客户这句话与需求无关时：用较高温度单独生成"接住这句话"的口语回应 ──────
    # 需求抽取仍然用 temperature=0（稳定、不编参数）；这里只生成措辞，
    # 让"接话"不至于每次都是同一句死板的话（config.ACK_TEMPERATURE，默认 0.7）。
    # 硬性约束：只接住这句话，绝不回答任何知识性问题、不给参数/型号/价格/建议。
    try:
        from ....rag.query_understanding import extract_slots

        _this_turn_slots = {
            k: v
            for k, v in (extract_slots(current_msg_text) or {}).items()
            if not str(k).startswith("_")
        }
        # 客户这一轮是不是在"说需求"（哪怕是重复一遍）——用于区分
        # "回答问题"（走自由问答）和"陈述需求"（重新按需求推荐）。
        turn_states_requirement = bool(
            any(_this_turn_slots.get(key) for key in _TURN_REQUIREMENT_KEYS)
        ) or bool(semantic_payload) or bool((extracted or {}).get("usage"))
        _provides_requirement = bool(_this_turn_slots) or bool(semantic_payload) or bool(
            (extracted or {}).get("usage")
        )
        # 只处理"与业务无关的话/闲聊"（哪怕它是问句，例如 "do u like watching TV series?"）。
        # 客户如果问的是产品/规格/价格/交期/公司信息，交给正常路径正面回答，
        # 不能当成无关话只回一句"接住"，也不能反过来倒一堆型号。
        if not _provides_requirement and not _is_product_or_business_question(current_msg_text):
            from ....rag.query_understanding import detect_language

            _ack = _generate_offtopic_ack(
                current_msg_text,
                requirement=str(state.get("requirements") or ""),
                language=detect_language(current_msg_text),
            )
            if _ack:
                state["acknowledgement"] = _ack
                logger.info("Off-topic ack (temp=%.1f): %s", _ack_temperature(), _ack)
                # 客户只是说了句无关的话：这一轮就是"接住这句话 + 继续问需求"，
                # 不能绕到 Solution 的自由问答（否则会倒一堆型号和参数）。
                state["offtopic_turn"] = True
                state["intent"] = "need_query"
                logger.info("Off-topic turn → keep requirement mining (intent=need_query)")
    except Exception as exc:  # pragma: no cover - 防御式
        logger.warning("Off-topic ack step failed: %s", exc)

    # 补充推断 usage 字段（_rule_based_inference 不推断 usage）
    if not state["requirements"].get("usage"):
        for kw, usage_val in _USAGE_KEYWORDS:
            if kw in current_msg_text:
                state["requirements"]["usage"] = usage_val
                logger.info(f"Inferred usage='{usage_val}' from keyword '{kw}'")
                break

    # ── 关键修复：场景回答不能被 classify 误判成 others/product_question ──
    # 当客户正在回答"用在什么场景"（例如只说 church），这属于需求采集，
    # 必须继续走 need_query → Gate → 追问，而不是绕到 Solution 的自由问答检索。
    current_intent = str(state.get("intent") or "")
    if current_intent in ("others", "product_question"):
        from ....rag.query_understanding import extract_slots

        current_slots = extract_slots(current_msg_text)
        # 【关键】只看**本轮这句话**：不能拿"历史里已经收集到的 usage"当真 ——
        # 否则场景一旦确定，客户之后说的每一句话（包括"你们在肯尼亚有代理商吗？"）
        # 都会被当成"在报需求"→ 被判 need_query → Gate 已就绪 → 又推荐一遍，
        # 客户问什么都得不到回答（实测日志出现过）。
        #
        # 但"混合句"要照旧留在需求采集里：客户一句话里既说场景 / 要规格、又问问题
        # （"we are going for smart class room ... could you help me out about specs"），
        # 只要本轮真的带来了需求信息，就算 need_query。
        this_turn_requirement = bool(
            current_slots.get("purpose")
            or current_slots.get("environment")
            or current_slots.get("installation")
            or (semantic_payload or {}).get("purpose")
            or (semantic_payload or {}).get("environment")
            or (semantic_payload or {}).get("installation")
            or (extracted or {}).get("usage")
        ) or any(kw in current_msg_text for kw, _ in _USAGE_KEYWORDS)
        # 客户"报需求"（室内外 / 安装方式 / 视距 / 尺寸）而不是"问问题"时，
        # 必须留在需求采集流程：否则会绕到自由问答，在 Gate 没通过的情况下
        # 把一堆型号和参数倒给客户（实测出现过）。
        # 但"提问"（？/吗/怎么/有没有…）必须留给回答路径，不能被改成 need_query。
        if this_turn_requirement or (
            not _is_question_message(current_msg_text)
            and _looks_like_requirement_answer(current_msg_text, current_slots)
        ):
            logger.info(
                "Detected requirement answer %r under intent '%s' → reclassify as need_query",
                current_msg_text, current_intent,
            )
            state["intent"] = "need_query"

    # 注：上下文变化检测（场景切换清理）已经移到 Profile 上做（见下方 Extractor 之后），
    # 这里不再在 legacy 字典上重复一遍。

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

    # ── v2.0 Phase 4：Recommendation Ready Gate 门控推荐（唯一推荐闸门）──────
    # 【M5/M11】是否推荐**只**由下面的 Gate 写入 should_generate_solution：
    # 旧版这里还有一行 `should_trigger_solution(state["requirements"])`（只看 usage），
    # 一旦 Gate 抛异常就会留下 True，从异常分支绕过闸门去推荐 —— 已删除。
    # 旧逻辑只要拿到 usage 就立刻推荐；现在由 Gate 决定：
    #   未就绪 → 不触发推荐，一次只问一个高价值问题
    #   已就绪 → 触发推荐（即使旧 usage 规则尚未满足，例如英文场景表达）
    try:
        from ....models.requirement import RequirementProfile
        from ....rag.query_understanding import extract_slots
        from ....rag.readiness import check_recommendation_ready, first_missing_slot
        from ..question_planner import plan_next_question, should_ask_before_recommend

        # ── M1：RequirementProfile 是唯一主状态 ──────────────────────────────
        # 优先加载 runner 从 memory 取出来的 Profile（上一轮的完整状态），
        # 只有"老会话还没有 Profile"时才退回 from_legacy 构建一次。
        # 旧实现每轮都 from_legacy 重建 → Profile 与 legacy requirements 之间
        # 形成 Profile→legacy→Profile 的往返搬运（本次改造要消除的双向同步）。
        loaded_profile = state.get("requirement_profile")
        if isinstance(loaded_profile, RequirementProfile):
            profile = loaded_profile
        elif isinstance(loaded_profile, dict) and loaded_profile:
            try:
                profile = RequirementProfile.model_validate(loaded_profile)
            except Exception:  # pragma: no cover - 防御式
                profile = RequirementProfile.from_legacy(state["requirements"])
        else:
            profile = RequirementProfile.from_legacy(state["requirements"])

        # ── Phase 3：本轮需求理解统一走 RequirementExtractor ─────────────────
        # 规则解析（数字/尺寸/单位/明确关键词）+ 语义结果（复用本轮 Sales 的那一次
        # LLM 调用，semantic_override）+ purpose 标准化 + 环境/安装方式统一解析 +
        # 冲突检测，全部由 Extractor 负责，这里不再自己拼关键词。
        from ....core.requirement_extractor import get_requirement_extractor

        previous_purpose = profile.purpose
        # 本轮开始前的需求事实快照（用于判断"这一轮客户有没有给出新需求"）
        facts_before = _snapshot_facts(profile)
        message_slots = extract_slots(current_msg_text)
        profile = get_requirement_extractor().extract(
            current_msg_text,
            previous_profile=profile,
            semantic_override=semantic_payload or None,
            # 语义结果的缓存必须按会话隔离（否则会串到别的对话框）
            session_id=str(state.get("session_id") or ""),
        )

        # ── 图片识别结果的确认（客户口径）───────────────────────────────────
        # 带图的那一轮我们已经把"图片里看到什么"说给客户听了；
        # 这一轮（没带图）就是客户的回应：说"对"→ 记为客户确认；
        # 给了别的值 → 客户的值为准（Extractor 已按客户优先合并），并留一条纠正记录。
        if not state.get("vision_applied"):
            from ....vision.integration import resolve_vision_confirmation

            vision_stats = resolve_vision_confirmation(profile, current_msg_text)
            if (
                vision_stats.get("confirmed")
                or vision_stats.get("accepted")
                or vision_stats.get("corrected")
            ):
                logger.info(
                    "[VisionConfirm] confirmed=%s accepted=%s corrected=%s corrections=%s",
                    vision_stats.get("confirmed"), vision_stats.get("accepted"),
                    vision_stats.get("corrected"), profile.vision_corrections,
                )

        # ── v2.1 Phase 3：客户回答意图（不知道 / 你决定 / 不提供）──────────────
        # 客户这句话可以同时给出多个决策（"I don't know the viewing distance either,
        # you can decide the pitch." → viewing_distance=UNKNOWN + pitch=DELEGATED），
        # 所以这里按"槽位 → 决策"逐条落地，而不是只看 last_asked_slot。
        import json as _json

        from ....core.customer_response import detect_response_intents, is_customer_correction

        last_asked = str(getattr(profile, "last_asked_slot", "") or "")
        response_intents = detect_response_intents(
            current_msg_text, last_asked_slot=last_asked
        )
        handled_slots = set()
        for item in response_intents:
            # 只处理"决策类"回答；客户直接给值的情况由 Extractor 负责
            if item.intent not in ("delegated", "declined", "unknown"):
                continue
            if profile.slot_is_confirmed(item.slot):
                # 客户这一轮其实给了值 → 以值为准，不记决策
                continue
            new_state = profile.mark_decision(item.slot, item.intent)
            handled_slots.add(item.slot)
            # 结构化日志（计划第 20 节）：回答"为什么 AI 不问了 / 还在问"
            logger.info(
                "[FieldDecision] %s",
                _json.dumps(
                    {
                        "slot": item.slot,
                        "new_state": new_state,
                        "ask_count": profile.ask_count(item.slot),
                        "action": {
                            "DELEGATED": "infer",
                            "DECLINED": "skip_question",
                            "DEFERRED": "skip_question",
                            "UNKNOWN": "ask_easier",
                        }.get(new_state, "record"),
                        "reason": item.intent,
                        "evidence": item.evidence,
                    },
                    ensure_ascii=False,
                ),
            )

        if is_customer_correction(current_msg_text):
            logger.info("[CustomerResponse] correction detected: %r", current_msg_text[:60])

        # 兜底：新模块没覆盖的说法（如 "not available" / "没有了"）仍按老规则处理，
        # 且"客户让 AI 决定"要映射成 DELEGATED（不是 unknown）。
        if last_asked and last_asked not in handled_slots:
            from ....core.unknown_detector import detect_no_answer

            no_answer_reason = detect_no_answer(current_msg_text)
            if no_answer_reason and not profile.slot_is_confirmed(last_asked):
                mapped = {
                    "customer_skip": "declined",
                    "customer_does_not_know": "unknown",
                    "customer_defers": "delegated",
                }.get(no_answer_reason, "unknown")
                new_state = profile.mark_decision(last_asked, mapped)
                logger.info(
                    "[FieldDecision] %s",
                    _json.dumps(
                        {"slot": last_asked, "new_state": new_state,
                         "ask_count": profile.ask_count(last_asked),
                         "reason": no_answer_reason},
                        ensure_ascii=False,
                    ),
                )
        logger.info(
            "[RequirementExtraction] message=%r extracted=%s confirmed=%s unknown=%s",
            current_msg_text[:80],
            sorted(k for k in message_slots if not str(k).startswith("_")),
            sorted(s for s, v in profile.status.items() if v == "confirmed"),
            profile.unknown_slots(),
        )

        # ── 场景切换 → 清掉依赖上一个场景的字段（原来在 legacy 字典上做）──────
        old_category = _get_scene_category(previous_purpose or "")
        new_category = _get_scene_category(profile.purpose or "")
        if (
            previous_purpose
            and profile.purpose
            and old_category != new_category
            and old_category != "unknown"
        ):
            fields_to_clear: list = []
            for _old_scene, fields in CONTEXT_DEPENDENT_FIELDS.items():
                if _should_clear_field_on_scene_change(
                    old_category, new_category, fields[0] if fields else ""
                ):
                    fields_to_clear.extend(fields)
            for legacy_field in set(fields_to_clear):
                for profile_field in _LEGACY_TO_PROFILE_FIELDS.get(legacy_field, ()):
                    if getattr(profile, profile_field, None) is not None:
                        logger.info(
                            "Clearing stale profile field '%s' (scene %s → %s)",
                            profile_field, old_category, new_category,
                        )
                        setattr(profile, profile_field, None)
                        profile.sources.pop(profile_field, None)

        # IFP 安全规则同理：legacy 侧既然已经把 display_type 去掉了，Profile 也要去掉
        if state["requirements"].get("display_type") in (None, "") and profile.display_type == "IFP":
            logger.info("Clearing IFP display_type from profile (kept only in meeting-room context)")
            profile.display_type = None
            profile.sources.pop("display_type", None)

        # 客户只报了一个长度（"129,2cm"）时，等他指认这是宽 / 高 / 对角线：
        #   - 说"宽度" → 记成宽度；说"高度" → 记成高度
        #   - 说"对角线" → 这个数字不能直接用于箱体排布，丢掉线索，回到"问宽高"
        #   - 客户直接给了宽高 → 线索作废
        if profile.screen_size_hint_mm is not None and not profile.has_target_size:
            axis = str(message_slots.get("size_axis") or "").strip().lower()
            if axis in ("width", "height"):
                hint_m = profile.screen_size_hint_mm / 1000.0
                if axis == "width":
                    profile.target_width_m = hint_m
                    profile.sources["target_width_m"] = "explicit"
                else:
                    profile.target_height_m = hint_m
                    profile.sources["target_height_m"] = "explicit"
                profile.screen_size_hint_mm = None
                profile.sources.pop("screen_size_hint_mm", None)
                logger.info("Resolved size hint to %s = %s m", axis, hint_m)
            elif axis == "diagonal":
                logger.info("Customer said the measurement is a diagonal — asking for width/height instead")
                profile.screen_size_hint_mm = None
                profile.sources.pop("screen_size_hint_mm", None)
        elif profile.has_target_size:
            profile.screen_size_hint_mm = None

        state["requirement_profile"] = profile
        logger.debug("\n%s", profile.describe())

        # ── M1：旧字段由 Profile 单向投影出来（Legacy Adapter）────────────────
        # 旧写法是 `setdefault`（Profile → legacy 只补不删 + legacy → Profile 重建），
        # 属于双向同步：旧值会漂移、线索字段还会"复活"。
        # 现在改为"按 Profile 重建这份视图"：旧模块读到的永远是 Profile 当前值。
        rebuild_legacy_view(state["requirements"], profile)

        # 提问话术轮换：同一槽位有多种自然说法，按"会话 + 轮次"稳定地换一句，
        # 保证问的内容完全一致、措辞不重复。
        # 轮次按"客户已经说过几句话"推进（每轮 +1），这样所有问法都会被轮到；
        # 用消息总条数会每轮跳两步，问法轮换不完整。
        _session_id = str(state.get("session_id") or "")
        _user_turns = sum(
            1
            for msg in (state.get("messages") or [])
            if (
                msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", "")
            ) in ("user", "human")
        )
        _turn_seed = _user_turns + (sum(ord(ch) for ch in _session_id) % 7)
        decision = check_recommendation_ready(profile, variant_seed=_turn_seed)
        state["recommendation_gate"] = decision.to_dict()
        logger.info(
            "[RecommendationGate] status=%s missing=%s unknown=%s",
            decision.status or ("READY" if decision.ready else "CONTINUE_ASKING"),
            decision.missing, decision.unknown_slots,
        )
        current_intent = str(state.get("intent") or "")

        if decision.ready:
            # Gate 放行：触发推荐（后面的 greeting/closing 分支仍可再否决）
            #
            # 【关键】已经推荐过一次之后，客户这一轮如果只是"接着问问题"
            # （既没有给出新的需求信息，也没有明确要求"再推荐 / 报价 / 下单"），
            # 就不该再推荐一遍 —— 否则客户问代理商、付款、认证等任何问题时，
            # 得到的都是同一份推荐（实测反馈："推荐完就一直推荐，接不住客户的话"）。
            # 这种轮次改为走"回答问题"的路径（others / product_question）。
            turn_added_facts = _facts_added(facts_before, profile)
            explicit_reco_request = bool(
                _EXPLICIT_RECO_REQUEST_RE.search(str(current_msg_text or ""))
            )
            if (
                state.get("already_recommended")
                and not turn_added_facts
                and not explicit_reco_request
                # 客户这一轮如果是在**说需求**（哪怕是把视距/尺寸重复一遍、或改了个值），
                # 就按需求重新走推荐，不能丢给自由问答 —— 否则会出现
                # "推荐说 P3.9、回答却讲 around 5"这种自相矛盾（实测日志）。
            ):
                state["should_generate_solution"] = False
                state["pending_question"] = ""
                if current_intent not in ("product_question", "others"):
                    # 保留"回答问题"语义，交给 Solution 的自由问答分支，
                    # 而不是被当成需求采集（否则又会绕回来推荐）
                    state["intent"] = "others"
                    current_intent = "others"
                logger.info(
                    "已推荐过且本轮无新需求（intent=%s）→ 先回答客户，不重复推荐",
                    current_intent,
                )
            else:
                state["should_generate_solution"] = True
                state["pending_question"] = ""
        else:
            # Gate 未放行：本轮不推荐，改为追问一个关键问题
            # 追问内容以 Gate 的 missing 为准（保证问的就是拦住推荐的那一项），
            # 没有对应模板时再退回 question_planner 的扩展问题（如预算）。
            question = decision.next_question or ""
            slot = first_missing_slot(decision.missing) or ""
            # 【关键】图片里已经"看到"、并且这一轮正要跟客户核对的字段，**不要再问同一个问题**。
            # 实测 bug：回复里刚说完 "it looks like … a fixed installation … correct me if I've
            # misread it"，紧接着又问 "is this a long-term installation, or rental?" —— 自相矛盾。
            # 处理：跳过被图片确认覆盖的槽位，改问下一个缺失项；全被覆盖了就本轮不再提问。
            vision_pending = {
                str(x) for x in (getattr(profile, "vision_confirmation_pending", None) or [])
            }
            vision_covered_everything = False
            if vision_pending and slot and slot in vision_pending:
                from ....rag.query_understanding import (  # noqa: PLC0415
                    detect_language,
                )
                from ....rag.readiness import question_for as _question_for
                from ....rag.reply_composer import reply_language as _reply_language

                _language = _reply_language(current_msg_text) or detect_language(current_msg_text)
                alternatives = [s for s in decision.missing if s not in vision_pending]
                if alternatives:
                    next_slot = alternatives[0]
                    logger.info(
                        "[VisionConfirm] slot=%s 已经在图片确认里问过 → 改问 %s",
                        slot, next_slot,
                    )
                    slot = next_slot
                    question = _question_for(slot, _language, _turn_seed, easier=False) or ""
                    if slot == "size_axis":
                        from ....rag.readiness import _size_axis_question  # noqa: PLC0415

                        question = (
                            _size_axis_question(profile, _language, _turn_seed) or question
                        )
                else:
                    logger.info(
                        "[VisionConfirm] 缺少的字段都在图片确认里 → 本轮不再提问"
                    )
                    slot = ""
                    question = ""
                    vision_covered_everything = True
            if not question and not vision_covered_everything:
                plan = plan_next_question(profile, seed=_turn_seed) or {}
                question = plan.get("question") or ""
                slot = plan.get("slot") or slot
            # 需求重置时（客户要换产品 / 换项目），无论 classify 把这句话分到哪一类，
            # 都必须回到"问下一个关键问题"，不能沿用 others / product_question 的自由问答路由。
            # 其余意图（product_question / others）同样要产出待问项：
            # 由 orchestrator 把"回答客户问题"与"继续追问需求"拼成一句回复。
            if question and current_intent != "closing":
                state["pending_question"] = question
                state["pending_slot"] = slot
                # Phase 5/6：记录"这一项已经问过一次"，下一轮最多再问一次；
                # 同时记下 last_asked_slot，客户答"不知道"时才知道是哪一项
                profile.last_asked_slot = slot
                if slot:
                    profile.record_ask(slot)
                logger.info(
                    "[QuestionState] asking slot=%s ask_count=%d status=%s",
                    slot, profile.ask_count(slot), profile.slot_status(slot),
                )
                logger.info("[QuestionPlanner] next_slot=%s (gate=%s)", slot, decision.status)
                logger.info(
                    "RecommendationGate blocked (missing=%s, slot=%s) — asking: %s",
                    decision.missing, slot, question,
                )
            else:
                state["pending_question"] = state.get("pending_question", "")
                state["pending_slot"] = state.get("pending_slot", "")
            state["should_generate_solution"] = False
    except Exception as exc:  # pragma: no cover - 防御式
        logger.warning("Requirement profile planning failed: %s", exc)
        state["pending_question"] = state.get("pending_question", "")
        # 【M11】Gate 异常时必须安全降级：绝不推荐（旧代码会留下 legacy 的 True）
        state["should_generate_solution"] = False
        state["recommendation_gate"] = {
            "ready": False,
            "gate": "recommendation",
            "missing": [],
            "reason": f"gate_error: {exc}",
            "next_question": None,
        }

    # ── M3：旧字段降级为"Gate 结果的投影"（只读兼容，不再参与任何判断）────────
    # 旧写法用 REQUIRED_KEYS（只看 usage）自己算一遍，与 Gate 是两套标准；
    # 现在 required_met / required_missing 只是 Gate 的镜像。
    _gate = state.get("recommendation_gate") or {}
    state["required_met"] = bool(_gate.get("ready", False))
    state["required_missing"] = list(_gate.get("missing") or [])[:2]

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
    
    # Closing：只有在客户**明确要结束**时才收尾。
    # 实测教训：客户回答 "close"（=近）被误判成 closing 时，这一支会把
    # "Gate 已经放行的推荐"也一起否掉，结果不推荐产品、也不问尺寸。
    # 所以：Gate 已就绪 + 客户没有明确说结束 → 照常推荐（把意图纠回 need_query）。
    elif intent == "closing":
        from .classify import is_explicit_closing

        message_text = str(state.get("current_message") or "")
        if state["should_generate_solution"] and not is_explicit_closing(message_text):
            state["intent"] = "need_query"
            state["next_action"] = "router"
            logger.info(
                "Intent 'closing' 但客户并未明确结束且 Gate 已放行 → 照常推荐（next_action=router）"
            )
        else:
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

    # ── 需求重置后必须回到需求采集 ─────────────────────────────────────────
    # 客户在同一会话里换产品 / 换项目时，本轮以"重新问一个关键问题"收尾：
    # 覆盖 classify 给出的 product_question / others 路由，避免绕到自由问答检索。
    if state.get("requirements_reset") and not state.get("should_generate_solution"):
        if str(state.get("intent") or "") != "closing":
            if state.get("pending_question"):
                if state.get("intent") in ("others", "product_question", "industry"):
                    state["intent"] = "need_query"
                state["next_action"] = "ask"
                logger.info(
                    "Requirement reset (%s) — back to requirement mining, asking: %s",
                    state.get("reset_reason"), state.get("pending_question"),
                )

    return state
