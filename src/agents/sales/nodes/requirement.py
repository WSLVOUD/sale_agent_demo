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
6. 只返回 JSON，不要任何解释。""")
    
    current_msg = HumanMessage(content=f"已有需求：{state['requirements']}\n\n当前对话：\n{conversation}")
    response = llm.invoke([prompt, current_msg])
    
    # 保存合并前的旧状态，用于上下文变化检测
    pre_merge_requirements = dict(state["requirements"])
    
    # Parse extracted requirements
    import json
    semantic_payload: dict = {}
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

    # ── 关键修复：场景回答不能被 classify 误判成 others/product_question ──
    # 当客户正在回答"用在什么场景"（例如只说 church），这属于需求采集，
    # 必须继续走 need_query → Gate → 追问，而不是绕到 Solution 的自由问答检索。
    current_intent = str(state.get("intent") or "")
    if current_intent in ("others", "product_question"):
        from ....rag.query_understanding import extract_slots

        current_slots = extract_slots(current_msg_text)
        has_scenario = bool(state["requirements"].get("usage")) or bool(
            current_slots.get("purpose")
        )
        # 客户"报需求"（室内外 / 安装方式 / 视距 / 尺寸）而不是"问问题"时，
        # 必须留在需求采集流程：否则会绕到自由问答，在 Gate 没通过的情况下
        # 把一堆型号和参数倒给客户（实测出现过）。
        if has_scenario or _looks_like_requirement_answer(current_msg_text, current_slots):
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
        message_slots = extract_slots(current_msg_text)
        profile = get_requirement_extractor().extract(
            current_msg_text,
            previous_profile=profile,
            semantic_override=semantic_payload or None,
            # 语义结果的缓存必须按会话隔离（否则会串到别的对话框）
            session_id=str(state.get("session_id") or ""),
        )

        # ── Phase 4~9：Unknown 容错 ──────────────────────────────────────────
        # 1) 客户"不知道 / 跳过"的是**上一轮问的那一项**（last_asked_slot）；
        #    如果这一轮他其实给了其它字段，Extractor 已经全量记录，互不影响。
        # 2) 问满两次仍无值 → unknown，不再阻塞；Gate 会转 DEGRADED_READY。
        from ....core.unknown_detector import detect_no_answer

        last_asked = str(getattr(profile, "last_asked_slot", "") or "")
        no_answer_reason = detect_no_answer(current_msg_text)
        # 注意：判定用 slot_is_confirmed 而不是 slot_value_present ——
        # 场景默认值（例如"教堂默认固装"）虽然"有值"，但客户对安装方式说了
        # "不知道"时仍必须走 unknown 流程，否则同一问题会被无限追问。
        if last_asked and no_answer_reason and not profile.slot_is_confirmed(last_asked):
            profile.mark_unknown(last_asked, no_answer_reason)
            logger.info(
                "[QuestionState] slot=%s ask_count=%d status=%s reason=%s",
                last_asked, profile.ask_count(last_asked),
                profile.slot_status(last_asked), no_answer_reason,
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
            state["should_generate_solution"] = True
            state["pending_question"] = ""
        else:
            # Gate 未放行：本轮不推荐，改为追问一个关键问题
            # 追问内容以 Gate 的 missing 为准（保证问的就是拦住推荐的那一项），
            # 没有对应模板时再退回 question_planner 的扩展问题（如预算）。
            question = decision.next_question or ""
            slot = first_missing_slot(decision.missing) or ""
            if not question:
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
