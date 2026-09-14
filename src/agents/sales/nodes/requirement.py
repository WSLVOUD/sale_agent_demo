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


def _missing_requirements(requirements: dict) -> list:
    """Return a priority-ordered list of unmet requirement keys."""
    missing = [k for k in REQUIRED_KEYS if not requirements.get(k)]
    if len(missing) >= 2:
        return missing
    missing += [k for k in OPTIONAL_KEYS if not requirements.get(k)]
    return missing


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


def should_trigger_solution(requirements: dict) -> bool:
    """Trigger Solution Agent when all REQUIRED_KEYS are filled."""
    has_all_required = all(requirements.get(k) for k in REQUIRED_KEYS)
    return has_all_required


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
        conversation = "\n".join([
            f"{getattr(msg, 'type', 'unknown')}: {getattr(msg, 'content', str(msg))}"
            for msg in state["messages"][-3:]
        ])
    
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
1. **绝对不要**推断或补全以下任何工程参数：室内/室外、固装/租赁、观看距离、
   屏幕尺寸、亮度、分辨率、点间距 —— 这些由系统的确定性解析器从用户原话提取，
   如果用户没说，必须让系统继续询问，而不是由你猜测。
2. usage 只写用户实际描述的场景原词，不要映射、不要扩展。
3. 没有对应信息就返回 null 或 []，禁止编造。
4. 只返回 JSON，不要任何解释。""")
    
    current_msg = HumanMessage(content=f"已有需求：{state['requirements']}\n\n当前对话：\n{conversation}")
    response = llm.invoke([prompt, current_msg])
    
    # 保存合并前的旧状态，用于上下文变化检测
    pre_merge_requirements = dict(state["requirements"])
    
    # Parse extracted requirements
    import json
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

    # ── v2.0 Phase 4：Recommendation Ready Gate 门控推荐 ────────────────────
    # 旧逻辑只要拿到 usage 就立刻推荐；现在由 Gate 决定：
    #   未就绪 → 不触发推荐，一次只问一个高价值问题
    #   已就绪 → 触发推荐（即使旧 usage 规则尚未满足，例如英文场景表达）
    try:
        from ....models.requirement import RequirementProfile
        from ....rag.query_understanding import extract_slots
        from ....rag.readiness import check_recommendation_ready, first_missing_slot
        from ..question_planner import plan_next_question, should_ask_before_recommend

        profile = RequirementProfile.from_legacy(state["requirements"])
        # 本轮消息里解析出的字段是**客户刚说过的**，必须标记为 explicit，
        # 否则会被误判成"推断值"而挡住推荐（例如客户明说"固定安装"却被当成没说过）。
        message_slots = extract_slots(current_msg_text)
        profile = profile.merge(
            RequirementProfile.from_slots(message_slots, explicit_keys=set(message_slots))
        )

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

        # 把本轮从消息里解析出的结构化事实回写到 requirements，
        # 供下一轮 / 方案 Agent 复用（不覆盖 LLM 或客户已明确的既有值）。
        # 没有这一步时，多轮对话每轮都会从零开始，LLM 不可用时会反复追问同一问题。
        #
        # 【关键防呆】只有"客户明确确认"的字段才允许持久化，
        # 否则"能容纳15个人"估算出的视距会在下一轮被误当成客户说过的话。
        facts = profile.to_facts()
        confirmed = profile.sources

        def _confirmed(field: str) -> bool:
            return confirmed.get(field) in ("explicit", "confirmed")

        writeback: dict = {}
        environment = facts.get("environment")
        if environment:
            writeback["location_type"] = "室外" if environment in ("outdoor", "semi_outdoor") else "室内"
            writeback["indoor"] = environment == "indoor"
            writeback["outdoor"] = environment in ("outdoor", "semi_outdoor")
        if facts.get("installation") and _confirmed("installation"):
            writeback["is_rental"] = facts["installation"] == "rental"
        if profile.purpose:
            writeback["usage"] = profile.purpose
        if profile.viewing_distance_m is not None and _confirmed("viewing_distance_m"):
            writeback["distance"] = f"{profile.viewing_distance_m:g}米"
        if profile.display_type and _confirmed("display_type"):
            writeback["display_type"] = profile.display_type
        if profile.pixel_pitch_mm is not None and _confirmed("pixel_pitch_mm"):
            writeback["pixel_pitch"] = profile.pixel_pitch_mm
        if profile.brightness_min_nit is not None and _confirmed("brightness_min_nit"):
            writeback["brightness_min"] = profile.brightness_min_nit
        if profile.has_target_size and (
            _confirmed("target_width_m") or _confirmed("target_height_m")
        ):
            # 只写客户真正给过的那一维：缺高度时不能补成 "x0米"
            # （实测日志里出现过 size='9.144米x0米' 这种被"补 0"的脏数据）
            width, height = profile.target_width_m, profile.target_height_m
            if width and height:
                writeback["size"] = f"{width:g}米x{height:g}米"
            elif width:
                writeback["size"] = f"{width:g}米宽"
            elif height:
                writeback["size"] = f"{height:g}米高"
        # 只报了一个长度、还没指认方向的线索，必须接着传下去，
        # 否则下一轮客户回"是宽度"，系统已经忘了那个数字
        if (
            profile.screen_size_hint_mm is not None
            and _confirmed("screen_size_hint_mm")
        ):
            writeback["screen_size_hint_mm"] = profile.screen_size_hint_mm
        if profile.size_axis:
            writeback["size_axis"] = profile.size_axis
        for key, value in writeback.items():
            if value not in (None, "", []):
                state["requirements"].setdefault(key, value)

        # 尺寸线索以本轮档案为准：线索已消费（指认成宽/高，或改给宽高）时
        # 必须从 requirements 里删掉，否则 setdefault 会让它在下一轮"复活"
        if profile.screen_size_hint_mm is None:
            state["requirements"].pop("screen_size_hint_mm", None)
        if not profile.size_axis:
            state["requirements"].pop("size_axis", None)

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
