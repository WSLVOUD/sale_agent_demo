"""Deterministic IFP intent rules shared by classification and filtering."""
import re
from typing import Any, Dict, Iterable, List, Optional


EXPLICIT_IFP_TERMS = (
    "ifp", "interactive flat panel", "会议一体机", "触摸一体机",
    "交互平板", "交互式平板", "电子白板",
)

SCENARIO_TERMS = (
    "会议室", "会议", "教室", "学校", "培训室", "培训",
    "conference", "meeting", "classroom", "school", "training room",
)

INTERACTION_TERMS = (
    "手写", "书写", "白板", "批注", "触控", "触摸", "多点触控",
    "annotate", "handwriting", "whiteboard", "touch", "interactive",
)

# 旧 API 兼容别名：测试代码用全小写
_greeting_terms = ("你好", "您好", "hi", "hello", "嗨", "hey", "在吗", "有人吗", "早", "晚安")
_recommendation_terms = ("推荐", "推荐一款", "帮我", "需要", "想要", "想要一款", "选一款",
                        "选一块", "采购", "购买", "用什么样的", "用哪些", "用什么",
                        "会议室", "教室", "培训", "舞台", "广告", "租赁",
                        "演唱会", "体育", "户外", "室内", "屏幕", "手写",
                        "触控", "交互", "白板")
_parameter_terms = ("亮度", "多亮", "几nit", "多少nit", "点间距", "分辨率", "多少寸",
                    "多少英寸", "型号", "保修", "质保", "warranty", "多少毫安",
                    "多大", "多重", "刷新率", "重量", "HDR", "hdr", "IP65", "ip65",
                    "尺寸", "price", "价格", "贵不贵", "多少钱")
_objection_terms = ("贵", "贵吗", "太贵", "便宜", "便宜吗", "比别家", "比.*贵",
                   "质量", "售后", "保修", "保证", "为什么不", "区别", "不同")
_conversation_terms = ("谢谢", "再见", "拜拜", "好的", "ok", "ok", "thanks")


def user_messages_text(messages: Iterable[Any]) -> str:
    """Join only customer-authored messages, excluding assistant suggestions."""
    parts = []
    for message in messages or []:
        role = message.get("role") if isinstance(message, dict) else getattr(message, "type", None)
        if role not in ("user", "human"):
            continue
        content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
        if content:
            parts.append(str(content))
    return " ".join(parts)


IFP_OUTPUT_RE = re.compile(
    r"(?:ifp|interactive flat panel|会议一体机|触摸一体机|交互(?:式)?平板|电子白板|T\d{2}Omni)",
    re.IGNORECASE,
)


def remove_unsupported_ifp_text(text: str) -> str:
    """Remove complete sentences that introduce IFP without customer intent."""
    if not text or not IFP_OUTPUT_RE.search(text):
        return text

    kept = []
    for line in text.splitlines():
        sentences = re.split(r"(?<=[。！？!?])", line)
        clean_line = "".join(
            sentence for sentence in sentences if not IFP_OUTPUT_RE.search(sentence)
        ).strip()
        if clean_line:
            kept.append(clean_line)
    return "\n".join(kept).strip()


def has_ifp_intent(requirement: Optional[Dict[str, Any]] = None, *, user_text: Optional[str] = None) -> bool:
    """Require explicit IFP wording or both a scenario and interaction need.

    When ``user_text`` is supplied it is authoritative, so inferred requirement
    fields cannot manufacture IFP intent. Explicitly negated IFP or interaction
    needs are removed before matching.
    """
    source = user_text if user_text is not None else " ".join(
        str(value) for value in (requirement or {}).values() if value
    )
    text = source.lower()

    # Check for explicit negation
    if re.search(
        r"(?:不要|不需要|无需|不考虑|排除|没有|没提到).{0,8}"
        r"(?:ifp|interactive flat panel|会议一体机|触摸一体机|交互(?:式)?平板|电子白板)",
        text,
    ):
        return False

    # Clean up negated requirements
    cleaned = text
    negations = r"不要|不需要|无需|不考虑|没有|没提到|不要求"
    interaction_group = "|".join(
        sorted((re.escape(term) for term in INTERACTION_TERMS), key=len, reverse=True)
    )
    
    # Remove coordinated negated requirements
    cleaned = re.sub(
        rf"(?:{negations})(?:任何)?\s*(?:{interaction_group})"
        rf"(?:(?:和|与|、|或|以及)\s*(?:{interaction_group}))*",
        "",
        cleaned,
    )
    
    for term in INTERACTION_TERMS:
        escaped = re.escape(term)
        cleaned = re.sub(rf"(?:{negations}).{{0,6}}{escaped}", "", cleaned)
        cleaned = re.sub(rf"{escaped}.{{0,6}}(?:{negations})", "", cleaned)

    # Check explicit IFP terms
    if any(term in cleaned for term in EXPLICIT_IFP_TERMS):
        return True

    # Check scenario + interaction combination
    return (
        any(term in cleaned for term in SCENARIO_TERMS)
        and any(term in cleaned for term in INTERACTION_TERMS)
    )


# ── 简化版意图分类（Phase 13 测试需要） ──────────────────────────
def classify_intent(message: str) -> str:
    """对单条消息做轻量意图分类（不调用 LLM，用于测试与快速分支）。

    返回值：
        - "greeting"             寒暄 / 问候
        - "product_recommendation"  想要/咨询某类产品的推荐
        - "parameter_query"      询问某个具体参数
        - "objection"            异议（价格、质量、售后）
        - "conversation"         一般闲聊（感谢 / 结束）
        - "general"              其它无法分类的（兜底）
    """
    if not message:
        return "general"

    text = message.strip()
    lower = text.lower()

    # 1) 寒暄
    stripped = re.sub(r"[，。,\s!\?！？,.。]+", "", lower)
    if stripped in ("你好", "您好", "hi", "hello", "嗨", "hey", "在吗", "有人吗"):
        return "greeting"

    # 2) 参数查询（"X 的亮度是多少" / "TW3 的型号"）
    param_patterns = (
        r"亮度.{0,3}(多少|几|是什么|多少|呢|\?|？)",
        r"(多少|几|几nit|多少nit|几尼特)",
        r".{0,4}(多大|多重|多厚|几寸|几英寸|几公斤)",
        r"(保修|质保|warranty|寿命|耐用).{0,5}(几年|多长|多久|\?|？)",
        r"支持.{0,3}(HDR|hdr|IP65|ip65|防水)",
        r".{0,6}(分辨率|点间距|像素|刷新率).{0,3}(是多少|几|什么|\?|？)",
    )
    for pat in param_patterns:
        if re.search(pat, text):
            # 排除包含 "推荐" 的语义（推荐优先）
            if "推荐" not in text and "选" not in text:
                return "parameter_query"

    # 3) 异议
    if any(re.search(p, text) for p in _objection_terms):
        return "objection"

    # 4) 推荐 / 需求
    if any(kw in text for kw in _recommendation_terms):
        return "product_recommendation"

    # 5) 一般闲聊
    if any(kw in text for kw in _conversation_terms):
        return "conversation"

    # 兜底
    return "general"


def extract_entities(message: str) -> Dict[str, Any]:
    """从用户消息中提取结构化实体。

    返回字典（可能为空）：
        {
            "product_models": ["TW21-3216", "TW11-OD", ...],
            "pixel_pitch":    2.5,
            "brightness":     5000,
            "size":           "65英寸",
            "budget":         200000,  # 预算（元）
            "capacity":       20,      # 容纳人数
            "distance":       "4米",
            "area":           "20平米",
            "environment":    "indoor" | "outdoor" | "semi_outdoor",
            "display_type":   "LED" | "LCD" | "IFP",
            "is_rental":      True/False,
            "interaction":    [列表],
            "purpose":        "...",
        }
    """
    entities: Dict[str, Any] = {}
    if not message:
        return entities

    text = message.strip()

    # 1) 产品型号（TW11-3216、OmniPAD-G4 等）
    # 中文文本无单词边界，需要更宽松的 lookahead/behind
    model_patterns = [
        # TW 系列：TW + 1+ 数字 + 可选 (./- + 字母数字)
        # 1) bare "TW<数字>"（如 TW3 / TW2.5）
        r"(?<![A-Za-z])(TW\d+\.?\d*)(?![A-Za-z0-9])",
        # 2) 复合 "TW<数字>-..." / "TW<数字>.<字母>"（如 TW11-OD / TW2.5-B）
        r"(?<![A-Za-z])(TW\d+\.?[\dA-Za-z]*(?:[\-\.][A-Za-z0-9]+)*)(?![A-Za-z0-9])",
        # OmniPAD-X
        r"(OmniPAD-[\w\d]+)",
        # HG / H 系列
        r"(HG\d+\b[A-Za-z\-]*)",
        r"(H\d{4,}[A-Za-z\-]*)",
        # DLP / RGB / DS 系列
        r"(DLP-[\w]+)", r"(RGB-[\w]+)", r"(DS-[\w]+)",
    ]
    product_models: List[str] = []
    seen: set = set()
    for pat in model_patterns:
        for m in re.finditer(pat, text):
            token = m.group(1)
            if token in seen:
                continue
            seen.add(token)
            product_models.append(token)
    if product_models:
        entities["product_models"] = product_models

    # 2) 点间距 P2.5 / 点间距3mm
    pitch = re.search(r"[Pp](\d+(?:\.\d+)?)", text)
    if pitch:
        entities["pixel_pitch"] = float(pitch.group(1))
    else:
        mm = re.search(r"点间距\s*(\d+(?:\.\d+)?)\s*mm", text)
        if mm:
            entities["pixel_pitch"] = float(mm.group(1))

    # 3) 亮度
    br = re.search(r"(\d{3,5})\s*nit", text, re.IGNORECASE)
    if br:
        entities["brightness"] = int(br.group(1))

    # 4) 尺寸
    size = re.search(r"(\d+)\s*(?:寸|英寸|inch)", text, re.IGNORECASE)
    if size:
        entities["size"] = f"{size.group(1)}英寸"
    meter = re.search(r"(\d+(?:\.\d+)?)\s*米\s*(?:宽|长|高|对角|diagonal|屏幕|屏)", text)
    if meter:
        entities["size"] = f"{meter.group(1)}米"

    # 5) 预算
    budget = re.search(r"预算\s*(\d+(?:\.\d+)?)\s*(万|w|w|元|块|rmb)?", text, re.IGNORECASE)
    if budget:
        value = float(budget.group(1))
        unit = budget.group(2)
        if unit and "万" in unit:
            value *= 10000
        entities["budget"] = int(value)

    # 6) 容纳人数
    cap = re.search(r"(\d+)\s*人", text)
    if cap:
        entities["capacity"] = int(cap.group(1))

    # 7) 距离（视距）
    dist = re.search(r"(\d+(?:\.\d+)?)\s*米", text)
    if dist:
        entities["distance"] = f"{dist.group(1)}米"

    # 8) 面积
    area = re.search(r"(\d+(?:\.\d+)?)\s*(?:平|平方米|㎡|平方)", text)
    if area:
        entities["area"] = f"{area.group(1)}平米"

    # 9) 环境
    if any(kw in text for kw in ("户外", "室外", "露天", "outdoor")):
        entities["environment"] = "outdoor"
    elif any(kw in text for kw in ("半户外", "半室外", "遮阳")):
        entities["environment"] = "semi_outdoor"
    elif any(kw in text for kw in ("室内", "户内", "indoor")):
        entities["environment"] = "indoor"

    # 10) 屏幕类型
    if any(kw in text for kw in ("lcd", "LCD", "拼接", "拼接屏")):
        entities["display_type"] = "LCD"
    elif any(kw in text.lower() for kw in ("ifp", "会议一体机", "触摸一体机", "交互平板", "电子白板")):
        entities["display_type"] = "IFP"
    elif any(kw in text for kw in ("led", "LED", "屏幕", "显示屏")):
        entities["display_type"] = "LED"

    # 11) 租赁
    if any(kw in text for kw in ("租赁", "租用", "短租", "rental", "活动", "演唱会")):
        entities["is_rental"] = True
    elif any(kw in text for kw in ("固定", "会议室", "教室", "永久", "固装")):
        entities["is_rental"] = False

    # 12) 交互特征
    ifp_interaction = []
    for kw in INTERACTION_TERMS:
        if kw in text:
            ifp_interaction.append(kw)
    if ifp_interaction:
        entities["interaction"] = ifp_interaction

    # 13) 场景 purpose
    for kw in ("会议室", "教室", "展厅", "商场", "户外广告", "演唱会",
              "酒店大堂", "指挥中心", "监控中心"):
        if kw in text:
            entities["purpose"] = kw
            break

    return entities
