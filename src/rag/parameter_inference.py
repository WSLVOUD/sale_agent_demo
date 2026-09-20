"""
Phase 5：统一的 Parameter Inference。

改造前的问题（计划文档「九、Phase 5」）：
    同一件事存在多套实现 —— ``_rule_infer_pitch`` / ``_rule_infer_brightness`` /
    ``_rule_infer_rental`` / ``ParameterInference`` / ``infer_parameters_node`` 里
    各自写了一份规则，结果互相覆盖、彼此冲突（例如 4m 视距在一处推出 P≤3.0，
    在另一处推出 P≤4.0）。

改造后的单一链路：

    客户语言
        ↓  extract_slots（规则事实提取，Phase 4；可选 LLM 补充事实）
    Requirement Profile（结构化事实）
        ↓  infer_technical_parameters（纯 Python 工程规则）
    技术参数（点间距区间 / 亮度区间 / 固装租赁 / 屏幕尺寸建议）

两条硬性约束：
  1. LLM 只负责"提取事实"，绝不参与工程参数计算；
  2. 客户显式给出的参数优先，推断值不得覆盖客户明确值。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

from src.rag.query_understanding import (
    _DISPLAY_TYPE_KEYWORDS,
    _FIXED_KEYWORDS,
    _INDOOR_KEYWORDS,
    _OUTDOOR_KEYWORDS,
    _RENTAL_KEYWORDS,
    _SEMI_OUTDOOR_KEYWORDS,
    extract_slots,
    looks_like_question,
)
from src.core.llm import get_llm

logger = logging.getLogger(__name__)

DEFAULT_PITCH_TOLERANCE = 0.5

# 点间距业务规则见下方 _INDOOR_PITCH_TABLE / _OUTDOOR_PITCH_TABLE
# （客户口径：室内 ≤3m→P2.5 及以下、>3m→P3 及以上；
#   室外 4m→P4、5m→P4/P5、6~20m→P5、>30m→P10）


# ── 唯一权威规则表 ──────────────────────────────────────────────────────────
# 观看距离 → 推荐点间距区间（计划文档示例：5m → 2.5~3.0mm）
VIEWING_DISTANCE_PITCH_TABLE: Tuple[Tuple[float, float, float], ...] = (
    (2.0, 0.6, 1.5),      # < 2m
    (4.0, 0.9, 2.0),      # 2–4m
    (8.0, 1.5, 3.0),      # 4–8m
    (15.0, 2.5, 5.0),     # 8–15m
    (30.0, 4.0, 8.0),     # 15–30m
    (float("inf"), 6.0, 10.0),  # ≥30m
)

# ── 环境 + 距离 → 点间距（业务规则，优先于上面的通用距离表）────────────────
# 室内：
#   ≤3m  → P2.5 及以下（越近越细，首选 2.5）
#   >3m  → P3 及以上（首选 P3）
# 室外：
#   ≤4m  → P4（客户 4m 就给 P4）
#   4–6m → P4/P5（首选 4.5 附近）
#   6–20m→ P5 最合适（首选 5.0）
#   20–30m → P6.67 左右
#   >30m → 一律 P10
_INDOOR_PITCH_TABLE: Tuple[Tuple[float, float, float, float], ...] = (
    # (距离上限, 下限, 上限, 首选)
    (3.0, 0.6, 2.5, 2.5),
    # 客户口径：特别远的（>30m）一律 P10 —— 室内外一样，不能因为"室内"就一直按 P3 推
    (30.0, 3.0, 10.0, 3.0),
    (float("inf"), 8.0, 10.0, 10.0),
)
_OUTDOOR_PITCH_TABLE: Tuple[Tuple[float, float, float, float], ...] = (
    (4.0, 3.9, 5.0, 4.0),        # ≤4m  → P4
    (5.0, 3.9, 5.5, 4.5),        # 4–5m → P4 / P5
    (20.0, 4.5, 6.7, 5.0),       # 6–20m → P5（客户说这个区间 P5 最合适）
    (25.0, 5.0, 8.0, 6.7),       # 20–25m → P6.67
    (30.0, 8.0, 10.0, 8.0),      # 25–30m → P8
    (float("inf"), 8.0, 10.0, 10.0),  # >30m → 一律 P10
)


def preferred_pitch_for_environment(
    environment: Optional[str],
    distance_m: Optional[float],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """按"环境 + 观看距离"给出 (点间距下限, 上限, 首选值)，单位 mm。

    这是客户口径的业务规则，比通用距离表优先；客户自己点名点间距时不使用。
    """
    env = str(environment or "").strip().lower()
    if distance_m is None:
        return None, None, None
    try:
        value = float(distance_m)
    except (TypeError, ValueError):
        return None, None, None
    if value <= 0:
        return None, None, None

    # semi_outdoor（半户外）按室外口径处理（客户口径里"室外"包含门头/半户外）
    table = (
        _INDOOR_PITCH_TABLE
        if env == "indoor"
        else _OUTDOOR_PITCH_TABLE
        if env in ("outdoor", "semi_outdoor")
        else None
    )
    if not table:
        return None, None, None
    # 业务口径按"以内"理解（例如"6~20m 以内 P5 都合适"→ 20m 也归 P5 档）
    for limit, low, high, target in table:
        if value <= limit:
            return low, high, target
    last = table[-1]
    return last[1], last[2], last[3]

# 使用环境 → 亮度区间（最低亮度取自目录中该类产品的最低规格）
BRIGHTNESS_BY_ENVIRONMENT: Dict[str, Tuple[Optional[int], Optional[int]]] = {
    "outdoor": (4500, None),
    "semi_outdoor": (800, None),
    "indoor": (400, 800),
}

# 观看距离 → 建议屏幕尺寸（对 LCD/IFP 场景仍有用，LED 场景仅作参考）
SCREEN_SIZE_BY_DISTANCE: Tuple[Tuple[float, str], ...] = (
    (3.0, "65-75英寸"),
    (4.0, "75-86英寸"),
    (6.0, "86-98英寸"),
    (float("inf"), "98英寸以上"),
)


# ── 纯函数：工程推断 ────────────────────────────────────────────────────────
def pitch_range_for_distance(distance_m: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    """观看距离（米）→ 推荐点间距区间（mm）。"""
    if distance_m is None:
        return None, None
    try:
        value = float(distance_m)
    except (TypeError, ValueError):
        return None, None
    if value <= 0:
        return None, None
    for index, (limit, pitch_min, pitch_max) in enumerate(VIEWING_DISTANCE_PITCH_TABLE):
        # 第一档（近距离）取闭区间：客户常说"2 米以内"，2.0m 应归入最细点间距档；
        # 其余档位保持左闭右开（4m 属于 4~8m 档），与 v1.0 的工程规则一致。
        if (value <= limit) if index == 0 else (value < limit):
            return pitch_min, pitch_max
    return 6.0, 10.0


def brightness_range_for_environment(
    environment: Optional[str],
) -> Tuple[Optional[int], Optional[int]]:
    """使用环境 → 亮度区间（nit）。"""
    if not environment:
        return None, None
    return BRIGHTNESS_BY_ENVIRONMENT.get(str(environment), (None, None))


def screen_size_for_distance(distance_m: Optional[float]) -> Optional[str]:
    """观看距离 → 建议屏幕尺寸（参考值）。"""
    if distance_m is None:
        return None
    try:
        value = float(distance_m)
    except (TypeError, ValueError):
        return None
    for limit, size in SCREEN_SIZE_BY_DISTANCE:
        if value <= limit:
            return size
    return None


# v2.1 Phase 7：客户授权 AI 决定尺寸时的**确定性**参考尺寸
# 规则（16:9 + 行业常用的"观看距离 ≈ 3 倍屏高"经验值）：
#     height = clamp(distance / 3, 1.0m, 12.0m)
#     width  = height * 16 / 9
# 没有观看距离 → None（交给 Calculation Gate 延后，绝不瞎猜）
DELEGATED_SIZE_MIN_H_M = 1.0
DELEGATED_SIZE_MAX_H_M = 12.0
DELEGATED_SIZE_ASPECT = 16 / 9
DELEGATED_SIZE_DISTANCE_FACTOR = 3.0      # 观看距离 ÷ 3 ≈ 舒适屏高


def suggest_screen_size(facts: Any) -> Optional[Tuple[float, float]]:
    """客户授权 AI 决定尺寸时，按观看距离给出**参考尺寸**（米）。

    注意（计划第 14 / 15 节）：这是 Python 的确定性推导，不是 LLM 猜数字；
    推导结果只用于工程计算与话术参考，**不会写进客户的确认事实**。
    """
    if facts is None:
        return None
    if hasattr(facts, "to_facts"):
        facts = facts.to_facts()
    distance = None
    if isinstance(facts, dict):
        distance = facts.get("viewing_distance_m") or facts.get("distance")
    distance = parse_distance(distance)
    if not distance or distance <= 0:
        return None
    height = max(DELEGATED_SIZE_MIN_H_M, min(DELEGATED_SIZE_MAX_H_M,
                                             float(distance) / DELEGATED_SIZE_DISTANCE_FACTOR))
    width = height * DELEGATED_SIZE_ASPECT
    return round(width, 2), round(height, 2)


def parse_distance(text: Any) -> Optional[float]:
    """从 "4米" / "4m" / "4000mm" 这类文本解析观看距离（米）。"""
    if text is None or text == "":
        return None
    if isinstance(text, (int, float)):
        return float(text) if text else None
    match = re.search(r"(\d+(?:\.\d+)?)", str(text))
    if not match:
        return None
    value = float(match.group(1))
    lowered = str(text).lower()
    if "mm" in lowered or "毫米" in lowered or value > 100:
        value = value / 1000
    return value or None


def environment_from_facts(facts: Dict[str, Any]) -> Optional[str]:
    """从事实字典推导使用环境（兼容 indoor/outdoor 布尔与 environment 字符串两套写法）。"""
    environment = facts.get("environment")
    if environment in ("indoor", "outdoor", "semi_outdoor"):
        return environment
    if _truthy(facts.get("semi_outdoor")):
        return "semi_outdoor"
    if _truthy(facts.get("outdoor")):
        return "outdoor"
    if _truthy(facts.get("indoor")):
        return "indoor"
    return None


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "是")


def infer_rental_from_text(text: str, purpose: str = "") -> Optional[bool]:
    """租赁 / 固装的统一判定（关键词表只有一份，来自 Phase 4）。"""
    haystack = f"{text} {purpose}".lower()
    if any(keyword.lower() in haystack for keyword in _RENTAL_KEYWORDS):
        return True
    if any(keyword.lower() in haystack for keyword in _FIXED_KEYWORDS):
        return False
    return None


def infer_technical_parameters(facts: Dict[str, Any]) -> Dict[str, Any]:
    """Requirement Profile（事实）→ 技术参数（工程推断）。

    客户显式指定的点间距 / 亮度优先，且不会被推断值覆盖。
    """
    facts = dict(facts or {})
    environment = environment_from_facts(facts)
    distance_m = facts.get("viewing_distance_m")
    if distance_m is None:
        distance_m = parse_distance(facts.get("distance"))

    # 业务规则优先：环境 + 距离 → 点间距区间与首选值（室内 P2.5/P3，室外 P4/P5/P10）
    pitch_min, pitch_max, pitch_target = preferred_pitch_for_environment(
        environment, distance_m
    )
    if pitch_min is None:
        # 环境/距离表不覆盖（例如半户外或没给距离）→ 退回通用距离表
        pitch_min, pitch_max = pitch_range_for_distance(distance_m)
        if distance_m is not None:
            logger.info(
                "点间距规则未命中（environment=%r / %sm）→ 用通用距离表 %s~%smm",
                environment, distance_m, pitch_min, pitch_max,
            )
    brightness_min, brightness_max = brightness_range_for_environment(environment)
    source: Dict[str, str] = {}

    # 客户显式指定点间距 → 覆盖推断区间
    explicit_pitch = facts.get("pixel_pitch_mm")
    if explicit_pitch is None:
        explicit_pitch = facts.get("pixel_pitch")
    if explicit_pitch is not None:
        tolerance = float(facts.get("pixel_pitch_tolerance") or DEFAULT_PITCH_TOLERANCE)
        pitch_min = float(explicit_pitch) - tolerance
        pitch_max = float(explicit_pitch) + tolerance
        source["pixel_pitch"] = "explicit"
        # 客户点名了点间距 → 首选值就是客户的（场景/环境偏好不再参与）
        pitch_target = float(explicit_pitch)
    else:
        if pitch_target is not None:
            source["pixel_pitch"] = "inferred_from_environment_distance"
            logger.info(
                "点间距规则（%s / %sm）：%s~%smm，首选 P%s",
                environment, distance_m, pitch_min, pitch_max, pitch_target,
            )
        elif pitch_min is not None:
            source["pixel_pitch"] = "inferred_from_distance"

    # 客户显式指定亮度下限 → 覆盖推断值
    explicit_brightness = facts.get("brightness_min")
    if explicit_brightness is None:
        explicit_brightness = facts.get("brightness_min_nit")
    if explicit_brightness is not None:
        brightness_min = int(explicit_brightness)
        source["brightness"] = "explicit"
    elif brightness_min is not None:
        source["brightness"] = "inferred_from_environment"

    installation = facts.get("installation")
    if installation in ("fixed", "rental"):
        is_rental: Optional[bool] = installation == "rental"
        source["installation"] = "explicit"
    else:
        is_rental = infer_rental_from_text(
            str(facts.get("_raw_message", "")), str(facts.get("purpose", ""))
        )
        if is_rental is not None:
            source["installation"] = "inferred_from_keywords"
        elif environment in ("indoor", "outdoor"):
            # 已确定环境且无租赁信号 → 固装（租赁必须显式说明）
            is_rental = False
            source["installation"] = "default_fixed"

    return {
        "environment": environment,
        "viewing_distance_m": distance_m,
        "pixel_pitch_min_mm": pitch_min,
        "pixel_pitch_max_mm": pitch_max,
        # 首选点间距（mm）：室内 ≤3m→2.5 / >3m→3.0；室外 4m→4.0 / 5m→4.5 /
        # 6~20m→5.0 / 20~30m→6.7 / >30m→10.0。客户端点名时 = 客户的值。
        "pitch_target_mm": pitch_target,
        "brightness_min_nit": brightness_min,
        "brightness_max_nit": brightness_max,
        "is_rental": is_rental,
        "screen_size": screen_size_for_distance(distance_m),
        "source": source,
    }


# ── 事实提取（可选 LLM 补充）────────────────────────────────────────────────
FACT_EXTRACTION_PROMPT = """你是一名需求事实提取器。请从客户消息中提取**事实**，不要做任何工程推断或推荐。

客户消息：
{message}

已有事实（不要覆盖非空值）：
{existing}

请输出 JSON（仅 JSON，不要解释）：
{{
  "environment": "indoor" / "outdoor" / "semi_outdoor" / null,
  "installation": "fixed" / "rental" / null,
  "purpose": "客户描述的使用场景原文" / null,
  "viewing_distance_m": 数字（米）/ null,
  "target_width_mm": 数字（毫米）/ null,
  "target_height_mm": 数字（毫米）/ null,
  "budget_level": "low" / "mid" / "high" / null,
  "special_requirements": ["客户明确提出的特殊要求"]
}}

规则：
1. 只提取客户明确说出的信息，不确定就返回 null，禁止猜测。
2. 不要输出点间距、亮度等技术参数 —— 这些由 Python 规则计算。
3. 只输出 JSON。"""


def _rule_slots_from_message(message: str) -> Dict[str, Any]:
    """纯规则提取槽位（不调用 LLM）。"""
    from src.rag.query_understanding import extract_slots

    slots = extract_slots(message or "")
    return {
        key: value
        for key, value in (slots or {}).items()
        if not str(key).startswith("_") and value not in (None, "", [], {})
    }


def _requirement_from_slots(slots: Dict[str, Any]) -> Dict[str, Any]:
    """槽位 → Sales 可读的 legacy requirements。"""
    from src.models.legacy_adapter import profile_to_legacy
    from src.models.requirement import RequirementProfile

    profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
    return profile_to_legacy(profile)


def extract_requirements(message: str, existing: Dict = None) -> Dict[str, Any]:
    """提取客户事实：规则槽位为主，LLM 只补充规则没拿到的字段。

    返回 ``{"requirement": {...}}``。即使 LLM 成功但返回全 null，
    仍保留 extract_slots 解析出的 LED / 尺寸 / 室内外等事实。
    """
    existing = existing or {}
    rule_slots = _rule_slots_from_message(message)
    facts: Dict[str, Any] = dict(rule_slots)
    try:
        prompt = FACT_EXTRACTION_PROMPT.format(
            message=message,
            existing=json.dumps(existing, ensure_ascii=False) if existing else "无",
        )
        response = get_llm(temperature=0).invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in content:
            content = content.split("```", 1)[1].split("```", 1)[0]
        parsed = json.loads(content.strip())
        llm_facts = {k: v for k, v in parsed.items() if v not in (None, "", [], {})}
        for key, value in llm_facts.items():
            facts.setdefault(key, value)
    except Exception as exc:
        logger.warning("extract_requirements LLM failed, keep rule slots: %s", exc)

    requirement = _requirement_from_slots(facts) if facts else {}
    for key, value in facts.items():
        requirement.setdefault(key, value)
    return {"requirement": requirement}


# ── 兼容层：旧 API（保持既有调用方与测试可用）──────────────────────────────
class ParameterInference:
    """参数推断器（兼容旧 API）。

    ``extract_constraints`` 现在是 Phase 4 ``extract_slots`` 的适配器，
    不再维护第二套关键词表；工程推断统一走 ``infer_technical_parameters``。
    """

    def extract_constraints(self, message: str) -> Dict[str, Any]:
        """从单条用户消息提取结构化约束（纯规则，不调用 LLM）。"""
        if not message:
            return {}
        text = str(message)
        slots = extract_slots(text)
        constraints: Dict[str, Any] = {}

        # 环境
        environment = slots.get("environment")
        if environment == "indoor":
            constraints["indoor"] = True
            constraints["outdoor"] = False
        elif environment == "outdoor":
            constraints["outdoor"] = True
            constraints["indoor"] = False
        elif environment == "semi_outdoor":
            constraints["semi_outdoor"] = True
            constraints["outdoor"] = True
            constraints["indoor"] = False

        # 产品类型
        if slots.get("display_type"):
            constraints["display_type"] = slots["display_type"]
        lowered = text.lower()
        if any(kw in text for kw in ("不需要手写", "不需要IFP", "不要IFP", "普通显示屏", "普通屏")):
            constraints["exclude_ifp"] = True
        if "拼接" in text or "拼接屏" in text or "拼接墙" in text:
            constraints["is_splicing"] = True
            constraints.setdefault("display_type", "LCD")

        # 点间距
        if slots.get("pixel_pitch_mm") is not None:
            constraints["pixel_pitch"] = float(slots["pixel_pitch_mm"])
            constraints["pixel_pitch_tolerance"] = DEFAULT_PITCH_TOLERANCE

        # 亮度
        if slots.get("brightness_min") is not None:
            constraints["brightness_min"] = int(slots["brightness_min"])

        # 尺寸
        if slots.get("size_inch") is not None:
            constraints["size_inch"] = slots["size_inch"]

        # 固装 / 租赁
        if slots.get("installation"):
            constraints["is_rental"] = slots["installation"] == "rental"
        else:
            rental = infer_rental_from_text(text, str(slots.get("purpose", "")))
            if rental is not None:
                constraints["is_rental"] = rental

        # 特殊功能
        for slot_key, constraint_key in (
            ("waterproof", "waterproof"),
            ("cob", "cob"),
            ("hdr", "hdr"),
            ("gob", "gob"),
        ):
            if slots.get(slot_key):
                constraints[constraint_key] = True

        # 场景 / 视距（保留旧键名，供 rerank / 话术使用）
        if slots.get("purpose"):
            constraints["purpose"] = slots["purpose"]
        if slots.get("viewing_distance_m") is not None:
            constraints["distance"] = f"{slots['viewing_distance_m']:g}米"
        if slots.get("model"):
            constraints["model"] = slots["model"]
        elif slots.get("series_id"):
            constraints["series_id"] = slots["series_id"]
        if slots.get("budget_level"):
            constraints["budget_level"] = slots["budget_level"]

        return constraints

    # ── 旧方法：统一委托到权威规则表 ────────────────────────────────────
    def infer_brightness(
        self,
        constraint: Dict[str, Any],
        message: str = "",
    ) -> Tuple[Optional[int], Optional[int]]:
        """环境 → 亮度区间（唯一规则表）。"""
        environment = environment_from_facts(constraint or {})
        if environment is None:
            environment = environment_from_facts(extract_slots(message or ""))
        return brightness_range_for_environment(environment)

    def infer_pitch(self, distance: str) -> Tuple[Optional[float], Optional[float]]:
        """观看距离 → 点间距区间（唯一规则表）。"""
        return pitch_range_for_distance(parse_distance(distance))

    def infer_rental(
        self,
        constraint: Dict[str, Any],
        message: str = "",
    ) -> Optional[bool]:
        """租赁判定（唯一关键词表）。"""
        return infer_rental_from_text(
            message or str((constraint or {}).get("_raw_message", "")),
            str((constraint or {}).get("purpose", "")),
        )

    def infer_technical_parameters(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        """对外暴露统一工程推断。"""
        return infer_technical_parameters(facts)


# ── Intent 检测（统一入口：规则优先，可选 LLM 兜底）─────────────────────────
_RECOMMENDATION_PATTERNS = (
    r"需要.*屏", r"需要.*屏幕", r"要.*屏", r"要.*屏幕", r"想买", r"采购", r"推荐",
    r"选.*屏", r"选.*屏幕", r"帮我选", r"用.*场景", r"用.*地方", r"安装.*地方",
    r"适合.*用", r"租赁", r"报价", r"方案",
)

_VALID_INTENTS = ("recommendation", "product_question", "conversation", "others")

# 客户**明确在要推荐**（即使写成疑问句："你们能推荐一款室内的吗？"）
_RECO_REQUEST_PATTERNS = (
    r"推荐", r"帮我选", r"选一款", r"选一个", r"给个建议", r"建议一下",
    r"哪个好", r"哪款好", r"哪个合适",
    r"\brecommend\b", r"\bsuggest\b", r"\bwhich one\b", r"\bwhat do you recommend\b",
)


def detect_intent(message: str, use_llm: bool = False) -> str:
    """统一的意图检测。

    - ``use_llm=False``（默认，销售 Agent 使用）：纯规则，命中返回
      ``"recommendation"``，否则返回空字符串。
    - ``use_llm=True``（方案 Agent 使用）：规则判定为推荐时直接返回；
      否则用 LLM 区分 product_question / conversation / others。
    """
    text = str(message or "").strip()
    if not text:
        return "" if not use_llm else "others"
    lowered = text.lower()

    # 明确要推荐 → 不管是不是疑问句，都按推荐处理
    if any(re.search(pattern, lowered) for pattern in _RECO_REQUEST_PATTERNS):
        return "recommendation"

    slots = extract_slots(text)
    has_signal = bool(slots) or any(
        re.search(pattern, lowered) for pattern in _RECOMMENDATION_PATTERNS
    )
    if has_signal:
        # 方案侧要排除"提问"：客户问"这个屏多久能发货？"时句子里带着"屏"，
        # 但那是提问、不是要推荐，交给 LLM 判成 others / product_question。
        if not use_llm or not looks_like_question(text):
            return "recommendation"
    if not use_llm:
        return ""

    prompt = (
        "Classify the user message into exactly one intent:\n"
        "- recommendation: describing needs or asking for a product recommendation\n"
        "- product_question: asking about specs/features/capabilities/model differences\n"
        "- conversation: greetings and casual chat\n"
        "- others: company info, warranty, business process, anything else\n\n"
        f"User message: {text}\n\nReturn the intent type only."
    )
    try:
        response = get_llm(temperature=0).invoke(prompt)
        intent = str(getattr(response, "content", response)).strip().lower()
        return intent if intent in _VALID_INTENTS else "others"
    except Exception as exc:
        logger.warning("detect_intent LLM fallback failed: %s", exc)
        return "others"


# ── LangGraph 节点 ──────────────────────────────────────────────────────────
def parameter_inference_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """事实 → 技术参数（纯 Python，不调用 LLM）。"""
    requirement = dict(state.get("requirement", {}) or {})
    messages = state.get("messages", []) or []

    last_user = ""
    for msg in reversed(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", "")
        if role in ("user", "human"):
            last_user = (
                msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            )
            break
    if not last_user:
        # 与 retrieval 节点保持一致：messages 里没有用户消息时回退到 current_message
        last_user = str(state.get("current_message") or "")

    facts = dict(extract_slots(last_user))
    facts.update({k: v for k, v in requirement.items() if v not in (None, "", [], {})})
    facts["_raw_message"] = last_user

    display_type = state.get("display_type", "BOTH")
    if display_type == "IFP":
        technical: Dict[str, Any] = {
            "environment": environment_from_facts(facts),
            "viewing_distance_m": facts.get("viewing_distance_m"),
            "pixel_pitch_min_mm": None,
            "pixel_pitch_max_mm": None,
            "brightness_min_nit": None,
            "brightness_max_nit": None,
            "is_rental": None,
            "screen_size": screen_size_for_distance(facts.get("viewing_distance_m")),
            "source": {"display_type": "ifp_skip_engineering_inference"},
        }
    else:
        technical = infer_technical_parameters(facts)

    logger.info(
        "ParameterInference: pitch=%s~%s brightness=%s~%s rental=%s source=%s",
        technical["pixel_pitch_min_mm"], technical["pixel_pitch_max_mm"],
        technical["brightness_min_nit"], technical["brightness_max_nit"],
        technical["is_rental"], technical["source"],
    )

    return {
        **state,
        "inferred_brightness_min_nit": technical["brightness_min_nit"],
        "inferred_brightness_max_nit": technical["brightness_max_nit"],
        "inferred_pixel_pitch_min_mm": technical["pixel_pitch_min_mm"],
        "inferred_pixel_pitch_max_mm": technical["pixel_pitch_max_mm"],
        "inferred_is_rental": technical["is_rental"],
        "inferred_screen_size": technical["screen_size"],
        "technical_parameters": technical,
        "next_action": "retrieve" if display_type == "IFP" else "check_info",
    }


__all__ = [
    "ParameterInference",
    "BRIGHTNESS_BY_ENVIRONMENT",
    "VIEWING_DISTANCE_PITCH_TABLE",
    "brightness_range_for_environment",
    "detect_intent",
    "environment_from_facts",
    "extract_requirements",
    "extract_slots",
    "infer_rental_from_text",
    "infer_technical_parameters",
    "parameter_inference_node",
    "parse_distance",
    "pitch_range_for_distance",
    "screen_size_for_distance",
    "suggest_screen_size",
]
