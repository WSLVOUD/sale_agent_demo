"""
产品参数推理模块
将客户需求翻译为具体技术参数
"""
from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict, Optional

from src.core.llm import get_llm

logger = logging.getLogger(__name__)

_RENTAL_KEYWORDS = (
    "租赁", "短租", "活动", "演唱会", "舞台", "演出", "展览", "车展", "快闪",
    "比赛", "体育", "赛事", "巡演", "临时",
    "rental", "event", "concert", "stage", "tour", "show", "exhibition",
)

_FIXED_KEYWORDS = (
    "会议室", "教室", "展厅", "商场", "店铺", "广告", "标牌", "橱窗",
    "指挥中心", "监控", "控制室", "广播", "电视台", "固定安装", "挂墙",
    "conference room", "meeting room", "classroom", "office", "retail",
    "store", "shop", "window", "fixed install", "billboard", "signage",
)

INFERENCE_PROMPT = """你是一名 LED 显示产品工程师，正在为客户的需求预生成一套技术参数。

客户原始需求：
{requirement}

对话历史：
{history}

请输出 JSON（仅 JSON，不要 markdown，不要解释）：
{{
  "brightness_min_nit": int 或 null,
  "brightness_max_nit": int 或 null,
  "pixel_pitch_min_mm": float 或 null,
  "pixel_pitch_max_mm": float 或 null,
  "is_rental": true / false / null,
  "rationale": "一句话解释"
}}

参数推断规则：
1. brightness_min_nit：全户外→4000以上，半户外→800，室内一般→500左右
2. pixel_pitch_min_mm/max_mm：距离≤2m→0.6~1.5，2-4m→0.9~2.0，4-8m→1.5~3.0，8-15m→2.5~5.0，15-30m→4.0~8.0，≥30m→6.0~10.0
3. is_rental：演唱会/舞台/演出→null，会议室/教室/固定安装→false
4. 不确定时返回 null

只输出 JSON。
"""


def _rule_infer_rental(requirement: Dict[str, Any], message: str) -> Optional[bool]:
    """Heuristic rental detection."""
    text = (message or "").lower()
    purpose = str(requirement.get("purpose", "")).lower()
    haystack = f"{text} {purpose}"
    
    if any(kw.lower() in haystack for kw in _RENTAL_KEYWORDS):
        return True
    if any(kw.lower() in haystack for kw in _FIXED_KEYWORDS):
        return False
    return None


def _rule_infer_pitch(distance: str) -> tuple:
    """Translate viewing distance into pitch range."""
    if not distance:
        return None, None
    match = re.search(r"(\d+(?:\.\d+)?)", str(distance))
    if not match:
        return None, None
    d = float(match.group(1))
    
    if d < 2: return 0.6, 1.5
    if d < 4: return 0.9, 2.0
    if d < 8: return 1.5, 3.0
    if d < 15: return 2.5, 5.0
    if d < 30: return 4.0, 8.0
    return 6.0, 10.0


def _rule_infer_brightness(requirement: Dict[str, Any], message: str) -> tuple:
    """Default brightness ranges from environment."""
    text = (message or "").lower()
    purpose = str(requirement.get("purpose", "")).lower()
    is_outdoor = requirement.get("outdoor") or any(kw in text for kw in ("户外", "室外", "outdoor"))
    is_semi_outdoor = any(kw in text for kw in ("半户外", "遮阳", "半室外"))
    
    if is_outdoor and not is_semi_outdoor:
        return 4000, None
    if is_semi_outdoor:
        return 800, None
    return 400, 1500


def _format_history(messages) -> str:
    """Build conversation history block."""
    if not messages:
        return "（无）"
    lines = []
    for msg in messages[-6:]:
        if isinstance(msg, dict):
            role = msg.get("role", "?")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "type", "?")
            content = getattr(msg, "content", "")
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) or "（无）"


def parameter_inference_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Translate requirement → brightness / pitch / rental constraints."""
    requirement = state.get("requirement", {}) or {}
    messages = state.get("messages", []) or []
    
    # Skip for IFP
    display_type = state.get("display_type", "BOTH")
    if display_type == "IFP":
        return {
            **state,
            "inferred_brightness_min_nit": None,
            "inferred_brightness_max_nit": None,
            "inferred_pixel_pitch_min_mm": None,
            "inferred_pixel_pitch_max_mm": None,
            "inferred_is_rental": None,
            "next_action": "retrieve",
        }
    
    last_user = ""
    for msg in reversed(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", "")
        if role in ("user", "human"):
            last_user = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            break
    
    # Rule-based fallback
    bmin, bmax = _rule_infer_brightness(requirement, last_user)
    pmin, pmax = _rule_infer_pitch(requirement.get("distance", ""))
    rental = _rule_infer_rental(requirement, last_user)
    
    # LLM refine
    try:
        req_text = ", ".join(f"{k}: {v}" for k, v in requirement.items() if v) or "未明确"
        prompt = INFERENCE_PROMPT.format(
            requirement=req_text,
            history=_format_history(messages),
        )
        response = get_llm(temperature=0.0).invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        
        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in content:
            content = content.split("```", 1)[1].split("```", 1)[0]
        
        parsed = json.loads(content.strip())
        
        if parsed.get("brightness_min_nit") is not None:
            bmin = int(parsed["brightness_min_nit"])
        if parsed.get("brightness_max_nit") is not None:
            bmax = int(parsed["brightness_max_nit"])
        if parsed.get("pixel_pitch_min_mm") is not None:
            pmin = float(parsed["pixel_pitch_min_mm"])
        if parsed.get("pixel_pitch_max_mm") is not None:
            pmax = float(parsed["pixel_pitch_max_mm"])
        if parsed.get("is_rental") is not None:
            rental = parsed["is_rental"]
        
        logger.info(f"ParameterInference: bmin={bmin} bmax={bmax} pmin={pmin} pmax={pmax} rental={rental}")
        
    except Exception as exc:
        logger.warning(f"ParameterInference LLM failed: {exc}")
    
    return {
        **state,
        "inferred_brightness_min_nit": bmin,
        "inferred_brightness_max_nit": bmax,
        "inferred_pixel_pitch_min_mm": pmin,
        "inferred_pixel_pitch_max_mm": pmax,
        "inferred_is_rental": rental,
        "next_action": "check_info",
    }


def detect_intent(message: str) -> str:
    """Detect if the message is about product recommendations/needs.
    
    Returns "recommendation" if the message appears to be describing needs
    or asking for product recommendations based on requirements.
    """
    import logging as _di_log
    _di_log.getLogger(__name__).warning("detect_intent CALLED with: %r", message)
    
    recommendation_patterns = [
        r"需要.*屏",
        r"需要.*屏幕",
        r"要.*屏",
        r"要.*屏幕",
        r"想买",
        r"采购",
        r"推荐",
        r"选.*屏",
        r"选.*屏幕",
        r"帮我选",
        r"用.*场景",
        r"用.*地方",
        r"安装.*地方",
        r"适合.*用",
        r"会议室",
        r"教室",
        r"培训",
        r"舞台",
        r"广告",
        r"租赁",
        r"演唱会",
        r"体育",
        r"户外",
        r"室内",
        r"屏幕",
        r"手写",
        r"触控",
        r"交互",
        r"白板",
    ]
    
    message_lower = message.lower()
    _di_log.getLogger(__name__).warning("detect_intent message_lower: %r", message_lower)
    for pattern in recommendation_patterns:
        if re.search(pattern, message_lower):
            return "recommendation"
    
    return ""


def extract_requirements(message: str, existing: Dict = None) -> Dict[str, Any]:
    """Extract requirements from a message using the LLM.
    
    Args:
        message: The user's message
        existing: Existing requirements to merge with
        
    Returns:
        Dict with 'requirement' key containing the extracted requirements
    """
    from src.core.llm import get_llm
    
    existing = existing or {}
    
    req_parts = [f"{k}: {v}" for k, v in existing.items() if v]
    existing_text = "\n".join(req_parts) if req_parts else "无"
    
    prompt = f"""从用户消息中提取 LED/LCD 显示产品需求：

已有需求：
{existing_text}

用户消息：{message}

请以 JSON 格式返回：
{{
    "requirement": {{
        "indoor": true/false/null,
        "outdoor": true/false/null,
        "distance": "可视距离描述",
        "purpose": "使用场景",
        "size": "尺寸要求",
        "brightness": "亮度要求",
        "display_type": "LED/LCD/BOTH/null"
    }}
}}

只返回 JSON，不要其他内容。"""
    
    try:
        llm = get_llm(temperature=0)
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        
        # Parse JSON
        import json
        import re
        # Extract JSON from response
        match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            return {"requirement": parsed.get("requirement", {})}
        else:
            return {"requirement": {}}
    except Exception as e:
        logger.warning(f"extract_requirements error: {e}")
        return {"requirement": {}}


class ParameterInference:
    """参数推断器（Phase 4 / Phase 11：独立类，便于复用 & 测试）。

    关键职责（Phase 8 规则工程化 + Phase 11 性能）：
        - 室内 / 室外 / 半户外 → outdoor / indoor 布尔
        - 屏幕类型 → display_type (LED / LCD / IFP)
        - 点间距 P 数字 → pixel_pitch
        - 租赁 / 固定 → is_rental
        - 防水 / COB / HDR / 拼接 → 各 flag
        - 业务上的 capability 表达式（如 "800nit"）→ brightness_min

    该类是 **Pure Rule** —— 不调用 LLM（性能 + 可测试）。
    所有规则从现有 router / requirement 中提取并集中。
    """

    # 室内场景关键词
    _INDOOR_PURPOSE = (
        "会议室", "会议", "教室", "培训", "教学", "学校", "课堂",
        "商场", "商店", "零售", "店铺", "超市",
        "展厅", "展馆", "展览", "博物馆",
        "医院", "诊所",
        "指挥中心", "监控中心", "中控室", "控制室",
        "机场", "车站", "码头", "地铁", "酒店", "大堂", "银行",
        "餐厅", "酒吧", "咖啡厅",
        "室内", "户内",
    )
    _OUTDOOR_PURPOSE = (
        "户外", "室外", "露天",
        "建筑外墙", "幕墙", "楼体",
        "体育场", "操场", "广场",
        "演唱会", "音乐会", "舞台", "演出", "表演", "剧场",
        "广告", "传媒",
    )
    _SEMI_OUTDOOR_PURPOSE = (
        "半户外", "半室外", "遮阳",
    )

    _DISPLAY_LED_KEYWORDS = ("led", "LED", "显示屏", "屏幕")
    _DISPLAY_LCD_KEYWORDS = ("lcd", "LCD", "拼接屏", "拼接")
    _DISPLAY_IFP_KEYWORDS = (
        "ifp", "会议一体机", "触摸一体机", "交互平板", "交互式平板",
        "电子白板", "interactive flat panel",
    )

    _DISPLAY_EXCLUDE_IFP = (
        "不需要手写", "不需要IFP", "不要IFP", "普通显示屏", "普通屏",
    )

    _RENTAL_KEYWORDS = (
        "租赁", "租用", "短租", "活动", "演唱会", "舞台", "演出",
        "展览", "车展", "快闪", "比赛", "体育", "赛事", "巡演", "临时",
        "rental", "event", "concert", "stage", "tour", "show", "exhibition",
    )
    _FIXED_KEYWORDS = (
        "会议室", "教室", "展厅", "商场", "店铺", "广告", "标牌",
        "橱窗", "指挥中心", "监控中心", "控制室", "固定安装", "挂墙",
        "固装", "永久", "fixed",
        "conference room", "meeting room", "classroom", "office",
        "retail", "store", "fixed install", "billboard",
    )

    def __init__(self):
        # 状态无关：纯函数式，所以不需要任何实例字段
        # 保留 __init__ 是为了让测试桩能 mock
        pass

    # ── 公开 API ────────────────────────────────────────────────
    def extract_constraints(self, message: str) -> Dict[str, Any]:
        """从单条用户消息中提取结构化约束。

        返回字典，可直接传给 ``ProductFilter.apply(**result)``。
        """
        if not message:
            return {}
        text = str(message).strip()
        lower = text.lower()
        constraints: Dict[str, Any] = {}

        # 1) 室内 / 室外 / 半户外（按优先级）
        is_semi = any(kw in text for kw in self._SEMI_OUTDOOR_PURPOSE)
        # 注意：把"室内" / "户外" 这种强信号挑出来，演唱会虽然一般是户外活动，
        # 但用户原文里说"室内 + 演唱会"更可能是用错表达，应该信 indoor 关键词
        has_indoor_kw = "室内" in text or "户内" in text or "indoor" in lower
        has_outdoor_kw = (
            "户外" in text or "室外" in text or "outdoor" in lower
        )
        other_outdoor_scene = any(
            kw for kw in self._OUTDOOR_PURPOSE
            if kw in text and kw not in ("户外", "室外")
        ) or has_outdoor_kw

        if is_semi:
            constraints["semi_outdoor"] = True
            constraints["purpose"] = "半户外"
            constraints.setdefault("outdoor", True)
            constraints.setdefault("indoor", False)
        elif has_indoor_kw:
            # 室内显式表达优先（即使有演唱会等户外活动关键词）
            constraints["indoor"] = True
            constraints["outdoor"] = False
        elif other_outdoor_scene:
            constraints["outdoor"] = True
            constraints["indoor"] = False
        elif any(kw in text for kw in self._INDOOR_PURPOSE):
            constraints["indoor"] = True
            constraints["outdoor"] = False

        # 2) display_type
        if any(kw in text for kw in self._DISPLAY_LCD_KEYWORDS):
            constraints["display_type"] = "LCD"
        elif any(kw in text for kw in self._DISPLAY_IFP_KEYWORDS):
            constraints["display_type"] = "IFP"
        elif any(kw in text for kw in self._DISPLAY_LED_KEYWORDS):
            constraints["display_type"] = "LED"

        # 显式排除 IFP（"普通显示屏就行" 等表达）
        if any(neg in text for neg in self._DISPLAY_EXCLUDE_IFP):
            constraints["exclude_ifp"] = True

        # 3) pixel_pitch（"P2.5"、"点间距3mm"）
        pitch = re.search(r"[Pp](\d+(?:\.\d+)?)", text)
        if pitch:
            constraints["pixel_pitch"] = float(pitch.group(1))
            constraints["pixel_pitch_tolerance"] = 0.5
        else:
            mm_match = re.search(r"点间距\s*(\d+(?:\.\d+)?)\s*mm", text)
            if mm_match:
                constraints["pixel_pitch"] = float(mm_match.group(1))

        # 4) 亮度（"5000nit"、"亮度5000"）
        brightness = re.search(r"(\d{3,5})\s*nit", lower)
        if brightness:
            constraints["brightness_min"] = int(brightness.group(1))
        else:
            br = re.search(r"亮度\s*(\d{3,5})", text)
            if br:
                constraints["brightness_min"] = int(br.group(1))

        # 5) 尺寸（"65英寸"、"55寸"）
        size = re.search(r"(\d+)\s*(?:寸|英寸|inch)", lower)
        if size:
            constraints["size_inch"] = int(size.group(1))

        # 6) 租赁 / 固定安装
        if any(kw in lower for kw in self._RENTAL_KEYWORDS):
            constraints["is_rental"] = True
        elif any(kw in lower for kw in self._FIXED_KEYWORDS):
            constraints["is_rental"] = False

        # 7) 防水
        if "防水" in text or "waterproof" in lower or re.search(r"ip65", lower):
            constraints["waterproof"] = True

        # 8) HDR
        if "hdr" in lower:
            constraints["hdr"] = True

        # 9) COB
        if "cob" in lower:
            constraints["cob"] = True

        # 10) 拼接（指挥中心 / 监控中心 偏好）
        if any(kw in text for kw in ("拼接", "拼接屏", "拼接墙")) or \
           any(kw in text for kw in ("指挥中心", "监控中心", "控制室", "中控室")):
            constraints["is_splicing"] = True
            if "display_type" not in constraints:
                constraints["display_type"] = "LCD"

        # 11) purpose（按优先级：户外 > 室内 > 半户外）
        for kw in self._OUTDOOR_PURPOSE:
            if kw in text:
                constraints.setdefault("purpose", kw)
                break
        if "purpose" not in constraints:
            for kw in self._INDOOR_PURPOSE:
                if kw in text:
                    constraints.setdefault("purpose", kw)
                    break

        return constraints

    # ── 兼容旧调用方式 ──────────────────────────────────────────
    def infer_brightness(
        self,
        constraint: Dict[str, Any],
        message: str = "",
    ) -> tuple:
        """对外暴露 _rule_infer_brightness，避免重复实现。"""
        return _rule_infer_brightness(constraint, message)

    def infer_pitch(self, distance: str) -> tuple:
        """对外暴露 _rule_infer_pitch。"""
        return _rule_infer_pitch(distance)

    def infer_rental(
        self,
        constraint: Dict[str, Any],
        message: str = "",
    ) -> Optional[bool]:
        """对外暴露 _rule_infer_rental。"""
        return _rule_infer_rental(constraint, message)


__all__ = [
    "ParameterInference",
    "extract_requirements",
    "parameter_inference_node",
    "detect_intent",
]
