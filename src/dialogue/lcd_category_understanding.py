"""结合语境理解客户的 LCD 回合（品类 + 需求事实，不许由关键词触发）。

客户口径原文（2026-09-30）：

    「不允许有关键词来触发 AI 的某个链路，需要让 AI 结合语境上下文，
      去选择品类，然后根据对应品类推进需求询问，然后锁定屏幕。」
    「不允许用关键词去触发，以后的所有情况都是，只允许是结合上下文语境去整理
      需要的信息，关键词只能是兜底。」

所以这里**不做关键词匹配**。模型读三样东西再下结论：

    1. 我们上一轮问客户的那句（例如 "What will the screens be used for?"）
    2. 最近的对话（越靠下越新；客户已经答过什么，一目了然）
    3. 客户最新一句

输出是结构化信号，两段：

    category / confidence / scene / reason      —— 场景品类（走哪条需求链）
    facts                                       —— 这一轮客户给出的需求事实
        environment        indoor / outdoor
        installation       fixed / rental
        is_splicing        是否拼接
        splicing_layout    "3x3" 这种排布
        screen_size_inch   英寸（数字）
        bezel_mm           拼缝（毫米）
        resolution         2K / 4K
        touch / handwriting / camera / ops / tender

关键口径（为什么不能只看最新一句）：

    · 客户答 "65'" / "indoor" 这类**参数**时，品类应该**沿用对话里已经定下的**，
      不能因为他这一句没提场景词就退回 unknown，否则又会出现"已经知道品类还在问用途"。
    · 客户只是回答我们上一轮问的那一项时（问"几寸"答 "65"）要按那一项理解。
    · 客户说"不用 / 没有"也是一个**有效答案**（false），要如实报出来。
    · 只报客户**说过**的，不许替客户推断（没提的一律留空，交给下游按业务默认走）。
    · 客户明确改口（"其实是监控室" / "改成广告机"）→ 以新口径为准。
    · 关键词解析（lcd_decision._CATEGORY_HINTS / extract_lcd_facts 的正则）只在
      模型不可用时兜底（降级路径）。
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

MONITORING = "monitoring"
ADVERTISING = "advertising"
NORMAL = "normal"
CONFERENCE_EDUCATION = "conference_education"
UNKNOWN = "unknown"

ALL_CATEGORIES = (
    MONITORING,
    ADVERTISING,
    NORMAL,
    CONFERENCE_EDUCATION,
    UNKNOWN,
)

_JUDGE_PROMPT = """你是商用显示（LCD）销售对话的**语境理解器**。

任务：读完语境后，做两件事 ——

  A. 判断客户这块 LCD 屏属于哪一类**使用场景**；
  B. 把客户在这段对话里**给出的需求事实**整理出来。

返回 JSON：
{
  "category": "monitoring" | "advertising" | "conference_education" | "normal" | "unknown",
  "confidence": 0.0~1.0,
  "scene": "客户原话里描述场景的短语，没有就留空",
  "reason": "一句话理由",
  "facts": {
    "environment": "indoor" | "outdoor" | "",
    "installation": "fixed" | "rental" | "",
    "is_splicing": true | false | null,
    "splicing_layout": "3x3" 这样的列x行，没有就留空,
    "screen_size_inch": 数字（英寸）或 null,
    "bezel_mm": 数字（毫米）或 null,
    "resolution": "2K" | "4K" | "",
    "touch": true | false | null,
    "handwriting": true | false | null,
    "camera": true | false | null,
    "ops": true | false | null,
    "tender": true | false | null
  },
  "evidence": {"字段名": "客户原话片段"}
}

facts 的取值口径：
- 只填**客户自己说过**的。客户没提的一律留空（"" / null），**绝对不要**替他推断
  （例如客户只说"会议室"，不能推断出一定要触控）。
- 客户只是**回答我们上一轮问的那一项**时，按那一项来理解：
  我们问"屏幕要几英寸"、客户答 "65"，就是 screen_size_inch=65（单位是英寸）；
  我们问"拼缝要多窄"、客户答 "0.88"，就是 bezel_mm=0.88。
- 客户说"不用 / 不需要 / 没有 / 不带"是一个**有效答案**，要如实填 false
  （例如"不用 OPS"→ ops=false），不要留空。
- 客户说"要 / 需要 / 带 / 有"填 true。
- "cam"、"camera"、"摄像头"、"视频会议镜头"这类说法都是 camera。
- resolution 只填客户明确点名的 4K / 2K；客户没提就留空。

四类场景的业务含义：
- monitoring（监控/指挥）：一块大画面看画面或看多路信号。典型：控制室、指挥中心、
  安防监控、值班室、调度中心、视频墙（拼接）。客户关心的是"看得全、能拼、拼缝窄、
  能 7x24 长时间开着"。
- advertising（广告/信息发布）：面向**路过的人**发布画面。典型：商场、门店、
  大堂、电梯口、橱窗、菜单牌、导视牌、信息发布、数字标牌、广告机。
- conference_education（会议/教育）：**一群人围着一块屏开会或上课**。典型：会议室、
  教室、培训室、报告厅。客户可能关心投屏、白板书写、触控、视频会议、OPS 电脑。
- normal（普通商用显示屏）：就是**单独一块普通显示屏**，既不拼接、也不做对外广告
  发布、也不开会教学。典型：门店里放一块屏、办公室看板、展厅里的一块显示器。
- unknown：语境里还看不出他要拿这块屏干什么。

判定规则（务必逐条遵守）：
1. **看客户真正要拿这块屏干什么**，不要因为出现了某个词就下结论。
   例如"会议室"三个字出现在别的话题里，不等于这次是会议场景。
2. 客户最新一句如果只是在**回答参数**（尺寸、室内/室外、拼缝、预算、确认），
   就**沿用对话里已经看出来的品类**，不要因为他这一句没提场景就改判 unknown。
3. 客户明确**改口**（"其实是监控室" / "我改成广告机了"）→ 以新口径为准。
4. 只有在**整段对话都看不出用途**时才返回 unknown；拿不准时给一个 confidence 较低的
   判断，而不是一律 unknown。
5. 不要替客户假设场景；不要引入对话里没有的信息。
6. facts 里没提的字段就留空 —— 下游会用业务默认值处理，不需要你补齐。
7. 只返回 JSON，不要任何解释或 markdown。"""

# 允许出现在 facts 里的字段（其余一律丢掉，防止模型自己加字段）
FACT_KEYS: tuple[str, ...] = (
    "environment",
    "installation",
    "is_splicing",
    "splicing_layout",
    "screen_size_inch",
    "bezel_mm",
    "resolution",
    "touch",
    "handwriting",
    "camera",
    "ops",
    "tender",
)

_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_LIMIT = 64


def clear_cache() -> None:
    """清空语境判断缓存（测试与需求重置时用）。"""
    _CACHE.clear()


def _cache_key(session_id: str, message: str) -> str:
    return f"{session_id}::{message}"


def _clean_facts(raw: Any) -> Dict[str, Any]:
    """模型给的 facts 只保留认得的字段，并且做类型/取值范围的把关。"""
    if not isinstance(raw, dict):
        return {}
    facts: Dict[str, Any] = {}
    for key in FACT_KEYS:
        value = raw.get(key)
        if value in (None, "", [], {}):
            continue
        if key in ("is_splicing", "touch", "handwriting", "camera", "ops", "tender"):
            if isinstance(value, bool):
                facts[key] = value
            elif str(value).strip().lower() in ("true", "yes", "1", "是", "要", "有"):
                facts[key] = True
            elif str(value).strip().lower() in ("false", "no", "0", "否", "不用", "不需要", "没有"):
                facts[key] = False
            continue
        if key == "screen_size_inch":
            try:
                number = float(str(value).replace(",", "."))
            except (TypeError, ValueError):
                continue
            if 10 <= number <= 500:
                facts[key] = number
            continue
        if key == "bezel_mm":
            try:
                number = float(str(value).replace(",", "."))
            except (TypeError, ValueError):
                continue
            if 0 <= number <= 100:
                facts[key] = number
            continue
        text = str(value).strip()
        if key == "environment":
            if text.lower() in ("indoor", "outdoor"):
                facts[key] = text.lower()
        elif key == "installation":
            if text.lower() in ("fixed", "rental"):
                facts[key] = text.lower()
        elif key == "resolution":
            upper = text.upper().replace(" ", "")
            if upper in ("2K", "1080P", "FHD"):
                facts[key] = "2K"
            elif upper in ("4K", "UHD", "2160P"):
                facts[key] = "4K"
        elif key == "splicing_layout":
            facts[key] = text[:12]
    return facts


def _parse(content: Any) -> Dict[str, Any]:
    """模型输出 → 结构化信号（不合法一律当"没判断"，交给调用方兜底）。"""
    raw = str(content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
    try:
        payload = json.loads(raw)
    except Exception:
        logger.warning("LCD category signal is not JSON: %r", raw[:120])
        return {}
    if not isinstance(payload, dict):
        return {}
    category = str(payload.get("category") or "").strip().lower()
    facts = _clean_facts(payload.get("facts"))
    if category not in ALL_CATEGORIES:
        # 品类判不出来，但客户这一轮的**需求事实**可能还是有效的 → 别一起丢掉
        if not facts:
            return {}
        category = UNKNOWN
    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return {
        "category": category,
        "confidence": confidence,
        "scene": str(payload.get("scene") or "")[:120],
        "reason": str(payload.get("reason") or "")[:200],
        "facts": _clean_facts(payload.get("facts")),
        "evidence": {
            str(k): str(v)[:120]
            for k, v in (payload.get("evidence") or {}).items()
            if isinstance(payload.get("evidence"), dict)
        }
        if isinstance(payload.get("evidence"), dict)
        else {},
    }


def understand_lcd_category(
    message: str,
    *,
    session_id: str = "",
    conversation: str = "",
    asked_question: str = "",
    profile: Optional[Any] = None,
    current_category: str = "",
) -> Dict[str, Any]:
    """结合语境理解这一轮 LCD 回合：品类（category）+ 客户给出的需求事实（facts）。

    Args:
        message: 客户最新一句
        session_id: 会话 id（缓存键 + 日志）
        conversation: 最近的对话文本（越靠下越新）
        asked_question: 我们上一轮问客户的那句话
        profile: 需求档案（有的话把已定品类一并作为语境）
        current_category: 已经确定的品类（档案里没有时由调用方传）

    Returns:
        {"category": ..., "confidence": ..., "scene": ..., "reason": ...,
         "facts": {...}, "evidence": {...}}；
        判断不了时返回 ``{}``（调用方会退回到关键词兜底）。
    """
    text = str(message or "").strip()
    if not text:
        return {}
    key = _cache_key(session_id, text)
    cached = _CACHE.get(key)
    if cached is not None:
        return dict(cached)

    context: list[str] = []
    if conversation:
        context.append(f"最近的对话（越靠下越新）：\n{conversation}")
    if asked_question:
        context.append(f"我们刚问客户的那句：{asked_question}")

    known = str(current_category or "").strip().lower()
    if not known and profile is not None:
        try:
            known = str(getattr(profile, "lcd_category", "") or "").strip().lower()
        except Exception:  # pragma: no cover - 防御式
            known = ""
    if known in (MONITORING, ADVERTISING, NORMAL, CONFERENCE_EDUCATION):
        context.append(
            f"对话里已经看出来的使用场景：{known}"
            "（客户没说改口就沿用这一条）"
        )
    context.append(f"客户最新一句：{text}")

    try:
        # 延迟导入：测试与运维都通过 src.core.llm.get_llm 打桩，这里必须是运行时取。
        from ..core.llm import get_llm

        llm = get_llm(temperature=0.0)
        response = llm.invoke(
            [SystemMessage(content=_JUDGE_PROMPT), HumanMessage(content="\n\n".join(context))]
        )
        signal = _parse(getattr(response, "content", response))
    except Exception as exc:
        logger.warning("[%s] LCD category understanding failed: %s", session_id, exc)
        return {}

    if signal:
        _CACHE[key] = dict(signal)
        while len(_CACHE) > _CACHE_LIMIT:
            _CACHE.pop(next(iter(_CACHE)))
    return dict(signal)


# 这一路现在既判品类也给事实，"回合理解"这个叫法更贴切；
# 旧名字保留，避免已有调用与测试被迫改名。
understand_lcd_turn = understand_lcd_category


__all__ = [
    "ADVERTISING",
    "ALL_CATEGORIES",
    "CONFERENCE_EDUCATION",
    "FACT_KEYS",
    "MONITORING",
    "NORMAL",
    "UNKNOWN",
    "clear_cache",
    "understand_lcd_category",
    "understand_lcd_turn",
]
