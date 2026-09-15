"""
v2.0 Phase 4 / Phase 9：两个 Gate 的判定逻辑。

核心原则（来自 v2.0 文档）：
    **"进入 Solution Agent" 与 "允许执行产品推荐" 必须彻底分开。**

Recommendation Ready Gate —— 判断"是否已经具备可靠选型条件"：
    - 客户直接点名型号 / 系列                  → 可以直接推荐
    - 客户明确给出技术规格（点间距 / 亮度）      → 可以直接推荐（v2.0 Case 4）
    - 室内外 + 使用场景 + （安装方式 或 观看距离）→ 可以直接推荐（v2.0 Case 3）
    - 其余情况                                  → 先问一个关键问题（Case 1 / Case 2）

Calculation Ready Gate —— 判断"是否具备屏体工程计算条件"：
    - 需要 屏体宽度 + 高度（v2.0 Phase 9）
    - 不满足时：**照常推荐产品，但不做 Cabinet / Module 计算**

两个 Gate 都是纯函数、无 LLM、可单独测试。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 缺失项 → 面向客户的英文提问片段
MISSING_LABELS: Dict[str, str] = {
    "display_type": "the display type (LED, LCD or IFP)",
    "environment": "whether it will be installed indoors or outdoors",
    "purpose": "the application scenario (meeting room, retail, advertising ...)",
    "installation": "whether it is a fixed installation or for rental/events",
    "viewing_distance": "roughly how far viewers will stand from the screen",
    "width": "the target screen width",
    "height": "the target screen height",
    "size_axis": "whether that measurement is the width, the height or the diagonal",
}

# 缺失项 → 默认提问顺序（先问最高价值的）
MISSING_ORDER: tuple[str, ...] = (
    # 客户刚报了一个裸尺寸（"129,2cm"）时，先确认它是宽 / 高 / 对角线，
    # 这是"回应客户刚说的话"，比继续问环境更该先问
    "size_axis",
    "environment",
    "purpose",
    "installation",
    "viewing_distance",
    "display_type",
    "width",
    "height",
)

# ── 提问话术：同一件事的多种自然说法 ─────────────────────────────────────────
# 每个槽位给出若干**语义完全相同**的问法，按会话轮次轮换使用，
# 避免每次都用同一句固定话术；但"问什么"始终不变（只问这一个槽位）。
QUESTION_VARIANTS: Dict[str, Dict[str, tuple[str, ...]]] = {
    "environment": {
        "en": (
            "Will the screen be installed indoors or outdoors?",
            "Is this for indoor or outdoor use?",
            "Just so I match the right models — will it be indoors or outdoors?",
            "Should I look at indoor or outdoor displays for you?",
            "Is the installation going to be indoors or outdoors?",
            "Will it be an indoor or outdoor setup?",
            "Are we talking about an indoor or an outdoor install?",
        ),
        "zh": (
            "这块屏是装在室内还是室外？",
            "是室内用还是室外用？",
            "方便确认下，装室内还是室外？",
            "这块屏放在室内还是室外？",
            "使用环境是室内还是户外？",
        ),
    },
    "purpose": {
        "en": (
            "What will the screen mainly be used for?",
            "What kind of application is this — meeting room, retail, advertising, or something else?",
            "Where will it be used, for example a meeting room, a store, or a venue?",
            "Could you tell me the main use case?",
            "What's the screen for, and where will it be used?",
            "What sort of venue will this be used in?",
            "Which application is this for — meeting room, classroom, retail, or advertising?",
        ),
        "zh": (
            "主要用在什么场景？",
            "这块屏主要用在哪里，比如会议室、门店还是广告位？",
            "方便说下主要的使用场景吗？",
            "这块屏主要用在什么场合？",
            "使用场景是哪一类，会议室、教室、门店还是广告？",
        ),
    },
    "installation": {
        "en": (
            "Is it a permanent install, or is it for rental/events?",
            "Will the screen stay fixed on site, or is it a rental?",
            "Just so I quote the right setup — is this a fixed install or a rental?",
            "Fixed installation or rental — which one is it for you?",
            "Is this a long-term installation, or do you need it for rental/events?",
            "Should I plan this as a permanent install or a rental?",
            "Quick one — fixed install or rental?",
            "Permanent install or rental — which one fits your project?",
        ),
        "zh": (
            "是固定安装，还是租赁/活动用？",
            "这块屏是固装还是租赁？",
            "安装方式是长期固定，还是临时租赁？",
            "这块屏是固定在现场，还是要租用/活动用的？",
            "简单确认下——固定安装还是租赁？",
            "这个是长期固定的项目，还是租赁/活动用的？",
        ),
    },
    "viewing_distance": {
        "en": (
            "Roughly how far will viewers be from the screen?",
            "What is the typical viewing distance?",
            "How far away will the audience usually be?",
            "About how many metres away will people be sitting?",
            "How far will the audience typically be sitting from the screen?",
            "What's the closest viewing distance I should design for?",
            "Where is the main viewing position — how many metres away?",
        ),
        "zh": (
            "观众通常离屏幕大概多远？",
            "大概的观看距离是多少？",
            "人一般坐得离屏幕多远？",
            "观众席离屏幕大概多少米？",
            "最近的一排观众离屏幕多远？",
        ),
    },
    "display_type": {
        "en": (
            "Do you need an LED, an LCD, or an interactive flat panel?",
            "Which display type do you have in mind — LED, LCD, or IFP?",
            "Should I look at LED, LCD, or an interactive panel?",
            "Are you leaning towards LED, LCD, or an interactive panel?",
        ),
        "zh": (
            "需要哪种类型的屏？LED、LCD，还是交互平板？",
            "屏的类型有偏好吗，LED、LCD 还是交互平板？",
            "您想了解的是 LED、LCD 还是交互平板？",
        ),
    },
    "width": {
        "en": (
            "What screen width are you aiming for?",
            "How wide should the screen be?",
            "Do you have a target width for the display?",
            "What width should I plan for the screen?",
        ),
        "zh": (
            "目标屏幕宽度大概多少？",
            "屏需要多宽？",
            "这块屏大概要做多宽？",
        ),
    },
    "size": {
        "en": (
            "What screen size do you have in mind (width x height)?",
            "Do you already know the target width and height?",
            "What width and height should the screen be?",
            "Could you share the screen dimensions — width and height?",
            "What width and height are you planning for the screen?",
            "Do you have the screen dimensions, in width x height?",
        ),
        "zh": (
            "目标屏幕尺寸大概多少（宽 x 高）？",
            "有具体的宽高吗？",
            "屏大概要多宽多高？",
            "这块屏计划做多大（宽 x 高）？",
        ),
    },
    "height": {
        "en": (
            "What screen height do you need?",
            "How tall should the screen be?",
            "Do you have a target height for the display?",
            "What height should I plan for the screen?",
        ),
        "zh": (
            "目标屏幕高度大概多少？",
            "屏需要多高？",
            "这块屏大概要做多高？",
        ),
    },
    # 客户只报了一个长度（"129,2cm"）→ 先确认这是宽 / 高 / 对角线（{value} 由 Gate 填入）
    "size_axis": {
        "en": (
            "Just so I use it correctly — is that {value} the width, the height, or the diagonal?",
            "Quick check: is the {value} you mentioned the width, the height, or the diagonal?",
            "So I plan this properly — is {value} the screen width, height, or diagonal?",
            "Is the {value} the width of the screen, its height, or the diagonal?",
        ),
        "zh": (
            "确认一下，{value} 指的是屏幕的宽度、高度还是对角线？",
            "这个 {value} 是屏宽、屏高，还是对角线尺寸？",
            "为了排箱体，{value} 是宽、高还是对角线？",
        ),
    },
}


def question_for(slot: Optional[str], language: str = "en", seed: int = 0) -> Optional[str]:
    """按槽位取一句问法；同一槽位的不同问法语义完全一致。

    ``seed`` 用于轮换：同一次会话里每问一次就换一种说法，
    保证"问的内容不变、措辞不重复"。
    """
    if not slot:
        return None
    variants = (QUESTION_VARIANTS.get(slot) or {}).get(language) or ()
    if not variants:
        variants = (QUESTION_VARIANTS.get(slot) or {}).get("en") or ()
    if not variants:
        label = MISSING_LABELS.get(slot, slot)
        return f"Could you tell me {label}?"
    return variants[seed % len(variants)]


@dataclass
class GateDecision:
    """Gate 判定结果。"""

    ready: bool
    gate: str                                   # "recommendation" | "calculation"
    missing: List[str] = field(default_factory=list)
    reason: str = ""
    next_question: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "gate": self.gate,
            "missing": list(self.missing),
            "reason": self.reason,
            "next_question": self.next_question,
        }


def _first_missing(missing: List[str]) -> Optional[str]:
    for slot in MISSING_ORDER:
        if slot in missing:
            return slot
    return missing[0] if missing else None


def first_missing_slot(missing: Any) -> Optional[str]:
    """对外暴露"按优先级取第一个缺失槽位"，供追问侧标记 pending_slot。"""
    return _first_missing([str(item) for item in (missing or [])])


def _question_for(slot: Optional[str], seed: int = 0) -> Optional[str]:
    return question_for(slot, seed=seed)


def format_measurement(mm: Optional[float]) -> str:
    """把毫米线索格式化成人话（1292 → 129.2 cm）。"""
    try:
        value = float(mm or 0)
    except (TypeError, ValueError):
        return ""
    if value <= 0:
        return ""
    if value >= 100 and abs(value % 10) < 1e-6:
        return f"{value / 10:g} cm"
    if value >= 100:
        return f"{value / 10:.1f} cm".replace(".0 cm", " cm")
    return f"{value:g} mm"


def _size_axis_question(profile: Any, language: str, seed: int) -> Optional[str]:
    """裸尺寸的方向确认问句（把客户给的数字填进模板）。"""
    question = question_for("size_axis", language, seed)
    if not question:
        return None
    value = format_measurement(getattr(profile, "screen_size_hint_mm", None))
    if not value:
        return None
    return question.replace("{value}", value)


def _is_confirmed(profile: Any, field_name: str) -> bool:
    """该字段是否来自**客户明确表达**（而非规则/上下文推断）。

    v2.0 Phase 4 的核心防呆：Gate 只能被"客户说过的事实"打开，
    不能被"系统猜出来的事实"打开 —— 否则会出现
    "只知道室内外 + 场景就直接推荐"以及"用估算视距选错点间距"。

    M2 四态口径：explicit / confirmed / scenario_derived 都算"客户侧"；
    default（系统默认）与 inferred（算法估算）不算。
    """
    from src.models.requirement import CONFIRMED_SOURCES

    source = (getattr(profile, "sources", None) or {}).get(field_name)
    return source in CONFIRMED_SOURCES


def _environment_settled(profile: Any) -> bool:
    """使用环境是否已经"确定"，不必再问客户。

      - 客户明说过室内/室外            → 确定
      - 场景本身就决定室内外（会议室 / 教室 / 教堂 / 户外广告 / 体育场…）→ 确定
        （客户反馈：说了 church 还问"室内还是室外"很傻）
      - 舞台 / 演唱会 / 租赁这类室内外都可能 → 不确定，继续问

    注意：用来推断环境的那个场景本身也必须是**客户说过的**（confirmed）；
    整套参数都是系统猜出来的时不在此列。
    """
    if not getattr(profile, "environment", None):
        return False
    if _is_confirmed(profile, "environment"):
        return True
    if not _is_confirmed(profile, "purpose"):
        return False
    try:
        from src.rag.query_understanding import environment_from_purpose

        return environment_from_purpose(getattr(profile, "purpose", None)) is not None
    except Exception:  # pragma: no cover - 防御式
        return False


def check_recommendation_ready(
    profile: Any,
    variant_seed: int = 0,
    language: str = "en",
) -> GateDecision:
    """Recommendation Ready Gate。

    对应 v2.0 第九章的四个 Case：
        Case 1  "I need an LED display"                        → False
        Case 2  "indoor LED display for a conference room"     → False
        Case 3  indoor + conference + fixed + 5m               → True
        Case 4  indoor + fixed + P2.5 + 600nit                 → True
    """
    if profile is None:
        missing = ["environment", "purpose", "installation", "viewing_distance"]
        return GateDecision(
            ready=False, gate="recommendation", missing=missing,
            reason="尚未建立需求档案",
            next_question=question_for(_first_missing(missing), language, variant_seed),
        )

    # 1) 客户直接点名型号 / 系列
    if getattr(profile, "model", None) or getattr(profile, "series_id", None):
        return GateDecision(
            ready=True, gate="recommendation",
            reason="客户已点名型号/系列",
        )

    # 2) 客户明确给出技术规格（点间距 / 亮度）
    if getattr(profile, "pixel_pitch_mm", None) is not None or \
            getattr(profile, "brightness_min_nit", None) is not None:
        return GateDecision(
            ready=True, gate="recommendation",
            reason="客户已明确给出技术规格（点间距/亮度）",
        )

    # 3) 场景充分：室内外 + 场景 + 安装方式 + 观看距离
    #    （v2.0 Phase 4「推荐所需的核心信息」= Display Type / Environment /
    #      Purpose / Installation / Viewing Distance；多轮示例也是在第四轮
    #      补齐观看距离后才推荐）
    #
    #    其中 installation 与 viewing_distance **必须是客户明确给出的**：
    #    - installation 若无客户说明，会被"场景默认固装"猜出来
    #    - viewing_distance 若无客户说明，旧逻辑会从"人数/面积"估算出来
    #    这两种推断值只能用于打分/检索提示，不能作为推荐依据。
    missing: List[str] = []
    # 客户报了一个裸尺寸（"129,2cm"）但没说方向 → 先确认，绝不替他猜
    if getattr(profile, "screen_size_hint_mm", None) and not getattr(profile, "has_target_size", False):
        missing.append("size_axis")
    if not getattr(profile, "environment", None):
        missing.append("environment")
    elif not _environment_settled(profile):
        # 环境要么客户明说，要么场景本身就能确定（会议室/教堂/户外广告…）；
        # 舞台 / 演唱会 / 租赁这类室内外都可能，仍需追问
        missing.append("environment")
    if not getattr(profile, "purpose", None):
        missing.append("purpose")
    if not getattr(profile, "installation", None):
        missing.append("installation")
    elif not _is_confirmed(profile, "installation"):
        missing.append("installation")
    if getattr(profile, "viewing_distance_m", None) is None:
        missing.append("viewing_distance")
    elif not _is_confirmed(profile, "viewing_distance_m"):
        missing.append("viewing_distance")

    if missing:
        first = _first_missing(missing)
        question = (
            _size_axis_question(profile, language, variant_seed)
            if first == "size_axis"
            else None
        ) or question_for(first, language, variant_seed)
        return GateDecision(
            ready=False, gate="recommendation", missing=missing,
            reason="信息不足以做可靠选型：" + ", ".join(missing),
            next_question=question,
        )

    return GateDecision(
        ready=True, gate="recommendation",
        reason="环境 + 场景 + 安装方式 + 观看距离齐备",
    )


def check_calculation_ready(
    profile: Any,
    variant_seed: int = 0,
    language: str = "en",
) -> GateDecision:
    """Calculation Ready Gate：屏体宽高齐备才做工程计算。"""
    if profile is None:
        return GateDecision(
            ready=False, gate="calculation", missing=["width", "height"],
            reason="尚未建立需求档案",
            next_question=question_for("width", language, variant_seed),
        )

    missing: List[str] = []
    if getattr(profile, "target_width_m", None) is None:
        missing.append("width")
    if getattr(profile, "target_height_m", None) is None:
        missing.append("height")

    if missing:
        # 宽高都缺时一次性问"整块尺寸"，避免只问宽度、下一轮又追问高度
        slot = "size" if len(missing) == 2 else missing[0]
        return GateDecision(
            ready=False, gate="calculation", missing=missing,
            reason="缺少屏体尺寸，先推荐产品、暂不做箱体/模组计算",
            next_question=question_for(slot, language, variant_seed),
        )

    return GateDecision(
        ready=True, gate="calculation",
        reason=f"目标尺寸 {getattr(profile, 'target_width_m', None)}m x "
               f"{getattr(profile, 'target_height_m', None)}m 已具备",
    )


__all__ = [
    "GateDecision",
    "MISSING_LABELS",
    "MISSING_ORDER",
    "QUESTION_VARIANTS",
    "check_calculation_ready",
    "check_recommendation_ready",
    "first_missing_slot",
    "format_measurement",
    "question_for",
]
