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

# ── Phase 7：第二次追问的"降低门槛"问法 ─────────────────────────────────
# 客户第一次说"不知道"之后，第二次要给区间 / 二选一，让他更容易回答；
# 客户可以用"大概/大约/更远/更近/10 米以上"这种模糊说法回答。
EASIER_QUESTIONS: Dict[str, Dict[str, tuple[str, ...]]] = {
    "viewing_distance": {
        "en": (
            "That's okay — even a rough idea helps. Will viewers be fairly close to the screen, "
            "or more than about 10 metres away?",
            "No problem at all. Roughly speaking, is the audience within about 5 metres, "
            "5–10 metres, or further than 10 metres?",
            "Even an approximation is fine — closer than 5 m, around 5–10 m, or more than 10 m?",
        ),
        "zh": (
            "没关系，大概范围就行 —— 观众离屏幕是 5 米以内、5~10 米，还是 10 米以上？",
            "不清楚也没关系，给个大概：观众是坐得比较近，还是 10 米开外？",
        ),
    },
    "installation": {
        "en": (
            "That's fine — is this going to stay put permanently, or is it something you rent "
            "for events?",
            "If you are not sure, just tell me: long-term fixed install, or short-term rental?",
        ),
        "zh": (
            "不确定也没关系 —— 是长期固定安装，还是短期租赁使用？",
        ),
    },
    "size": {
        "en": (
            "No problem if you have not measured it — do you have a rough idea of the width, "
            "even approximately?",
            "Even an approximate width is helpful — is it closer to 3 m, 5 m, or wider?",
        ),
        "zh": (
            "没量过也没关系 —— 大概宽度是多少？3 米左右、5 米左右，还是更宽？",
        ),
    },
    "environment": {
        "en": (
            "That's okay — most installations are indoors. Will this one be indoors, or outside?",
        ),
        "zh": (
            "没关系 —— 大多数项目是室内，这一块装在室内还是室外？",
        ),
    },
    "purpose": {
        "en": (
            "No problem — even the general setting helps. Is it more like a meeting room / "
            "classroom, a retail space, or an advertising venue?",
        ),
        "zh": (
            "没关系，说个大概就行 —— 更像会议室/教室、门店零售，还是广告类场景？",
        ),
    },
}


def question_for(
    slot: Optional[str],
    language: str = "en",
    seed: int = 0,
    easier: bool = False,
) -> Optional[str]:
    """按槽位取一句问法；同一槽位的不同问法语义完全一致。

    ``seed`` 用于轮换：同一次会话里每问一次就换一种说法，
    保证"问的内容不变、措辞不重复"。

    ``easier=True``：第二次问同一个槽位时"降低回答门槛"（给区间 / 二选一），
    不允许机械重复第一遍的问法（Phase 7）。
    """
    if not slot:
        return None
    table = EASIER_QUESTIONS if easier else QUESTION_VARIANTS
    variants = (table.get(slot) or {}).get(language) or (table.get(slot) or {}).get("en") or ()
    if not variants and easier:
        # 没有专门的"降门槛"说法 → 退回普通问法
        variants = (QUESTION_VARIANTS.get(slot) or {}).get(language) or (
            QUESTION_VARIANTS.get(slot) or {}
        ).get("en") or ()
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
    # Phase 11：READY / CONTINUE_ASKING / DEGRADED_READY
    status: str = ""
    unknown_slots: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "gate": self.gate,
            "missing": list(self.missing),
            "reason": self.reason,
            "next_question": self.next_question,
            "status": self.status or ("READY" if self.ready else "CONTINUE_ASKING"),
            "unknown_slots": list(self.unknown_slots),
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


# 图片给出的尺寸只能用来"问客户确认"，不能直接当尺寸
_SIZE_SLOTS = {"size", "width", "height"}


def size_hint_sentence(profile: Any, language: str = "en") -> str:
    """把图片估计的尺寸变成一句提示（计划第十八阶段：只能当 size_hint）。"""
    hint = list(getattr(profile, "vision_size_hint_mm", None) or [])
    if len(hint) < 2:
        return ""
    try:
        width_m = float(hint[0]) / 1000
        height_m = float(hint[1]) / 1000
    except (TypeError, ValueError):
        return ""
    if width_m <= 0 or height_m <= 0:
        return ""
    if language == "zh":
        return f"图片上看大约是 {width_m:g} 米 × {height_m:g} 米。"
    return f"The image suggests roughly {width_m:g}m x {height_m:g}m."


def _with_size_hint(profile: Any, slot: Optional[str], language: str, question: Optional[str]) -> Optional[str]:
    """问尺寸时带上"图片估计值"，让客户只需要确认（不强加）。"""
    if not question or slot not in _SIZE_SLOTS:
        return question
    hint = size_hint_sentence(profile, language)
    if not hint:
        return question
    return f"{hint} {question}"


def _is_confirmed(profile: Any, field_name: str) -> bool:
    """该字段是否来自**客户明确表达**（而非规则/上下文推断）。

    v2.0 Phase 4 的核心防呆：Gate 只能被"客户说过的事实"打开，
    不能被"系统猜出来的事实"打开 —— 否则会出现
    "只知道室内外 + 场景就直接推荐"以及"用估算视距选错点间距"。

    M2 四态口径：explicit / confirmed / scenario_derived 都算"客户侧"；
    default（系统默认）与 inferred（算法估算）不算。

    《智谱视觉需求提取接入实施计划》补充：图片**明确可见**（vision_explicit）
    的环境 / 场景 / 安装方式 / 屏类型也算"已确定"，不必再问客户一遍；
    图片推测（vision_inferred）不算 —— 那只是猜测。
    """
    from src.models.requirement import CONFIRMED_SOURCES, VISION_TRUSTED_FIELDS

    source = (getattr(profile, "sources", None) or {}).get(field_name)
    if source in CONFIRMED_SOURCES:
        return True
    return source == "vision_explicit" and field_name in VISION_TRUSTED_FIELDS


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

    # 0) 冲突（Phase 18）：客户明确说的与推断互相矛盾时不允许推荐
    conflicts = list(getattr(profile, "conflicts", None) or [])
    if conflicts:
        missing = ["environment", "purpose", "installation", "viewing_distance"]
        # 《智谱视觉接入计划》第十一阶段：图片推断与客户说法冲突时，
        # 直接问**冲突的那一项**（例如图片像室内、客户说室外），而不是笼统问场景。
        conflict_slot = next(
            (slot for slot in (getattr(profile, "conflict_slots", None) or []) if slot),
            "purpose",
        )
        return GateDecision(
            ready=False, gate="recommendation", missing=missing,
            reason="需求存在冲突，需要澄清：" + ", ".join(conflicts),
            next_question=question_for(conflict_slot, language, variant_seed)
            or question_for("purpose", language, variant_seed),
            status="CONTINUE_ASKING",
            unknown_slots=[],
        )

    # 1) 客户直接点名型号 / 系列
    if getattr(profile, "model", None) or getattr(profile, "series_id", None):
        return GateDecision(
            ready=True, gate="recommendation",
            reason="客户已点名型号/系列",
            status="READY",
        )

    # 2) 客户明确给出技术规格（点间距 / 亮度）
    if getattr(profile, "pixel_pitch_mm", None) is not None or \
            getattr(profile, "brightness_min_nit", None) is not None:
        return GateDecision(
            ready=True, gate="recommendation",
            reason="客户已明确给出技术规格（点间距/亮度）",
            status="READY",
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

    # ── Phase 11/12：字段级 Unknown 容错 ────────────────────────────────
    # 已经问满两次（或客户明确跳过）仍然没有值的字段 → unknown，不再阻塞推荐。
    unknown_slots = [slot for slot in missing if profile.is_unknown(slot)]
    blocking = [slot for slot in missing if slot not in unknown_slots]

    if blocking:
        first = _first_missing(blocking)
        # Phase 7：同一个字段第二次提问时要降低回答门槛（给区间 / 二选一）
        easier = profile.ask_count(first) >= 1
        question = (
            _size_axis_question(profile, language, variant_seed)
            if first == "size_axis"
            else None
        ) or question_for(first, language, variant_seed, easier=easier)
        # 图片给过尺寸估计 → 问尺寸时带上，让客户只需确认
        question = _with_size_hint(profile, first, language, question)
        return GateDecision(
            ready=False, gate="recommendation", missing=blocking,
            reason="信息不足以做可靠选型：" + ", ".join(blocking),
            next_question=question, status="CONTINUE_ASKING",
            unknown_slots=unknown_slots,
        )

    if unknown_slots:
        # 核心信息（场景 / 环境）至少要有其一，否则没有选型依据
        has_basis = bool(getattr(profile, "purpose", None)) or bool(
            getattr(profile, "environment", None)
        )
        if has_basis:
            return GateDecision(
                ready=True, gate="recommendation",
                reason="按已确认信息做 Best-effort 推荐（部分字段客户不知道："
                       + ", ".join(unknown_slots) + "）",
                status="DEGRADED_READY", unknown_slots=unknown_slots,
            )
        return GateDecision(
            ready=False, gate="recommendation", missing=unknown_slots,
            reason="核心信息（场景/室内外）客户也未确认，暂时没有选型依据",
            next_question=question_for("purpose", language, variant_seed, easier=True),
            status="CONTINUE_ASKING", unknown_slots=unknown_slots,
        )

    return GateDecision(
        ready=True, gate="recommendation",
        reason="环境 + 场景 + 安装方式 + 观看距离齐备",
        status="READY",
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
        question = _with_size_hint(
            profile, slot, language, question_for(slot, language, variant_seed)
        )
        return GateDecision(
            ready=False, gate="calculation", missing=missing,
            reason="缺少屏体尺寸，先推荐产品、暂不做箱体/模组计算",
            next_question=question,
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
    "size_hint_sentence",
]
