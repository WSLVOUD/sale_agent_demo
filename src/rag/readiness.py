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
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 缺失项 → 面向客户的英文提问片段
MISSING_LABELS: Dict[str, str] = {
    "display_type": "the display type (LED, LCD or IFP)",
    "environment": "whether it will be installed indoors or outdoors",
    "purpose": "the application scenario",
    "content_type": "whether the screen will play video, show images, or both",
    "installation": "whether it is a fixed installation or for rental/events",
    "pixel_pitch": "the pixel pitch you have in mind (P2.5, P3, P5 ...)",
    "viewing_distance": "roughly how far viewers will stand from the screen",
    "price_preference": "whether the price or the quality matters more to you",
    "width": "the target screen width",
    "height": "the target screen height",
    "size_axis": "whether that measurement is the width, the height or the diagonal",
}

# 内部槽位名（历史遗留叫法）→ 人话。**任何**要发给客户的问句都必须先经过这里，
# 否则会出现实测日志里那种 "could you tell me: distance?" —— 把内部字段名抛给客户。
_HUMAN_SLOT_LABELS: Dict[str, str] = {
    "distance": "roughly how far viewers will stand from the screen",
    "distance_m": "roughly how far viewers will stand from the screen",
    "viewing_distance_m": "roughly how far viewers will stand from the screen",
    "size": "the screen size (width and height)",
    "target_size": "the screen size (width and height)",
    "screen_size": "the screen size (width and height)",
    "target_width": "the screen width",
    "target_height": "the screen height",
    "target_width_mm": "the screen width",
    "target_height_mm": "the screen height",
    "pitch": "the pixel pitch you have in mind (P2.5, P3, P5 ...)",
    "brightness": "the brightness you need",
    "installation_or_distance": "whether it will be a fixed installation or a rental",
}


def human_label(slot: Any) -> str:
    """槽位名 → 客户能看懂的说法（找不到时用通用问法，绝不回落到槽位名本身）。"""
    key = str(slot or "").strip()
    if not key:
        return ""
    for candidate in (key, key.lower()):
        if candidate in MISSING_LABELS:
            return MISSING_LABELS[candidate]
        if candidate in _HUMAN_SLOT_LABELS:
            return _HUMAN_SLOT_LABELS[candidate]
    normalized = key.lower().replace("-", "_")
    for candidate in (normalized, normalized.rstrip("_m"), normalized.replace("_", "")):
        if candidate in MISSING_LABELS:
            return MISSING_LABELS[candidate]
        if candidate in _HUMAN_SLOT_LABELS:
            return _HUMAN_SLOT_LABELS[candidate]
    return ""

# 缺失项 → 默认提问顺序（先问最高价值的）
MISSING_ORDER: tuple[str, ...] = (
    # 客户刚报了一个裸尺寸（"129,2cm"）时，先确认它是宽 / 高 / 对角线，
    # 这是"回应客户刚说的话"，比继续问环境更该先问
    "size_axis",
    # ── 硬性条件（客户口径）：室内外 → 固装/租赁 → P值 → 尺寸 ──────────
    "environment",
    "installation",
    "pixel_pitch",
    # 点间距与观看距离是一组（客户说不知道 P 值 → 紧接着问观看距离）
    "viewing_distance",
    "size",
    "width",
    "height",
    # ── 非硬性（只记录，不阻塞推荐）──────────────────────────────────
    "display_type",
    "purpose",
    "content_type",
    "price_preference",
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
            "What kind of application is this?",
            "Where will the screen be used?",
            "Could you tell me the main use case?",
            "What's the screen for, and where will it be used?",
            "What sort of application will this screen be used in?",
            "Which application is this for?",
        ),
        "zh": (
            "这块屏主要用来做什么？",
            "主要的使用场景是什么？",
            "方便说下主要的使用场景吗？",
            "这块屏主要用在什么场合？",
            "它是做什么用途的？",
        ),
    },
    # 问完场景紧接着问内容类型（视频 / 图片 / 两者都有）——只记录，不影响选型
    "content_type": {
        "en": (
            "Will the screen mainly play video, show images, or a mix of both?",
            "Is it mainly for video content, for images, or both?",
            "What will you mostly show on it — video, images, or a bit of both?",
            "Will you be playing video, displaying images, or doing both on this screen?",
        ),
        "zh": (
            "这块屏主要是放视频、放图片，还是两者都有？",
            "内容上主要是播放视频、显示图片，还是两种都有？",
            "平时主要放视频、放图片，还是两种都有？",
            "视频、图片，还是两者都有？",
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
    # 点间距：先问客户有没有指定 P 值（有就按客户的选型；没有就转问观看距离）
    "pixel_pitch": {
        "en": (
            "Do you have a pixel pitch in mind — for example P2.5, P3 or P5?",
            "Which pixel pitch are you aiming for (P2.5, P3, P5 …)?",
            "Is there a particular pixel pitch you need, or should I work it out from the viewing distance?",
            "Do you already know the pitch you want — P3, P4, P5 …?",
            "What pixel pitch do you have in mind: a finer one like P1.5–P2.5, or a wider one like P4–P5?",
        ),
        "zh": (
            "您对点间距有要求吗？比如 P2.5、P3 或 P5。",
            "您想用多大的点间距（P2.5、P3、P5…）？",
            "点间距上有指定吗？没有的话我可以根据观看距离帮您定。",
            "您有想好的 P 值吗？比如 P3、P4、P5。",
            "想要更细腻一点的（P1.5–P2.5），还是点间距大一点的（P4–P5）？",
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
    # 推荐前最后一问：最看重价格还是质量（不问数字，只问取向）
    "price_preference": {
        "en": (
            "Before I lock in a model — which matters more to you, price or quality?",
            "One quick question: do you care more about the price, or about the quality?",
            "Should I optimise for the best price, or for the best quality?",
            "What matters more for this project — keeping the price down, or getting the best quality?",
        ),
        "zh": (
            "推荐之前问一句：您更看重价格，还是质量？",
            "选型前确认一下：您更在意价格，还是更在意效果质量？",
            "您更希望我优先控制价格，还是优先保证质量？",
            "价格和质量，哪个对您更重要？",
        ),
    },
}

# ── 环境"确认"问法 ──────────────────────────────────────────────────────
# 客户只说了场景（教堂 / 会议室 / 商场 / 广告牌…）时，环境是**系统从场景推断**的，
# 不等于客户说过。这时主动跟客户确认一次，问法要带上"我在跟你核对"的语气，
# 而不是像什么都没听到一样重新问一遍。
CONFIRM_QUESTION_VARIANTS: Dict[str, Dict[str, tuple[str, ...]]] = {
    "environment": {
        "en": (
            "Just to double-check — will this be an indoor or an outdoor setup?",
            "Before I pick a model, let me confirm the setting: indoors or outdoors?",
            "Quick check on my side — is this going indoors, or outdoors?",
            "So I don't guess wrong — indoor installation, or outdoor?",
        ),
        "zh": (
            "先跟您核对一下 —— 这块屏是装在室内还是室外？",
            "选型前确认一下使用环境：室内还是室外？",
            "我这边先确认一句：是室内安装还是室外安装？",
            "别搞错了方向 —— 室内用还是室外用？",
        ),
    },
}


def environment_confirm_question(
    language: str = "en",
    variant_seed: int = 0,
):
    """场景推断出环境时用的"确认型"问法（意思与普通环境问题一致）。"""
    variants = (
        CONFIRM_QUESTION_VARIANTS["environment"].get(language)
        or CONFIRM_QUESTION_VARIANTS["environment"]["en"]
    )
    return variants[variant_seed % len(variants)]


# ── Phase 7：第二次追问的"降低门槛"问法 ─────────────────────────────────
# 客户第一次说"不知道"之后，第二次要给区间 / 二选一，让他更容易回答；
# 客户可以用"大概/大约/更远/更近/10 米以上"这种模糊说法回答。
EASIER_QUESTIONS: Dict[str, Dict[str, tuple[str, ...]]] = {
    # 第二次问安装方式：客户听不懂"fixed / rental"，就用大白话解释着问
    "installation": {
        "en": (
            "Let me put it another way — do you need it mounted on the wall for good, "
            "or should it be something you can put up and take down quickly and carry with you?",
            "No jargon then — is it fixed in place permanently, or a portable one you can "
            "assemble and pack away whenever you need to move it?",
            "Simply put: does it stay installed on site, or do you need to move it around "
            "from place to place?",
        ),
        "zh": (
            "我换个说法：你是需要固定在墙上长期用的，还是需要能快装快拆、随时可以带走的？",
            "不用专业词 —— 是装上去就不动了，还是要能快速拆装、随时搬走的？",
            "简单说：这块屏是固定装在现场，还是需要经常挪地方、随时带走的？",
        ),
    },
    # 第二次问内容类型：给三个选项（视频 / 图片 / 两者都有）
    "content_type": {
        "en": (
            "No problem if it's not decided yet — will it be video, images, or both?",
            "Just roughly: mostly video, mostly images, or a mix of both?",
            "Either way is fine — does the content lean towards video, images, or both?",
        ),
        "zh": (
            "还没定也没关系 —— 是视频、图片，还是两者都有？",
            "给个大概就行：以视频为主、以图片为主，还是两种都有？",
            "哪种都行 —— 内容是偏视频、偏图片，还是两者都有？",
        ),
    },
    # 第二次问价格/质量取向
    "price_preference": {
        "en": (
            "No problem if it's hard to choose — just tell me: keep the price down, "
            "or go for better quality?",
            "Either answer is fine — should I pick the best value, or the higher-quality option?",
        ),
        "zh": (
            "不好选也没关系 —— 是需要帮我控制价格，还是要更好的质量？",
            "哪种都行 —— 我按性价比来选，还是按质量优先来选？",
        ),
    },
    "pixel_pitch": {
        "en": (
            "No problem if you're not sure — would you prefer a finer image (P1.5–P2.5) "
            "or a wider pitch (P4–P5)?",
            "If you don't have a pitch in mind, just tell me: finer detail, or a wider pitch?",
        ),
        "zh": (
            "不确定也没关系 —— 您想要更细腻一些（P1.5–P2.5），还是点间距大一点（P4–P5）？",
            "没有具体 P 值也行 —— 更看重画面细腻度，还是点间距大一些的方案？",
        ),
    },
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
            "No jargon then — is it fixed in place for good, or a portable one you can put up "
            "and pack away whenever you need to move it?",
            "Let me put it another way — do you need it mounted on the wall for good, or "
            "something you can put up and take down quickly and carry with you?",
            "Simply put: does it stay installed on site, or do you need to move it from place to place?",
        ),
        "zh": (
            "我换个说法：你是需要固定在墙上长期用的，还是需要能快装快拆、随时可以带走的？",
            "不用专业词 —— 是装上去就不动了，还是要能快速拆装、随时搬走的？",
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
            # 客户口径：问场景时**不举例**，直接问问题（不要罗列会议室/教室/商场…）
            "No problem — even the general setting helps. What sort of application will it be used for?",
            "No problem at all — could you tell me roughly what the screen will be used for?",
        ),
        "zh": (
            "没关系，说个大概就行 —— 这块屏主要用来做什么？",
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
        label = human_label(slot)
        if not label:
            logger.warning(
                "question_for: 未知槽位 %r → 退回通用问句（绝不把内部槽位名抛给客户）", slot
            )
            return (
                "Could you tell me a bit more about your setup so I can narrow down "
                "the right model?"
            )
        return f"Could you tell me {label}?"
    return _strip_canned_preamble(variants[seed % len(variants)])


# ── 客户口径（2026-09-21）：问句库只当"意思种子"，不要自带模板铺垫 ──────────
# 实测：客户看到的 "Quick one, fixed install or rental?" 就是这里自带的铺垫
# （"Quick one —"）被清洗层把破折号换成逗号之后的产物。
_CANNED_PREAMBLE_RE = re.compile(
    r"^(?:"
    r"quick one|quick check|one quick question|one more quick one|"
    r"just so i [^—,.]{0,40}|so i can match[^—,.]{0,40}|so i can point[^—,.]{0,40}|"
    r"so i plan this properly|while we're at it|in the meantime|to narrow it down|"
    r"简单确认下|顺便问一下|另外|为了给您匹配"
    r")\s*[—–,:：\-]+\s*",
    re.IGNORECASE,
)


def _strip_canned_preamble(text: str) -> str:
    """去掉问句库里自带的固定铺垫，只留下"要问的意思"。"""
    value = str(text or "").strip()
    cleaned = _CANNED_PREAMBLE_RE.sub("", value).strip()
    if not cleaned:
        return value
    if cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned or value


# ── 问句"意图"（v2.7 修订：不再把整句模板丢给 LLM 照抄）────────────────────
# 每个槽位：LLM 要表达的意思（intent）+ 必须出现的关键词（校验用）
QUESTION_INTENTS: Dict[str, Dict[str, Any]] = {
    "environment": {
        "intent": "确认这块屏是室内用还是室外用（决定箱体与亮度）",
        "keywords": ("indoor", "outdoor", "inside", "outside", "室内", "室外", "户外"),
    },
    "installation": {
        "intent": "确认是长期固定安装，还是租赁 / 活动用",
        "keywords": ("install", "fixed", "rental", "rent", "安装", "固装", "租赁"),
    },
    "pixel_pitch": {
        "intent": "确认客户对点间距有没有指定（没有就说明可以按观看距离推荐）",
        "keywords": ("pitch", "p2", "p3", "p4", "p5", "点间距", "间距"),
    },
    "viewing_distance": {
        "intent": "了解观众 / 观看者通常离屏幕多远",
        "keywords": ("distance", "far", "away", "metre", "meter", "distance"),
    },
    "size": {
        "intent": "确认屏体尺寸（宽 x 高）",
        "keywords": ("size", "width", "height", "dimension", "尺寸", "宽", "高"),
    },
    "size_axis": {
        "intent": "确认客户给的尺寸里哪一个是宽、哪一个是高",
        "keywords": ("width", "height", "宽", "高"),
    },
    "purpose": {
        "intent": "了解这块屏主要用在什么场合 / 做什么",
        "keywords": ("use", "used", "application", "purpose", "场景", "用途", "做什么"),
    },
    "price_preference": {
        "intent": "了解客户更看重价格还是品质",
        "keywords": ("price", "quality", "budget", "价格", "品质", "质量", "预算"),
    },
    "content_type": {
        "intent": "了解主要放视频、图片还是两者都有",
        "keywords": ("video", "image", "picture", "content", "视频", "图片", "内容"),
    },
    "brightness": {
        "intent": "确认亮度要求（或按环境推荐）",
        "keywords": ("brightness", "nits", "亮度"),
    },
}


def question_intent(slot: Optional[str]) -> str:
    """这个槽位到底要问什么（给 LLM 的"意图"，不是成句模板）。"""
    key = str(slot or "")
    entry = QUESTION_INTENTS.get(key)
    if entry:
        return str(entry.get("intent") or "")
    return f"了解客户的 {human_label(key) or key}"


def question_keywords(slot: Optional[str]) -> tuple:
    """这个槽位的问句里**应该**出现的关键词（校验"问的是不是同一件事"）。"""
    entry = QUESTION_INTENTS.get(str(slot or ""))
    return tuple(entry.get("keywords") or ()) if entry else ()


@dataclass
class GateDecision:
    """Gate 判定结果。"""

    ready: bool
    gate: str                                   # "recommendation" | "calculation"
    missing: List[str] = field(default_factory=list)
    reason: str = ""
    next_question: Optional[str] = None
    # Phase 11 / v2.1：READY / CONTINUE_ASKING / DEGRADED_READY / BLOCKED
    status: str = ""
    unknown_slots: List[str] = field(default_factory=list)
    # ── v2.1：字段决策状态（计划第 20 节的 Gate 日志结构）──────────────
    # deferred_slots：不再追问、但也不阻塞推荐（只影响计算 / 降级）
    deferred_slots: List[str] = field(default_factory=list)
    # blocked_slots：真正无法继续的字段（无法推导 + 客户没授权 + 该 Action 必需）
    blocked_slots: List[str] = field(default_factory=list)
    # v2.1：客户授权 AI 决定尺寸时，按观看距离推导出的参考尺寸 [width_m, height_m]
    # （只用于本轮工程计算与话术参考，**不写进客户的确认事实**）
    derived_size_m: Optional[List[float]] = None
    # v2.2.4：本轮**实际问的是哪个槽位**（可能是插在硬性条件之间的软问题：
    # 场景 / 价位取向）。追问侧必须用它来记 pending_slot / last_asked_slot，
    # 否则"客户回答的是哪一个问题"会对不上（例如 bare "both" 落到错误的槽位）。
    next_slot: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "gate": self.gate,
            "missing": list(self.missing),
            "reason": self.reason,
            "next_question": self.next_question,
            "status": self.status or ("READY" if self.ready else "CONTINUE_ASKING"),
            "unknown_slots": list(self.unknown_slots),
            "deferred_slots": list(self.deferred_slots),
            "blocked_slots": list(self.blocked_slots),
            "derived_size_m": list(self.derived_size_m) if self.derived_size_m else None,
            "next_slot": self.next_slot,
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
    return environment_settled(profile)


def environment_settled(profile: Any) -> bool:
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
    # v2.5+ 修复：客户已经被问过这一项、而且档案里确实有值 —— 就不再重复问。
    # 实测 bug：客户答了"室内"，系统还追着问"室内还是室外"（因为那次回答落档时
    # 来源不够硬）。环境只可能来自"客户说的"或"场景判定的"（EnvironmentResolver
    # 不会给系统默认值），问过一次拿到值就该算数；客户真要改，后面说"户外的"
    # 会被 conflict 检测接住。
    try:
        if profile.ask_count("environment") >= 1:
            return True
    except Exception:  # pragma: no cover - 防御式
        pass
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

    # 0) 冲突（v2.3 §9：Conflict 独立状态）—— 冲突未解决前禁止推荐
    #    · 客户语义冲突（图片说室内、客户说室外…）：profile.conflicts
    #    · 工程冲突（屏比房间还大、室内却要 P10…）：engineering.conflicts
    from ..engineering import conflict_message, detect_engineering_conflicts

    stored_conflicts = list(getattr(profile, "conflicts", None) or [])
    engineering_conflicts = detect_engineering_conflicts(profile)
    if stored_conflicts or engineering_conflicts:
        # 图片推断与客户说法冲突时，直接问**冲突的那一项**，而不是笼统问场景。
        conflict_slot = next(
            (slot for slot in (getattr(profile, "conflict_slots", None) or []) if slot),
            engineering_conflicts[0].slot if engineering_conflicts else "purpose",
        )
        reasons = list(stored_conflicts) + [item.message for item in engineering_conflicts]
        return GateDecision(
            ready=False, gate="recommendation", missing=[conflict_slot],
            reason="需求存在冲突，需要澄清：" + ", ".join(reasons),
            next_question=conflict_message(profile)
            or question_for(conflict_slot, language, variant_seed)
            or question_for("purpose", language, variant_seed),
            status="CONFLICT",
            unknown_slots=[],
            blocked_slots=[conflict_slot],
        )

    # 1) 客户直接点名型号 / 系列
    if getattr(profile, "model", None) or getattr(profile, "series_id", None):
        return GateDecision(
            ready=True, gate="recommendation",
            reason="客户已点名型号/系列",
            status="READY",
        )

    # 2) 硬性条件（客户口径）：尺寸 + P值 + 室内外 + 固装/租赁
    #    这四项齐了就直接推荐，**不再问其他问题**（场景 / 内容类型 / 价格取向
    #    只记录，不阻塞）；缺哪一项就只问那一项。
    #
    #    这四项**不适用**"问满两次就跳过"的容错规则：客户一直在说无关的话导致
    #    没记录到，最后要推荐之前必须再问一次（否则推荐没有依据）。
    # ── v2.1 Phase 5：字段状态 + 字段策略 → Action（计划第 11 / 12 / 16 节）──
    #    字段缺失不再等于"整个流程停止"：只阻塞依赖它的 Action。
    from .field_policy import (
        ASK,
        ASK_EASIER,
        ASK_LATER,
        BLOCK,
        DEFER,
        DEFER_CALCULATION,
        DEGRADE,
        INFER,
        RECOMMENDATION_SLOTS,
        SKIP,
        USE,
        apply_cross_slot_rules,
        field_action,
        is_hard_condition,
        policy_for,
    )

    # 客户报了一个裸尺寸（"129,2cm"）但没说方向 → 先确认方向（绝不替他猜）
    if getattr(profile, "screen_size_hint_mm", None) and not getattr(profile, "has_target_size", False):
        question = (
            _size_axis_question(profile, language, variant_seed)
            or question_for("size_axis", language, variant_seed)
        )
        return GateDecision(
            ready=False, gate="recommendation", missing=["size_axis"],
            reason="客户报了一个尺寸但没说方向（宽 / 高 / 对角线）",
            next_question=question, status="CONTINUE_ASKING",
        )

    actions: Dict[str, str] = {
        slot: field_action(profile, slot) for slot in RECOMMENDATION_SLOTS
    }

    # 这几个字段必须来自"客户侧"（客户说的 / 明显场景 / 图片明确可见）：
    # 纯系统推断（场景默认固装、图片推测）不能当成已确认 —— 客户口径：
    # "说了教堂还问室内外"要避免，但"没问过固装还是租赁"也要避免。
    if actions.get("environment") == USE and not _environment_settled(profile):
        actions["environment"] = (
            ASK if not profile.is_exhausted("environment") else BLOCK
        )
    if actions.get("installation") == USE and not _is_confirmed(profile, "installation"):
        actions["installation"] = (
            ASK if not profile.is_exhausted("installation") else DEGRADE
        )
    # 场景（purpose）不阻塞推荐（计划第 16 节：强相关但非绝对阻塞），
    # 所以即使它是系统推断的也不在这里改问 —— 只参与打分。
    # 点间距 ↔ 观看距离的先后与依赖（客户口径：先问 P 值，P 值不知道才问观看距离；
    # 客户已经给了 P 值 / 授权 AI 决定 → 不再问观看距离）
    apply_cross_slot_rules(profile, actions)

    blocked = [slot for slot, action in actions.items() if action == BLOCK]
    deferred = [
        slot for slot, action in actions.items()
        if action in (DEFER, DEGRADE, DEFER_CALCULATION)
    ]
    askable = [
        (slot, action) for slot, action in actions.items()
        if action in (ASK, ASK_EASIER, ASK_LATER)
    ]
    askable.sort(key=lambda item: policy_for(item[0]).ask_priority)
    # v2.5+++（计划 §8 配套修复）：**上一轮刚问过的那一项排到最后** ——
    # 客户答非所问 / 只报了别的需求时，不能把同一个问题紧接着再问一遍
    # （实测 bug：客户回 "3*5"，系统又把"室内还是户外"问了一次）。
    # 客户口径不变：没答的那一项留到"其它问题问完"的硬性条件复问再问。
    _last_asked = str(getattr(profile, "last_asked_slot", "") or "")
    if _last_asked:
        _others = [item for item in askable if item[0] != _last_asked]
        _just_asked = [item for item in askable if item[0] == _last_asked]
        if _others:
            askable = _others + _just_asked
    # v2.2.5：客户说过"不知道"的字段（ASK_LATER）**不马上重复问** ——
    #   先把其它问题问完（immediate），最后才回头用降门槛的问法问一次（parked）。
    immediate = [(slot, action) for slot, action in askable if action != ASK_LATER]
    parked = [(slot, action) for slot, action in askable if action == ASK_LATER]
    # v2.2.4：硬性条件（室内外 / 固装租赁 / P值 / 尺寸）与软问题（场景 / 价位取向）
    #   · 只要还有硬性条件没问完 → 按优先级问（软问题插在中间，保留销售话术）；
    #   · 硬性条件都齐了 → 软问题一律不再问，直接推荐
    #     （客户口径：硬性条件齐了就推荐，不要再问别的）。
    hard_asks = [(slot, action) for slot, action in immediate if is_hard_condition(slot)]
    # `missing` 只列还要问的硬性条件：它表示"什么在拦住推荐"。
    # 软问题（场景 / 价位取向）不算拦住推荐；已延后 / 客户不提供的字段也不算
    # （它们放在 deferred_slots，追问侧不会再问）。
    blocking_missing = [slot for slot, _ in hard_asks] + [
        slot for slot, _ in parked if is_hard_condition(slot)
    ]
    # 还有硬性条件没解决（不管是"还没问过"还是"客户说过不知道、最后一轮再问"），
    # 才继续问；硬性条件全齐了就直接推荐，软问题不再问。
    pending_hard = bool(hard_asks) or any(is_hard_condition(slot) for slot, _ in parked)
    inferred = [slot for slot, action in actions.items() if action == INFER]
    unknown_slots = [slot for slot, _ in parked]

    # 结构化日志（计划第 20 节）：一眼看出"为什么问 / 为什么不问"
    logger.info(
        "[ActionPlanner] %s",
        {
            "actions": actions,
            "blocked": blocked,
            "deferred": deferred,
            "askable": [slot for slot, _ in askable],
            "infer": inferred,
        },
    )

    if blocked:
        slot = blocked[0]
        question = (
            policy_for(slot).blocked_message
            or question_for(slot, language, variant_seed, easier=True)
        )
        return GateDecision(
            ready=False, gate="recommendation",
            missing=blocked + [s for s, _ in hard_asks],
            reason="真正无法继续（无法推导 + 客户未授权 + 该 Action 必需）：" + ", ".join(blocked),
            next_question=question, status="BLOCKED",
            blocked_slots=blocked, deferred_slots=deferred,
            next_slot=slot,
        )

    if immediate and pending_hard:
        # 还有硬性条件要问 → 本轮按优先级问一个问题（可能是插在中间的软问题）
        slot, action = immediate[0]
        easier = action == ASK_EASIER or profile.ask_count(slot) >= 1
        question = (
            _size_axis_question(profile, language, variant_seed)
            if slot == "size_axis"
            else None
        ) or question_for(slot, language, variant_seed, easier=easier)
        question = _with_size_hint(profile, slot, language, question)
        return GateDecision(
            ready=False, gate="recommendation",
            # missing 只列出**还要问**的字段（延后 / 降级的字段放 deferred_slots，
            # 否则追问侧会照着 missing 把已经不再追问的字段又问一遍）
            missing=blocking_missing,
            reason="继续追问（其余字段已决策 / 延后）：" + ", ".join(
                [s for s, _ in hard_asks] + [slot]
            ),
            next_question=question, status="CONTINUE_ASKING",
            unknown_slots=unknown_slots, deferred_slots=deferred,
            next_slot=slot,
        )

    # 最后一轮只回头问**硬性条件**（软问题客户不知道就不问了：它们不影响推荐）
    parked_hard = [(slot, action) for slot, action in parked if is_hard_condition(slot)]
    if parked_hard:
        # 其它都问完了，只剩下客户说过"不知道"的硬性条件 → 按降门槛的问法再问一次
        # （客户口径：等到要推荐的时候，再问一次他说不知道的那一项）
        slot, _action = parked_hard[0]
        question = question_for(slot, language, variant_seed, easier=True)
        question = _with_size_hint(profile, slot, language, question)
        logger.info("[ActionPlanner] 最后一轮：对 %s 用降门槛问法再问一次", slot)
        return GateDecision(
            ready=False, gate="recommendation",
            missing=blocking_missing,
            reason="其它条件已问完，回头确认客户说过不知道的字段：" + slot,
            next_question=question, status="CONTINUE_ASKING",
            unknown_slots=unknown_slots, deferred_slots=deferred,
            next_slot=slot,
        )

    # 无需再问 → 放行；有延后/降级字段时按 Best-effort 推荐
    if deferred:
        return GateDecision(
            ready=True, gate="recommendation",
            reason="可推荐；以下字段不再追问（只影响计算或按降级处理）：" + ", ".join(deferred),
            status="DEGRADED_READY", unknown_slots=deferred, deferred_slots=deferred,
        )
    return GateDecision(
        ready=True, gate="recommendation",
        reason="推荐必需信息齐备（其余字段已按客户决策处理）",
        status="READY",
    )


def check_calculation_ready(
    profile: Any,
    variant_seed: int = 0,
    language: str = "en",
) -> GateDecision:
    """Calculation Ready Gate（v2.1）：只有**依赖尺寸的计算**才被阻塞。

    - 宽高齐备 → READY
    - 客户授权 AI 决定尺寸（DELEGATED）→ 按观看距离给**参考尺寸**（确定性推导）→ READY
    - 尺寸问不出来 / 客户不说（DEFERRED / DECLINED）→ 挂起（DEFERRED）：
      照常推荐产品，只是这一轮不做箱体/模组计算（计划第 13 节）
    """
    if profile is None:
        return GateDecision(
            ready=False, gate="calculation", missing=["width", "height"],
            reason="尚未建立需求档案",
            next_question=question_for("width", language, variant_seed),
        )

    from .field_policy import (  # noqa: PLC0415
        ASK,
        ASK_EASIER,
        CALCULATION_SLOTS,
        field_action,
        plan_actions,
        policy_for,
    )

    # 1) 客户授权 AI 决定尺寸 → 用观看距离推导参考尺寸（Python 推导，不是 LLM 猜）
    delegated_size = any(
        profile.is_delegated(slot) for slot in ("size", "width", "height")
    )
    if delegated_size:
        from .parameter_inference import suggest_screen_size

        suggestion = suggest_screen_size(profile)
        if suggestion:
            width_m, height_m = suggestion
            decision = GateDecision(
                ready=True, gate="calculation",
                reason=(
                    "客户授权 AI 决定尺寸 → 按观看距离推导参考尺寸 "
                    f"{width_m}m x {height_m}m（仅供本轮计算参考）"
                ),
                status="READY",
            )
            decision.derived_size_m = [width_m, height_m]
            return decision
        # 客户授权了 AI 决定尺寸，但没有观看距离 → 推导不出来。
        # 这时候不要回头再问尺寸（客户已经授权了），而是问"推导需要的输入"：
        # 观看距离还能问就问观看距离，否则把计算挂起（推荐照常）。
        # 观看距离还能问（客户从没说过 / 说过不知道但还没到头）→ 问观看距离；
        # 客户已经授权 / 拒绝 / 延后观看距离 → 本轮只推荐产品，计算挂起
        distance_action = field_action(profile, "viewing_distance")
        if distance_action in (ASK, ASK_EASIER):
            question = question_for("viewing_distance", language, variant_seed)
            return GateDecision(
                ready=False, gate="calculation", missing=["viewing_distance"],
                reason="客户授权 AI 决定尺寸，但缺少观看距离（推导尺寸需要它）",
                next_question=question, status="CONTINUE_ASKING",
            )
        return GateDecision(
            ready=False, gate="calculation", missing=["width", "height"],
            reason="客户授权 AI 决定尺寸，但没有任何可用于推导的信息 → 本轮只推荐产品",
            next_question=None, status="DEFERRED",
            deferred_slots=["size"],
        )

    # 2) 宽高齐备 → 直接算
    missing: List[str] = []
    if getattr(profile, "target_width_m", None) is None:
        missing.append("width")
    if getattr(profile, "target_height_m", None) is None:
        missing.append("height")

    if not missing:
        return GateDecision(
            ready=True, gate="calculation",
            reason=f"目标尺寸 {getattr(profile, 'target_width_m', None)}m x "
                   f"{getattr(profile, 'target_height_m', None)}m 已具备",
        )

    # 3) 尺寸还能问 → 继续问（宽高都缺时一次问"整块尺寸"）
    actions = plan_actions(profile, CALCULATION_SLOTS)
    askable = [slot for slot, action in actions.items() if action in (ASK, ASK_EASIER)]
    deferred = [
        slot for slot, action in actions.items()
        if action not in (ASK, ASK_EASIER) and action != "use"
    ]
    if askable:
        # 宽高都缺时一次性问"整块尺寸"，避免只问宽度、下一轮又追问高度
        slot = "size" if len(missing) == 2 else missing[0]
        question = _with_size_hint(
            profile, slot, language, question_for(slot, language, variant_seed)
        )
        return GateDecision(
            ready=False, gate="calculation", missing=missing,
            reason="缺少屏体尺寸，先推荐产品、暂不做箱体/模组计算",
            next_question=question, status="CONTINUE_ASKING",
            deferred_slots=deferred,
        )

    # 4) 尺寸已经问不出来 / 客户不说 → 计算挂起（推荐照常）
    return GateDecision(
        ready=False, gate="calculation", missing=missing,
        reason="尺寸已延后（客户不知道 / 不提供）→ 本轮只推荐产品，不做箱体/模组计算",
        next_question=None, status="DEFERRED", deferred_slots=deferred or missing,
    )


__all__ = [
    "CONFIRM_QUESTION_VARIANTS",
    "GateDecision",
    "MISSING_LABELS",
    "MISSING_ORDER",
    "QUESTION_VARIANTS",
    "check_calculation_ready",
    "check_recommendation_ready",
    "environment_confirm_question",
    "environment_settled",
    "first_missing_slot",
    "format_measurement",
    "question_for",
    "size_hint_sentence",
]
