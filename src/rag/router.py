"""
查询复杂度路由模块。

三层业务路由（Phase 1 优化）：
  - FAST  ：纯参数查询（亮度多少、P2.5亮度是多少）→ 结构化过滤 + 模板
  - NORMAL：明确条件推荐（户外广告屏、会议室P2.5）   → 结构化过滤 + 轻量 RAG
  - AGENT ：需要推理的复杂推荐（什么屏好、帮我选）   → 完整 LangGraph

关键原则：
  有场景关键词（会议室、教室、户外）但无推理词 → NORMAL
  有场景关键词 + 强推理词（怎么选、哪个好、够用吗、有什么区别）→ AGENT
  纯参数词（亮度多少、P几）                   → FAST
  纯参数词 + 场景词                           → NORMAL（场景推荐优先）

修复历史（Phase 13 回归测试）：
  - 场景词 + 通用参数词（如"户外...P3...亮度要高"）不再误判为 FAST
  - 场景词 + 弱推理词（如"指挥中心用什么屏好"）应走 NORMAL 而非 AGENT
  - 纯 P-pitch + 通用疑问（如"适合什么场景"）走 FAST
  - 强比较/推理词（"够用吗"、"有什么区别"）→ AGENT
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class QueryRoute(Enum):
    """业务路由方向，与 phase 1 架构对齐。"""
    FAST   = "fast"    # 纯参数查询，不进 Agent
    NORMAL = "normal"  # 明确条件推荐，进 Agent Path 但不触发复杂推理
    AGENT  = "agent"   # 需要推理，进完整 Agent Path


@dataclass
class RoutingDecision:
    route: QueryRoute
    reason: str
    inferred_constraints: Optional[dict] = None
    # 以下字段保留兼容性
    complexity: str = ""   # "trivial" | "parameter" | "simple" | "complex"
    _legacy_route: str = ""  # "fast" | "agent"


# ── 场景关键词：标识"有明确推荐意图"的查询 ──────────────────────────────
# 有这些词的查询，不是纯参数查询，需要进 NORMAL 或 AGENT
_SCENE_KEYWORDS = frozenset({
    "会议室", "会议", "教室", "培训", "教学", "学校", "课堂",
    "商场", "商店", "零售", "店铺", "超市",
    "展厅", "展馆", "展览", "博物馆",
    "医院", "医疗", "诊所",
    "监控中心", "指挥中心", "中控室", "控制室", "监控",
    "机场", "车站", "码头", "地铁", "酒店", "大堂", "银行",
    "餐厅", "酒吧", "咖啡厅",
    "室内", "户内", "indoor",         # 室内场景（用于"室内固定安装"等 query）
    "户外", "室外", "露天", "建筑外墙", "幕墙", "楼体",
    "体育场", "操场", "广场",
    "演唱会", "音乐会", "舞台", "演出", "表演", "剧场",
    "广告", "传媒",
    "租赁", "临时", "活动",
    "半户外", "半室外", "遮阳",
})

# ── 推理关键词：需要 AGENT 处理的模糊/比较型查询 ──────────────────────────
# 强推理词：场景词存在时仍应走 AGENT
_STRONG_REASONING_PATTERNS = (
    r"够用",                # "P5够用吗"
    r"有什么区别",          # "跟别人家有什么区别"
    r"有什么不同",          # "有什么不同"
    r"比.*贵",              # "比别家贵吗"
    r"不知道",              # "不知道怎么选"
    r"大概.*差不多",        # 模糊量词
    r"性价比",              # 性价比推荐
    r"预算.*多少",          # "预算多少"
    r"安装.*多长",          # "安装要多长时间"
)

# 弱推理词：场景词存在时不再触发 AGENT（让位给 NORMAL）
_WEAK_REASONING_PATTERNS = (
    r"什么.*好",            # "会议室用什么屏好"
    r"怎么选",              # "怎么选一块会议室的屏"
    r"哪个.*合适",          # "哪个合适"
    r"帮我选",              # "帮我选"
    r"哪种",                # "哪种好"
    r"有.*推荐",            # "有什么推荐"
    r"比较.*推荐",
    r"能不能", r"可不可以", r"适合.*吗",
)

# 全部推理词（用于无场景词时的默认判断）
_REASONING_PATTERNS = _STRONG_REASONING_PATTERNS + _WEAK_REASONING_PATTERNS

# ── FAST 模式：纯参数查询，无推荐意图 ─────────────────────────────────────
# 重要原则：
#   1. pattern 避免短字符串（如"尺寸"匹配"尺寸多少"子串）
#   2. 空格敏感的 P 参数模式不带空格要求（避免"P3左右的"被误判为纯参数）
_PARAM_ONLY_PATTERNS = (
    r"亮度是?多少",      # "亮度是多少"、"亮度多少"
    r"点间距",           # "点间距"、"点间距多少"
    r"分辨率",
    r"尺寸是",           # "尺寸是多少"
    r"多少寸",           # "多少寸"
    r"多少.?nit",
    r"多少.?mm",
    r"质保",
    r"保修",
    r"warranty",
    r"支持.*吗",
    r"IP\d+",
    r"HDR",
    r"型号是",
    r"价格是多少",
    r"价格多少",
    r"价格是",
    r"price",
    r"有没有",    # "有没有防⽔的产品"、"有防⽔的吗"
    # 注意："有吗" 不加入参数模式（"有P1.2吗"、"有合适的吗" 是推荐请求的一部分）
)

# ── 寒暄模式（优先于所有其他判断）─────────────────────────────────────────
_GREETING_PATTERNS = (
    r"^你好[，。,\s]*$",
    r"^您好[，。,\s]*$",
    r"^hi[，。,\s]*$",
    r"^hello[，。,\s]*$",
    r"^嗨[，。,\s]*$",
    r"^hey[，。,\s]*$",
    r"^在吗[，。,\s]*$",
    r"^有人吗[，。,\s]*$",
    r"^早[，。,\s]*$",
    r"^晚安[，。,\s]*$",
    r"^谢谢[，。,\s]*$",
    r"^好的[，。,\s]*$",
    r"^知道了[，。,\s]*$",
    r"^再见[，。,\s]*$",
    r"^可以[，。,\s]*$",
)


def _has_scene_intent(query: str) -> bool:
    """判断 query 是否含有场景推荐意图（有场景关键词但不是纯参数查询）。"""
    q = query.strip()
    # 如果只有 P2.5/P3 这种单独型号词，不算场景
    if re.fullmatch(r"[Pp]\d+(\.\d+)?", q):
        return False
    return any(kw in q for kw in _SCENE_KEYWORDS)


def has_structured_requirement(requirements: dict | None) -> bool:
    """结构化需求里是否已经带有"场景级"上下文（Phase 6）。

    这是 Router 的第一判据：只要上游（RequirementExtractor / RequirementProfile）
    已经识别出 purpose / environment / installation / usage，就说明这是"有场景的
    推荐类查询"，不必再靠关键词表去猜 —— 关键词表降级为 fallback。
    """
    if not requirements:
        return False
    for key in ("purpose", "usage", "environment", "location_type", "installation"):
        value = requirements.get(key)
        if value not in (None, "", [], {}):
            return True
    return False


def _has_strong_reasoning_intent(query: str) -> bool:
    """强推理词（场景词存在时仍触发 AGENT）。"""
    q_lower = query.lower().strip()
    return any(re.search(p, q_lower) for p in _STRONG_REASONING_PATTERNS)


def _has_weak_reasoning_intent(query: str) -> bool:
    """弱推理词（场景词存在时让位给 NORMAL）。"""
    q_lower = query.lower().strip()
    return any(re.search(p, q_lower) for p in _WEAK_REASONING_PATTERNS)


def _has_followup_constraint(query: str) -> bool:
    """检测 query 是否包含后续补充的明确约束。

    例：
        "指挥中心用什么屏好？要能7x24小时运行"   →  True
        "会议室用什么屏好"                          →  False
        "户外屏怎么选？需要防水 IP65"               →  True
    """
    # 1) 显式 "要 / 需要 / 必须" 等约束前缀
    if re.search(r"[\?？]\s*[，,]?\s*(要|需要|必须|得|应该|希望)", query):
        return True
    if re.search(r"(要|需要|必须|得|支持|支持)\s*(.{1,12}(运行|小时|nit|寸|英寸|频率|距离|米|亮度|防水|防尘|防护|hd|hdr|赫兹))", query, re.IGNORECASE):
        return True
    # 2) 出现明确的运营/规格补充关键词，且不在推理句起始
    constraints_keywords = (
        r"7x24", r"24小时", r"全天", r"长时间运行",
        r"防水", r"防尘", r"ip65", r"ip54",
        r"hdr", r"hd r", r"高亮", r"高亮度",
        r"拼接", r"超窄边",
    )
    # 排除主问题句（？之前的内容）
    question_part = re.split(r"[\?？]", query, 1)[0]
    followup_part = query[len(question_part):]
    if not followup_part:
        return False
    return any(re.search(p, followup_part, re.IGNORECASE) for p in constraints_keywords)


def _has_reasoning_intent(query: str) -> bool:
    """判断 query 是否需要深度推理（模糊推荐、比较、疑问）。"""
    q_lower = query.lower().strip()
    return any(re.search(p, q_lower) for p in _REASONING_PATTERNS)


def _is_param_only(query: str) -> bool:
    """判断 query 是否是纯参数/规格查询（无场景推荐意图）。

    注意：场景词存在时直接返回 False（让给 NORMAL）。
    """
    q = query.strip()
    q_lower = q.lower()

    # 场景词存在 → 不是纯参数查询
    if _has_scene_intent(q):
        return False

    # 强推理词存在 → 不是纯参数查询（让给 AGENT 路由判断）
    if _has_strong_reasoning_intent(q):
        return False

    # 弱推理词 + 单独 P-pitch + 场景追问（如"适合什么场景"）→ FAST
    #   例："P1.2的小间距屏有吗？适合什么场景？"
    has_pitch = re.search(r"[Pp]\d+(?:\.\d+)?", q) is not None
    if has_pitch and _has_weak_reasoning_intent(q):
        return True

    # 纯参数词：亮度多少、点间距、IP65、型号等
    # 用 re.IGNORECASE 是因为模式中包含 "IP"、"HDR"、"warranty" 等可能大写
    if any(re.search(p, q, re.IGNORECASE) for p in _PARAM_ONLY_PATTERNS):
        return True
    # 单独出现的 P2.5、P3 等型号（前后无场景词）→ FAST
    if re.fullmatch(r"[Pp]\d+(\.\d+)?", q):
        return True
    return False


def classify_complexity(
    query: str,
    existing_requirements: dict | None = None,
) -> RoutingDecision:
    """
    Phase 1 三层业务路由（Phase 13 回归测试修复版）。

    决策顺序：
      1. 寒暄（纯） → FAST
      2. 纯参数查询（无场景词 + 无强推理词）→ FAST
      3. 场景词 + 强推理词（"够用"、"有什么区别"）→ AGENT
      4. 推理词（无场景词）→ AGENT
      5. 场景词（无论是否有弱推理词）→ NORMAL（场景推荐优先）
      6. 已有充分 requirements → NORMAL（复用已有上下文）
      7. 英文产品关键词 → NORMAL
      8. 默认 → AGENT

    关键修复（Phase 13 回归）：
      - 场景词存在时，纯参数词（"点间距"、"亮度"）不再触发 FAST
      - 场景词 + 弱推理词（"什么好"、"怎么选"）走 NORMAL（场景优先）
      - 场景词 + 强推理词（"够用"、"有什么区别"、"比别家"）走 AGENT
      - 单独 P-pitch + 弱推理词（"适合什么场景"）走 FAST（参数查询优先）
    """
    q = query.strip()
    q_lower = q.lower()
    # Phase 6：优先用"结构化需求"判断是否有场景（关键词表只作 fallback）
    has_structured_scene = has_structured_requirement(existing_requirements)
    has_scene = _has_scene_intent(q) or has_structured_scene
    has_strong_reasoning = _has_strong_reasoning_intent(q)
    has_weak_reasoning = _has_weak_reasoning_intent(q)

    # 0. 寒暄 → FAST
    for pat in _GREETING_PATTERNS:
        if re.search(pat, q_lower):
            return RoutingDecision(
                route=QueryRoute.FAST,
                reason="寒暄/问候，模板回复",
                complexity="trivial",
                _legacy_route="fast",
            )

    # 1. 纯参数查询（无场景意图 + 无强推理意图）→ FAST
    #    例："亮度是多少"、"P2.5"、"IP65防水等级"、"支持HDR吗"
    if _is_param_only(q):
        inferred = _extract_parameter_constraints(q)
        return RoutingDecision(
            route=QueryRoute.FAST,
            reason=f"纯参数查询，提取约束: {inferred or '无额外约束'}",
            inferred_constraints=inferred,
            complexity="parameter",
            _legacy_route="fast",
        )

    # 1b. 纯型号查询（含 P\d 模式但无场景词）→ FAST
    #    例："P2.5亮度"、"P3左右的"、"P1.2的小间距屏有吗？适合什么场景？"
    #    修复：这种 query 是参数查询，不应被英文推荐模式拦走 NORMAL
    #    注意：[Pp]\d 必须出现在单词起始位置，避免匹配 "IP65" 中的 "P6"
    has_pitch_only = bool(re.search(r"(?<![A-Za-z0-9])[Pp]\d", q)) and not has_scene
    has_pitch_term = bool(re.search(r"(?<![A-Za-z0-9])[Pp]\d", q))
    # 问号结尾 / "有吗" + 适合什么 等典型参数查询模式
    if has_pitch_only and (
        re.search(r"[\?？]\s*$", q)
        or "有吗" in q
        or "适合什么" in q
        or re.search(r"[Pp]\d\.?\d?\s*(亮度|点间距|尺寸|价格|nit)", q_lower)
        or re.fullmatch(r"[Pp]\d\.?\d?(\s*(左右的|左右的的?))?", q)
        # P\d 后必须跟至少一个字母或数字（如 P2.5B、P3mm），排除含推理词的中文
        or re.fullmatch(r"[Pp]\d+\.?[a-zA-Z0-9]+", q_lower)
    ):
        # 例外：已有 requirements 上下文时，P\d 查询走 NORMAL（复用上下文）
        req_ctx = existing_requirements or {}
        if any(req_ctx.get(k) for k in ("usage", "purpose", "indoor", "outdoor", "display_type")):
            inferred = _extract_parameter_constraints(q)
            return RoutingDecision(
                route=QueryRoute.NORMAL,
                reason="已有需求上下文，P\\d 查询 → NORMAL",
                inferred_constraints=inferred,
                complexity="simple",
                _legacy_route="agent",
            )
        inferred = _extract_parameter_constraints(q)
        return RoutingDecision(
            route=QueryRoute.FAST,
            reason=f"纯型号查询（P\d 参数），提取约束: {inferred or '无额外约束'}",
            inferred_constraints=inferred,
            complexity="parameter",
            _legacy_route="fast",
        )
    # 单独 P\d 单独出现（"P2.5"、"P3左右的"）→ FAST
    #  修复：只匹配真正简短的 P\d 表达式，不匹配含推理词的长 query（如 "P5够用吗"）
    #  策略：query 总长度 <= 8 时才认为是真的 bare pitch expression
    PITCH_MAX_LEN = 8
    if (has_pitch_term
            and re.fullmatch(r"[Pp]\d+\.?\d*(左右的?|左右的?左右)?", q)
            and len(q) <= PITCH_MAX_LEN):
        if has_strong_reasoning:
            # 含强推理词 → 交给后续逻辑
            pass  # fall through to step 2
        else:
            inferred = _extract_parameter_constraints(q)
            return RoutingDecision(
                route=QueryRoute.FAST,
                reason=f"单独型号 P\d → FAST",
                inferred_constraints=inferred,
                complexity="parameter",
                _legacy_route="fast",
            )

    # 2. 场景词 + 强推理词 → AGENT（深度推理）
    #    例："P5够用吗"、"跟别人家有什么区别"、"比别家贵吗"
    if has_scene and has_strong_reasoning:
        inferred = _extract_parameter_constraints(q)
        return RoutingDecision(
            route=QueryRoute.AGENT,
            reason="场景词 + 强推理词，需要 Agent 推理",
            inferred_constraints=inferred,
            complexity="complex",
            _legacy_route="agent",
        )

    # 3. 推理词（无场景词）→ AGENT
    #    例："什么屏比较好"、"怎么选"、"哪个好"、"P2.5还是P3"
    if has_strong_reasoning or has_weak_reasoning:
        # 例外：场景词 + 弱推理词 + 后续补充约束 ("指挥中心用什么屏好？要能7x24小时运行")
        # 这种情况下用户实际上已经追加了明确条件，可以走 NORMAL
        if (
            has_scene
            and has_weak_reasoning
            and not has_strong_reasoning
            and _has_followup_constraint(q)
        ):
            inferred = _extract_parameter_constraints(q)
            return RoutingDecision(
                route=QueryRoute.NORMAL,
                reason="场景词 + 弱推理词 + 后续补充约束 → NORMAL",
                inferred_constraints=inferred,
                complexity="simple",
                _legacy_route="agent",
            )
        return RoutingDecision(
            route=QueryRoute.AGENT,
            reason="模糊推荐/比较/疑问，需要 Agent 推理",
            complexity="complex",
            _legacy_route="agent",
        )

    # 4. 有场景关键词的明确推荐 → NORMAL
    #    例："户外广告屏"、"会议室P2.5"、"租赁屏P3.9"、"展厅大屏"
    #    修复：原来这些会进 FAST，现在进 NORMAL（需要 RAG + 推荐，但不需复杂推理）
    if has_scene:
        inferred = _extract_parameter_constraints(q)
        return RoutingDecision(
            route=QueryRoute.NORMAL,
            reason=f"明确场景推荐（{inferred and '有参数约束' or '无额外约束'}），进 NORMAL",
            inferred_constraints=inferred,
            complexity="simple",
            _legacy_route="agent",
        )

    # 5. 已有充分 requirements 上下文（多轮对话中途进入）→ NORMAL
    req = existing_requirements or {}
    if any(req.get(k) for k in ("usage", "purpose", "indoor", "outdoor", "display_type")):
        # 已有上下文：如果当前 query 是 bare 参数查询，仍然 FAST
        if _is_param_only(q):
            inferred = _extract_parameter_constraints(q)
            return RoutingDecision(
                route=QueryRoute.FAST,
                reason=f"已有上下文，但 query 是纯参数查询 → FAST: {inferred or '无额外约束'}",
                inferred_constraints=inferred,
                complexity="parameter",
                _legacy_route="fast",
            )
        return RoutingDecision(
            route=QueryRoute.NORMAL,
            reason="已有需求上下文，复用 NORMAL",
            inferred_constraints=None,
            complexity="simple",
            _legacy_route="agent",
        )

    # 5. 英文产品推荐关键词 → NORMAL（修复：原来走FAST，现在进NORMAL）
    #    例："meeting room LED screen"、"outdoor advertising display"
    _ENGLISH_RECOMMEND_PATTERNS = (
        r"\bindoor\b", r"\boutdoor\b",
        r"\bled\s*(display|screen|panel)?",
        r"\blcd\s*(display|screen|panel)?",
        r"\bifp\b",
        r"\bdigital\s*(signage|display|screen)?",
        r"\bretail\b", r"\bexhibition\b", r"\bexhibit\b",
        r"\bp\d+(\.\d+)?\b",
    )
    if any(re.search(p, q_lower) for p in _ENGLISH_RECOMMEND_PATTERNS):
        inferred = _extract_parameter_constraints(q)
        return RoutingDecision(
            route=QueryRoute.NORMAL,
            reason=f"英文产品推荐，进 NORMAL（{inferred or '无额外约束'}）",
            inferred_constraints=inferred,
            complexity="simple",
            _legacy_route="agent",
        )

    # 6. 默认 → AGENT
    return RoutingDecision(
        route=QueryRoute.AGENT,
        reason="默认：无法确定简单程度，委托 Agent",
        complexity="complex",
        _legacy_route="agent",
    )


def _extract_parameter_constraints(query: str) -> dict | None:
    """从 Query 中提取结构化参数约束。"""
    constraints: dict = {}
    q_lower = query.lower()

    # 点间距
    pitch = re.search(r"[Pp](\d+(?:\.\d+)?)", query)
    if pitch:
        constraints["pixel_pitch"] = float(pitch.group(1))
        constraints["pixel_pitch_tolerance"] = 0.5  # ±0.5mm 容差

    # 亮度
    brightness = re.search(r"(\d+)\s*nit", q_lower)
    if brightness:
        constraints["brightness_min"] = int(brightness.group(1))

    # 尺寸
    size = re.search(r"(\d+)\s*(?:寸|英寸|inch)", query)
    if size:
        constraints["size_inch"] = int(size.group(1))

    # 环境
    if "户外" in query or "outdoor" in q_lower:
        constraints["outdoor"] = True
        constraints["indoor"] = False
    elif "室内" in query or "indoor" in q_lower:
        constraints["indoor"] = True
        constraints["outdoor"] = False

    # 场景 purpose → 环境推断（补充室内推断）
    purpose_indoor = (
        "商场", "商店", "零售", "店铺", "超市",
        "展厅", "展馆", "展览", "博物馆",
        "指挥中心", "监控中心", "中控室",
        "机场", "车站", "码头", "地铁",
        "酒店", "大堂", "银行",
        "会议室", "教室", "培训室", "学校",
        "医院", "诊所",
        "餐厅", "酒吧", "咖啡厅",
    )
    purpose_outdoor = (
        "户外", "室外", "露天",
        "建筑外墙", "幕墙", "楼体",
        "体育场", "操场", "广场",
        "演唱会", "舞台", "演出",
    )
    for kw in purpose_indoor:
        if kw in query:
            constraints.setdefault("indoor", True)
            constraints.setdefault("outdoor", False)
            constraints["purpose"] = kw
            break
    else:
        for kw in purpose_outdoor:
            if kw in query:
                constraints.setdefault("outdoor", True)
                constraints.setdefault("indoor", False)
                constraints["purpose"] = kw
                break
        else:
            # 半户外场景：不是完全户外，也不是纯室内
            if "半户外" in query or "半室外" in query or "遮阳" in query:
                constraints["semi_outdoor"] = True
                constraints["purpose"] = "半户外"
            # 指挥中心 / 监控：偏好 LCD 拼接屏（7x24h 运行）
            elif "指挥" in query or "监控中心" in query or "控制室" in query:
                constraints["display_type"] = "LCD"
                constraints["is_splicing"] = True
                constraints["purpose"] = "指挥监控"
                constraints.setdefault("indoor", True)

    # 租赁
    if any(kw in q_lower for kw in ("租赁", "租用", "rental", "临时")):
        constraints["is_rental"] = True

    # 固定安装
    if any(kw in q_lower for kw in ("固定", "fixed", "安装", "permanent")):
        constraints["is_rental"] = False

    # 防水
    if "防水" in query or "waterproof" in q_lower:
        constraints["waterproof"] = True

    # IFP 特征关键词（含否定检测：不需要/不要/普通屏 → 排除 IFP）
    ifp_keywords = ("手写", "书写", "白板", "批注", "触控", "触摸", "多点触控",
                    "interactive", "whiteboard", "annotate", "handwriting", "touch screen")
    has_ifp_word = any(kw in query for kw in ifp_keywords)
    # 否定 IFP 的表达
    has_ifp_negation = any(
        kw in query for kw in ("不需要", "不需", "不要", "不需要手写", "普通显示屏",
                               "普通屏", "不需要IFP", "led屏", "lcd屏")
    )
    if has_ifp_word and not has_ifp_negation:
        constraints["display_type"] = "IFP"
    elif has_ifp_negation:
        constraints["exclude_ifp"] = True

    return constraints if constraints else None
