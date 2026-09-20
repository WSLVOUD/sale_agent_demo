"""v2.1 Phase 5：Field Policy + Action Planner（计划第 10 / 11 / 16 / 17 节）。

核心思想：

    **一个字段缺失，不等于整个销售流程必须停止。**

    字段只阻塞"依赖它的 Action"：

        environment  → 选型强依赖，无法判断时只能 BLOCK（产品硬过滤必需）
        installation → 可降级（按场景默认 + 客户授权）
        pixel_pitch  → 可由观看距离/场景推导
        viewing_distance / size → 不阻塞推荐，只影响箱体计算

本模块是纯函数：输入 RequirementProfile（或字段状态字典），输出"现在该做什么"，
供 Recommendation Ready Gate / Calculation Gate / Question Planner 共用，
以后加报价、方案书等功能也能复用同一套 Action 判断。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Action 取值
USE = "use"                       # 已有值（客户确认 / 系统推断）→ 直接用
INFER = "infer"                   # 客户授权 AI 决定 / 可推导 → 交给 Python 推导
ASK = "ask"                       # 从没问过 → 正常问
ASK_EASIER = "ask_easier"         # 客户说过不知道 → 降门槛再问一次
ASK_LATER = "ask_later"           # 客户说过不知道 → **先换别的问题**，最后一轮再问一次
DEFER = "defer"                   # 不再追问，且不阻塞推荐
DEGRADE = "degrade"               # 不再追问，按降级推荐（记录缺失）
DEFER_CALCULATION = "defer_calculation"   # 不阻塞推荐，只延后箱体/模组计算
BLOCK = "block"                   # 真正阻塞（无法推导 + 不能代理 + 该 Action 必需）
SKIP = "skip"                     # 不关心（非硬性且不打算问）

ACTION_LABELS: Dict[str, str] = {
    USE: "已具备", INFER: "推导", ASK: "询问", ASK_EASIER: "降门槛询问",
    ASK_LATER: "稍后再问",
    DEFER: "延后", DEGRADE: "降级推荐", DEFER_CALCULATION: "延后计算", BLOCK: "阻塞",
    SKIP: "跳过",
}


@dataclass(frozen=True)
class FieldPolicy:
    """单个字段的策略（计划第 10 节的 FIELD_POLICIES）。"""

    slot: str
    # 选型是否依赖它（依赖 ≠ 一定会 BLOCK，能否阻塞由 on_exhausted 决定）
    required_for_recommendation: bool = False
    # 箱体/模组计算是否依赖它
    required_for_calculation: bool = False
    # 是否可以由 Python 推导
    can_infer: bool = False
    # 是否可由观看距离推导（点间距）
    can_infer_from_distance: bool = False
    # 客户授权 AI 决定时是否可以交给推导
    can_delegate: bool = True
    # 问不出来怎么办：BLOCK / DEGRADED / DEFER / DEFER_CALCULATION
    on_exhausted: str = "DEFER"
    # 是不是"硬性条件"（客户口径：室内外 / 固装租赁 / P值 / 尺寸）。
    # 软问题（场景 / 内容类型 / 价格取向）只在硬性条件还没问完时顺带问，
    # 硬性条件一齐就直接推荐 —— 这样既保留了销售话术，也不会让推荐被软问题拖住。
    hard_condition: bool = False
    # 要不要主动问（content_type / price 这类非硬性项不主动问，只记录）
    askable: bool = True
    # 询问优先级（数字越小越先问）
    ask_priority: int = 50
    # 真正阻塞时给客户的说明（计划第 18 节 Case 5 的口径）
    blocked_message: str = ""


DEFAULT_POLICY = FieldPolicy(slot="*", on_exhausted="DEFER", ask_priority=99)

# ── 策略表（对应计划第 16 节的表格）─────────────────────────────────────
FIELD_POLICIES: Dict[str, FieldPolicy] = {
    "environment": FieldPolicy(
        "environment",
        required_for_recommendation=True,
        required_for_calculation=True,
        hard_condition=True,
        can_infer=False,          # 场景能判定时会在抽取阶段落成 scenario_derived，不在这里猜
        can_delegate=False,       # 客户说"你决定室内外"也不能替他决定
        on_exhausted="BLOCK",
        ask_priority=10,
        blocked_message=(
            "To narrow down the suitable models, I need to know whether this will be used "
            "indoors or outdoors, because the product requirements are different."
        ),
    ),
    "purpose": FieldPolicy(
        "purpose",
        required_for_recommendation=False,   # 非绝对阻塞
        hard_condition=False,
        can_infer=False,
        can_delegate=True,
        on_exhausted="DEGRADED",
        # 场景是销售话术里的第一个问题（环境之后），但要排在硬性条件之前问；
        # 硬性条件已经齐了就不再问，直接推荐（见 hard_condition 的说明）。
        askable=True,
        ask_priority=20,
    ),
    "installation": FieldPolicy(
        "installation",
        required_for_recommendation=True,
        hard_condition=True,
        can_infer=False,
        can_delegate=True,        # 授权 AI → 按场景/产品默认固装处理
        on_exhausted="DEGRADED",
        ask_priority=30,
    ),
    "pixel_pitch": FieldPolicy(
        "pixel_pitch",
        required_for_recommendation=False,
        hard_condition=True,
        can_infer=True,
        can_infer_from_distance=True,
        can_delegate=True,
        on_exhausted="DEFER",
        ask_priority=40,
    ),
    "viewing_distance_m": FieldPolicy(
        "viewing_distance_m",
        required_for_recommendation=False,
        # 观看距离本身不阻塞推荐，但它是"反推点间距"的输入：只要点间距还没确定，
        # 它就算硬性待问项（客户口径：P 值不知道 → 转问观看距离，用它推 P 值）。
        # 点间距一旦确定（客户给的 / 已推导），cross-slot 规则会把它设成 SKIP。
        hard_condition=True,
        can_infer=False,
        can_delegate=True,
        on_exhausted="DEFER",
        ask_priority=45,
    ),
    "size": FieldPolicy(
        "size",
        required_for_recommendation=False,
        required_for_calculation=True,
        hard_condition=True,
        can_infer=True,           # 客户授权时按观看距离给参考尺寸
        can_delegate=True,
        on_exhausted="DEFER_CALCULATION",
        ask_priority=50,
    ),
    "width": FieldPolicy("width", required_for_calculation=True, hard_condition=True, can_infer=True,
                         can_delegate=True, on_exhausted="DEFER_CALCULATION", ask_priority=52),
    "height": FieldPolicy("height", required_for_calculation=True, hard_condition=True, can_infer=True,
                          can_delegate=True, on_exhausted="DEFER_CALCULATION", ask_priority=54),
    # size_axis 只在"客户给了一个裸尺寸、没说方向"时才问（由 Gate 显式处理），
    # 所以这里不设成"可主动询问"，避免无缘无故问"这是宽还是高"。
    "size_axis": FieldPolicy("size_axis", can_infer=False, can_delegate=False,
                             on_exhausted="DEFER_CALCULATION", askable=False, ask_priority=5),
    # display_type：默认按 LED 处理（目录里也只有 LED），不主动问 —— 保持既有口径
    "display_type": FieldPolicy("display_type", can_infer=True, can_delegate=True,
                                on_exhausted="DEGRADED", askable=False, ask_priority=15),
    "content_type": FieldPolicy("content_type", can_infer=False, can_delegate=True,
                                on_exhausted="DEFER", askable=False, ask_priority=90),
    # 价位取向（价格 / 质量 / 两者都行）：销售话术里的"推荐前那一问"。
    # 客户口径：客户没提到就不影响推荐（两者都行 → 走默认档）。
    "price_preference": FieldPolicy("price_preference", can_infer=False, can_delegate=True,
                                    on_exhausted="DEFER", askable=True, ask_priority=35),
    "budget": FieldPolicy("budget", can_infer=False, can_delegate=True,
                          on_exhausted="DEFER", askable=False, ask_priority=96),
}

# 推荐 Gate 关心（并按优先级询问）的槽位
# 注意：对外（日志 / missing 列表 / 话术）继续用 "viewing_distance" 这个老写法，
# 槽位内部统一由 canonical_slot() 归一到 viewing_distance_m。
RECOMMENDATION_SLOTS: tuple[str, ...] = (
    "size_axis", "environment", "display_type", "purpose", "installation",
    "pixel_pitch", "viewing_distance", "size", "content_type", "price_preference",
)
# 计算 Gate 关心的槽位
CALCULATION_SLOTS: tuple[str, ...] = ("size_axis", "width", "height")


def policy_for(slot: str) -> FieldPolicy:
    """取某槽位的策略（未知槽位用默认策略）。"""
    from src.models.requirement import canonical_slot

    return FIELD_POLICIES.get(canonical_slot(slot), DEFAULT_POLICY)


def field_action(profile: Any, slot: str) -> str:
    """这个字段现在该做什么（Action Planner 的核心判断）。

    顺序（计划第 11 节）：
        有值 → USE；客户 DELEGATED → INFER；可推导 → INFER；
        DECLINED / DEFERRED / 问到头 → 按 on_exhausted 决定 DEGRADE / DEFER / BLOCK；
        UNKNOWN → 降门槛再问一次；MISSING → 正常问（不值得问的跳过）。
    """
    if profile is None:
        return ASK

    from src.models.requirement import MAX_ASKS_PER_SLOT, canonical_slot

    key = canonical_slot(slot)
    policy = policy_for(key)
    state = profile.field_decision(key)

    if state in ("CONFIRMED", "INFERRED"):
        return USE

    # 点间距：客户没给（没问过 / 说不知道 / 不愿提供都一样），
    # 只要已经知道观看距离 → 按距离**确定性推导**，不再追问
    # （计划第 16 节：Pixel Pitch 可推导；客户口径：P 值不知道就按观看距离推）
    if key == "pixel_pitch" and policy.can_infer_from_distance and _distance_known(profile):
        return INFER

    if state == "DELEGATED":
        if policy.can_delegate:
            return INFER
        # 客户授权了、但这个字段不能由 AI 决定（例如室内外）→ 仍需问/阻塞
        if policy.askable and not profile.is_exhausted(key):
            return ASK
        return _exhausted_action(policy)

    if state in ("DECLINED", "DEFERRED"):
        return _exhausted_action(policy)

    # state == UNKNOWN
    if state == "UNKNOWN" and policy.askable and not profile.is_exhausted(key):
        if key == "viewing_distance_m" and _space_facts_given(profile):
            # 客户已经给了人数 / 面积 / 进深 → 观看距离可以推导，不用再问
            return INFER
        if profile.ask_count(key) >= MAX_ASKS_PER_SLOT:
            # 已经问过两次（第二次是"最后一轮"的降门槛问法）→ 不再问
            return _exhausted_action(policy)
        # 客户口径：说了"不知道"就**先换下一个问题**，不要立刻追着问同一件事；
        # 等到其它都问完了、真的要推荐时，再用降门槛的问法问一次（见 Gate 的 parked）。
        return ASK_LATER

    # state == MISSING
    if not policy.askable:
        return SKIP
    if key == "viewing_distance_m" and _space_facts_given(profile):
        # 同上：有场地几何事实就直接推导，省掉一个问题
        return INFER
    # ── v2.2 守卫：问满上限仍然没拿到值 → 一律不再问 ─────────────────────
    #   实测 bug：客户的回答没被解析出来（拼写错误 / 答非所问 / 系统没听懂）时，
    #   字段一直停在 MISSING，于是同一个问题被问了第 3 遍、第 4 遍……
    #   "问满两次就不许再问"必须由记账保证，**不能依赖意图识别是否命中**。
    if profile.ask_count(key) >= MAX_ASKS_PER_SLOT:
        logger.info(
            "[ActionPlanner] slot=%s 已问 %d 次仍无值 → 按 %s 处理（不再追问）",
            key, profile.ask_count(key), policy.on_exhausted,
        )
        return _exhausted_action(policy)
    if policy.required_for_recommendation or policy.required_for_calculation:
        return ASK
    # 非硬性但值得问（例如场景）→ 也问；纯可选则跳过
    return ASK if policy.ask_priority < 90 else SKIP


def _distance_known(profile: Any) -> bool:
    """档案里是否已经有**客户确认过**的观看距离（用于反推点间距）。

    只认"客户侧确认"的距离：纯粹由系统估算出来的距离不能拿去推点间距，
    否则一个推断值会把另一个字段也变成推断值（P 值还是要问客户）。
    """
    distance = getattr(profile, "viewing_distance_m", None)
    if distance in (None, "", [], {}):
        return False
    try:
        return bool(profile.slot_is_confirmed("viewing_distance_m"))
    except Exception:  # pragma: no cover - 防御式
        return False


def _space_facts_given(profile: Any) -> bool:
    """客户是否已经给了"人数 / 面积 / 进深"这类场地几何事实。

    给了就说明观看距离**可以从这些事实推出来**，不必再问客户同一个问题
    （实测：客户已经说了"大约 50 个人"，系统还在问"他们站多远"）。
    """
    return any(
        getattr(profile, name, None)
        for name in ("audience_count", "room_area_sqm", "room_depth_m")
    )


def apply_cross_slot_rules(profile: Any, actions: Dict[str, str]) -> Dict[str, str]:
    """槽位之间的依赖（计划第 11 / 16 节）：点间距与观看距离是一对。

    规则（客户口径：**先问点间距**，P 值不知道才用观看距离反推）：

        · 客户已经给了点间距 / 授权 AI 决定点间距 → 不用再问观看距离
          （观看距离只用于"反推点间距"和工程参考）
        · 点间距还没问过 → 先问点间距，观看距离本轮不问
        · 点间距已经问到 DEFERRED / DECLINED（客户不知道 / 不愿说）
          → 转问观看距离，用它反推点间距

    返回同一个 actions 字典（原地更新后返回，方便调用方链式使用）。
    """
    pitch = actions.get("pixel_pitch")
    distance_key = next(
        (key for key in ("viewing_distance_m", "viewing_distance") if key in actions),
        "viewing_distance_m",
    )
    if actions.get(distance_key) in (ASK, ASK_EASIER, DEFER, DEGRADE, DEFER_CALCULATION):
        # 点间距已经确定（客户给的 / 已推导）：观看距离对选型没有作用，不再占用
        # 推荐理由（SKIP），也不该被记成"降级项"
        if pitch in (USE, INFER, ASK):
            actions[distance_key] = SKIP
    return actions


def _exhausted_action(policy: FieldPolicy) -> str:
    return {
        "BLOCK": BLOCK,
        "DEGRADED": DEGRADE,
        "DEFER": DEFER,
        "DEFER_CALCULATION": DEFER_CALCULATION,
    }.get(str(policy.on_exhausted or "").upper(), DEFER)


def plan_actions(profile: Any, slots: Optional[tuple] = None) -> Dict[str, str]:
    """Action Planner：字段状态 → 动作表（计划第 17 节）。"""
    wanted = slots or RECOMMENDATION_SLOTS
    return {slot: field_action(profile, slot) for slot in wanted}


def ask_candidates(profile: Any, slots: Optional[tuple] = None) -> List[tuple]:
    """还需要问的字段（按优先级排序），返回 ``[(slot, action)]``。"""
    actions = plan_actions(profile, slots)
    items = [
        (slot, action) for slot, action in actions.items()
        if action in (ASK, ASK_EASIER)
    ]
    items.sort(key=lambda item: policy_for(item[0]).ask_priority)
    return items


def slots_with_action(profile: Any, action: str, slots: Optional[tuple] = None) -> List[str]:
    return [slot for slot, value in plan_actions(profile, slots).items() if value == action]


def is_hard_condition(slot: str) -> bool:
    """这个槽位是不是"硬性条件"（室内外 / 固装租赁 / P值 / 尺寸）。"""
    return bool(policy_for(slot).hard_condition)


__all__ = [
    "ACTION_LABELS",
    "ASK",
    "ASK_EASIER",
    "ASK_LATER",
    "BLOCK",
    "CALCULATION_SLOTS",
    "DEFER",
    "DEFER_CALCULATION",
    "DEGRADE",
    "FIELD_POLICIES",
    "FieldPolicy",
    "INFER",
    "RECOMMENDATION_SLOTS",
    "SKIP",
    "USE",
    "apply_cross_slot_rules",
    "ask_candidates",
    "field_action",
    "is_hard_condition",
    "plan_actions",
    "policy_for",
    "slots_with_action",
]
