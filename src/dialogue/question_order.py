"""v2.3.2：提问顺序随机化（按会话种子洗牌，可复现）。

客户口径（2026-09-20）：

  · 提问顺序不再固定（原来是 室内外 → 场景 → 固装租赁 → 价位取向 → P值 → 视距 → 尺寸），
    而是**每个会话随机一个顺序**；
  · 同一个会话内顺序稳定（客户感受是随机的，但同一会话不会跳来跳去）；
  · 顺序可复现、可测试：随机种子 = 会话 id（可用 `seed_override` 固定）。

这里只负责"顺序"这一件事，不判断该问谁（那是 Gate 的职责）。
"""
from __future__ import annotations

import random
from typing import Iterable, List, Sequence


def shuffled_slots(slots: Iterable[str], seed: str = "", seed_override: int = None) -> List[str]:
    """按会话种子把槽位洗成一个稳定顺序（同一 seed 结果一致）。"""
    items = [str(slot) for slot in slots]
    if seed_override is not None:
        rng = random.Random(int(seed_override))
    else:
        rng = random.Random(str(seed or "default"))
    rng.shuffle(items)
    return items


def next_in_order(ordered: Sequence[str], candidates: Iterable[str]) -> str:
    """按"已洗好的顺序"从候选里挑第一个（候选为空返回空串）。

    这样同一会话里：先问顺序里的第 1 个，下轮问第 2 个……而不是每轮重新洗牌。
    """
    wanted = {str(item) for item in candidates}
    for slot in ordered:
        if slot in wanted:
            return slot
    return ""


__all__ = ["next_in_order", "shuffled_slots"]
