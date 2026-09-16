"""
客户"我不知道 / 跳过这一项"的确定性识别（《全量需求捕获与 Unknown 容错优化》Phase 4）。

设计要点：
  - **纯规则**，不为了判断 "I don't know" 额外调用 LLM；
  - 只回答"这句话有没有表达不知道/跳过"，至于"是哪一个字段不知道"，由调用方
    结合当前待问槽位判断（客户可能同时报了其它字段，那些字段照常记录）。
"""
from __future__ import annotations

import re
from typing import Optional


# 客户明确表示"不知道 / 不确定"
_DONT_KNOW_RE = re.compile(
    r"\bi (?:really )?(?:do ?n[o']t|don'?t) know\b|"
    r"\b(?:do ?n[o']t|don'?t) know (?:yet|for sure)\b|"
    r"\bno idea\b|\bhave no idea\b|"
    r"\bi'?m not sure\b|\bnot sure\b|\bunsure\b|"
    r"\bi (?:do ?n[o']t|don'?t) have (?:that|this) information\b|"
    r"\b(?:it'?s )?unknown\b|"
    r"\bi can(?:'?t|not) (?:tell|say|estimate|guess)\b|"
    r"\bcan(?:'?t|not) estimate\b|"
    r"\bhard to (?:say|tell)\b|"
    r"不知道|不清楚|不确定|没了解|没有这个信息|暂时不知道|不太清楚|无法确定|没法估计|说不好|还不确定",
    re.IGNORECASE,
)

# 客户明确要求跳过这一项
_SKIP_RE = re.compile(
    r"\bskip\b|"
    r"\b(?:please )?do ?n[o']t ask\b|"
    r"\bnot available\b|"
    r"\bwe (?:do ?n[o']t|don'?t) have (?:this|that) information\b|"
    r"不用了|这个不用|这个先跳过|先跳过|跳过|这个没有|不用问|免了|没有了|没这个",
    re.IGNORECASE,
)


def detect_no_answer(message: str) -> Optional[str]:
    """返回 ``customer_skip`` / ``customer_does_not_know`` / ``None``。"""
    text = str(message or "")
    if not text:
        return None
    if _SKIP_RE.search(text):
        return "customer_skip"
    if _DONT_KNOW_RE.search(text):
        return "customer_does_not_know"
    return None


def mentions_no_answer(message: str) -> bool:
    return detect_no_answer(message) is not None


__all__ = ["detect_no_answer", "mentions_no_answer"]
