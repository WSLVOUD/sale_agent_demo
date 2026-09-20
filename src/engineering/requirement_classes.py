"""v2.3 §8：A / B / C 需求模型。

    A Physical  ：能换算成物理量的说法（50 people / 25㎡ / 5m×3m / last row 12m …）
    B Constraint：能换算成产品约束的说法（4K / no flicker / waterproof / small text …）
    C Semantic  ：不能直接换算成工程参数的说法（better effect / premium / you decide …）

关键规则：**C 类不得直接产生具体 P 值** —— 它们只能走"澄清"或"保守默认 + INFERRED"。
本模块只做分类与审计，不参与数值计算。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

PHYSICAL = "physical"
CONSTRAINT = "constraint"
SEMANTIC = "semantic"

_PHYSICAL_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:m|metres?|meters?|米|mm|毫米|cm|厘米|㎡|m2|sqm|平米|平方米|"
    r"people|persons?|viewers?|seats?|guests?|attendees?|人|观众|座位)"
    r"|\b\d+(?:[.,]\d+)?\s*(?:x|×|by)\s*\d+",
    re.IGNORECASE,
)
_CONSTRAINT_RE = re.compile(
    r"\b(?:4k|8k|1080p|resolution|refresh|flicker|waterproof|ip6[56]|hdr|cob|gob|"
    r"small text|text|camera|broadcast|luminance|nit|brightness|contrast)\b|"
    r"分辨率|刷新|防水|防尘|闪|拍摄|直播|小字|字号|亮度|对比度|色准",
    re.IGNORECASE,
)
_SEMANTIC_RE = re.compile(
    r"\b(?:better effect|premium|high[- ]end|good enough|same as before|you decide|"
    r"whatever you (?:think|recommend)|as good as|nice looking|quality feel)\b|"
    r"效果(?:好|要)?一点|更高(?:端|级)|高级(?:感|的)?|和.{0,6}一样|差不多就行|"
    r"你决定|看着办|随便",
    re.IGNORECASE,
)


@dataclass
class RequirementClassification:
    physical: List[str] = field(default_factory=list)
    constraint: List[str] = field(default_factory=list)
    semantic: List[str] = field(default_factory=list)

    @property
    def only_semantic(self) -> bool:
        return bool(self.semantic) and not self.physical

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "physical": list(self.physical),
            "constraint": list(self.constraint),
            "semantic": list(self.semantic),
        }


def classify_requirement(text: str) -> RequirementClassification:
    """把一句话里的说法分成 A / B / C 三类（纯规则，只用于分类与审计）。"""
    source = str(text or "")
    result = RequirementClassification()
    if not source.strip():
        return result
    for pattern, bucket in (
        (_PHYSICAL_RE, result.physical),
        (_CONSTRAINT_RE, result.constraint),
        (_SEMANTIC_RE, result.semantic),
    ):
        for match in pattern.finditer(source):
            fragment = match.group(0).strip()
            if fragment and fragment not in bucket:
                bucket.append(fragment)
    return result


def semantic_only(text: str) -> bool:
    """这句话是不是"只有 C 类说法"（不能直接产生具体工程参数）。"""
    classification = classify_requirement(text)
    return classification.only_semantic


__all__ = [
    "CONSTRAINT",
    "PHYSICAL",
    "SEMANTIC",
    "RequirementClassification",
    "classify_requirement",
    "semantic_only",
]
