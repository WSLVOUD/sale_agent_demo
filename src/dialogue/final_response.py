"""v2.6 §4：最终回复收口 —— ONE TURN → ONE ACTION → ONE RESPONSE。

计划 §4.2/§4.3 的问题：系统里存在多个"客户可见文本出口"
（主回复 / ``extra_messages`` / 各节点自己拼的追加气泡），于是出现

    AI: What pixel pitch are you considering?
    AI: What's the main viewing distance?          ← 同一个 turn 里第二条

本模块把这件事从架构上收口：

    · 所有内部节点只允许"生成数据"（``FinalResponse`` 的字段）；
    · 客户最终看到的文本**只有一处**：``FinalResponse.text``；
    · 追加气泡不再单独发给客户，而是并入这一段文本，再交给
      ``FinalResponseGuard`` 收成"最多一个问题"；
    · 一轮只有一个最终 ``action``（多个候选由 Dialogue Policy 先挑一个）。

唯一豁免：First Contact 固定接待流程（自我介绍 + 案例视频 + 名片），
它是**素材投递**而不是对话回复，走 ``extra_channels`` 单独记录（计划 §24）。
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .final_guard import FinalResponseGuard
from .action import MAX_QUESTIONS_PER_TURN

logger = logging.getLogger(__name__)

# 计划 §4.5：一个 Turn 只允许一个 Action / 一个客户可见回复
ONE_TURN_ONE_ACTION = "ONE_TURN_ONE_ACTION"
ONE_TURN_ONE_RESPONSE = "ONE_TURN_ONE_RESPONSE"

VALIDATION_PASS = "PASS"
VALIDATION_REPAIRED = "REPAIRED"
VALIDATION_FAIL = "FAIL"


def count_questions(text: str) -> int:
    """客户可见文本里的问句数量（中英文问号都算）。"""
    return str(text or "").count("?") + str(text or "").count("？")


@dataclass
class FinalResponse:
    """这一轮**唯一**的客户可见回复（计划 §4.3 的结构）。"""

    text: str = ""
    action: str = ""
    question_count: int = 0
    question_slot: str = ""
    facts_used: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    validation_result: str = VALIDATION_PASS
    response_count: int = 1
    # 非对话通道（First Contact 素材等）：不计入 text，但要留痕
    extra_channels: List[str] = field(default_factory=list)
    dropped_questions: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_single_response(self) -> bool:
        return self.response_count == 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "action": self.action,
            "question_count": self.question_count,
            "question_slot": self.question_slot,
            "facts_used": list(self.facts_used),
            "warnings": list(self.warnings),
            "turn_id": self.turn_id,
            "validation_result": self.validation_result,
            "response_count": self.response_count,
            "extra_channels": list(self.extra_channels),
            "dropped_questions": list(self.dropped_questions),
        }


class FinalResponseCoordinator:
    """把"这一轮所有想说的话"收成一条客户可见回复。"""

    def __init__(
        self,
        guard: Optional[FinalResponseGuard] = None,
        *,
        max_questions: int = MAX_QUESTIONS_PER_TURN,
    ):
        self.guard = guard or FinalResponseGuard(max_questions=max_questions)
        self.last_guard_result: Any = None

    # ── 主入口 ──────────────────────────────────────────────────────────
    def build(
        self,
        *,
        text: str = "",
        extras: Sequence[str] = (),
        questions: Optional[Sequence[Any]] = None,
        action: str = "",
        question_slot: str = "",
        turn_id: str = "",
        facts: Optional[Iterable[Any]] = None,
        first_contact_messages: Optional[Sequence[Any]] = None,
        language: str = "en",
    ) -> FinalResponse:
        """收口：合并追加气泡 → 最多一个问题 → 一条回复。"""
        merged, merged_any = self._merge_extras(str(text or ""), extras)
        guarded = self.guard.finalize(merged, questions=questions, language=language)
        self.last_guard_result = guarded

        final_text = str(guarded.text or "").strip()
        warnings: List[str] = []
        if merged_any:
            warnings.append("extras_merged_into_single_response")
        if guarded.questions_dropped:
            warnings.append("extra_question_dropped")

        count = count_questions(final_text)
        validation = VALIDATION_PASS
        if guarded.changed or merged_any:
            validation = VALIDATION_REPAIRED
        if count > self.guard.max_questions:
            # 兜底（正常不会走到）：Guard 之后仍然多个问句 → 只留第一句问句
            final_text = self._hard_repair(final_text, warnings)
            count = count_questions(final_text)
            validation = VALIDATION_REPAIRED
        if count > self.guard.max_questions:  # pragma: no cover - 极端兜底
            final_text = self.guard.strip_questions(final_text)
            count = count_questions(final_text)
            validation = VALIDATION_FAIL
            warnings.append("question_budget_exceeded")

        final = FinalResponse(
            text=final_text,
            action=str(action or ""),
            question_count=count,
            question_slot=str(question_slot or "") if count else "",
            facts_used=[self._fact_dict(item) for item in (facts or [])],
            warnings=warnings,
            turn_id=str(turn_id or uuid.uuid4().hex[:12]),
            validation_result=validation,
            response_count=1,
            dropped_questions=list(guarded.questions_dropped or []),
        )
        if first_contact_messages:
            # 固定接待流程的素材通道（不并入 text，也不算第二条对话回复）
            final.extra_channels.append("first_contact")
        if count == 0:
            final.question_slot = ""
        if validation != VALIDATION_PASS:
            logger.info(
                "[FinalResponse] turn=%s action=%s questions=%d validation=%s warnings=%s",
                final.turn_id, final.action, final.question_count, final.validation_result,
                final.warnings,
            )
        return final

    # ── 内部 ────────────────────────────────────────────────────────────
    @staticmethod
    def _merge_extras(text: str, extras: Sequence[str]) -> tuple[str, bool]:
        """追加气泡并入**同一条**回复（客户不会再看到两个气泡）。"""
        merged = str(text or "").strip()
        changed = False
        for extra in extras or []:
            piece = str(extra or "").strip()
            if not piece:
                continue
            if piece in merged:
                continue
            merged = f"{merged}\n\n{piece}".strip() if merged else piece
            changed = True
        return merged, changed

    @staticmethod
    def _hard_repair(text: str, warnings: List[str]) -> str:
        """只保留第一个问句所在的整句，其余带问号的句子删掉。"""
        import re

        sentences = [part for part in re.split(r"(?<=[.!?。！？])(?=\s|$)", text) if part.strip()]
        kept: List[str] = []
        used = False
        for sentence in sentences:
            if ("?" in sentence or "？" in sentence):
                if used:
                    warnings.append("extra_question_removed_in_repair")
                    continue
                used = True
            kept.append(sentence)
        return " ".join(part.strip() for part in kept if part.strip()).strip()

    @staticmethod
    def _fact_dict(item: Any) -> Dict[str, Any]:
        if isinstance(item, dict):
            return dict(item)
        for attr in ("to_dict", "model_dump"):
            method = getattr(item, attr, None)
            if callable(method):
                try:
                    return dict(method())
                except Exception:  # pragma: no cover - 防御式
                    break
        return {"value": str(item)}


_COORDINATOR = FinalResponseCoordinator()


def build_final_response(**kwargs: Any) -> FinalResponse:
    """便捷函数：走全局收口器。"""
    return _COORDINATOR.build(**kwargs)


__all__ = [
    "FinalResponse",
    "FinalResponseCoordinator",
    "ONE_TURN_ONE_ACTION",
    "ONE_TURN_ONE_RESPONSE",
    "VALIDATION_FAIL",
    "VALIDATION_PASS",
    "VALIDATION_REPAIRED",
    "build_final_response",
    "count_questions",
]
