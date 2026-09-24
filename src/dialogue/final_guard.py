"""v2.5+++（对话决策与输出链路优化 · Phase 1）：FinalResponseGuard。

计划 §3：客户最终看到的内容**只有一个出口**，并且

    question_count <= 1     （一轮最多一个"需要客户回答"的问题）

多个内部节点（Sales / Solution / 服务口径 / 图片核对）都可能各自带出一个问题；
一旦直接拼接，客户就会在一轮里看到两个问题：

    AI: Is this indoor or outdoor?
    AI: What sort of application will this screen be used in?

Guard 的职责（只做收口，不改业务决策）：

  1. **单问题**：把多余的问题去掉，只留优先级最高的那一个
     （硬性 Gate 缺失 > 工程决策必要 > 推荐优化 > 销售偏好），其余进下一轮；
  2. **内部术语**：带内部字段/系统名的句子直接删掉，绝不发给客户；
  3. **语言护栏**：交给上层（api 层已有 enforce_english），这里只保证文本非空。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .action import question_priority
from .action import MAX_QUESTIONS_PER_TURN
from .response_validator import INTERNAL_TERMS

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])(?=\s|$)")
_QUESTION_MARK_RE = re.compile(r"[?？]")


@dataclass
class QuestionCandidate:
    """一个候选问题（谁想在这一轮问、问哪一项、优先级多少）。"""

    text: str = ""
    slot: str = ""
    priority: Optional[int] = None
    source: str = ""

    def effective_priority(self) -> int:
        if self.priority is not None:
            return int(self.priority)
        return question_priority(self.slot) if self.slot else 99

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot": self.slot,
            "priority": self.effective_priority(),
            "source": self.source,
            "text": self.text[:160],
        }


@dataclass
class GuardResult:
    text: str = ""
    questions_kept: List[str] = field(default_factory=list)
    questions_dropped: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    changed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "questions_kept": list(self.questions_kept),
            "questions_dropped": list(self.questions_dropped),
            "issues": list(self.issues),
            "changed": self.changed,
        }


def _sentences(text: str) -> List[str]:
    return [part for part in _SENTENCE_SPLIT_RE.split(str(text or "")) if part.strip()]


def _has_internal_terms(sentence: str) -> bool:
    return any(term in sentence for term in INTERNAL_TERMS)


class FinalResponseGuard:
    """客户可见文本的最后一道关卡（不改变业务决策，只做收口）。"""

    def __init__(self, *, max_questions: int = MAX_QUESTIONS_PER_TURN):
        self.max_questions = max(1, int(max_questions))

    # ── 主入口 ──────────────────────────────────────────────────────────
    def finalize(
        self,
        response: str,
        *,
        questions: Optional[Sequence[Any]] = None,
        conversation: Any = None,
        language: str = "en",
    ) -> GuardResult:
        """收口：最多保留一个问题，并清掉内部术语。"""
        result = GuardResult(text=str(response or "").strip())
        if not result.text:
            return result

        candidates = self._normalize_candidates(questions)
        sentences = _sentences(result.text)
        kept: List[str] = []
        question_indexes: List[int] = []
        for sentence in sentences:
            if _has_internal_terms(sentence):
                result.issues.append("internal_term_sentence_dropped")
                continue
            kept.append(sentence)
            if _QUESTION_MARK_RE.search(sentence):
                question_indexes.append(len(kept) - 1)

        if len(question_indexes) > self.max_questions:
            keep_index = self._pick_question(kept, question_indexes, candidates)
            dropped = []
            for index in question_indexes:
                if index == keep_index:
                    continue
                dropped.append(
                    {
                        "text": kept[index].strip()[:160],
                        "slot": self._slot_for(kept[index], candidates),
                        "priority": self._priority_for(kept[index], candidates),
                    }
                )
                kept[index] = ""
            result.questions_dropped = dropped
            result.issues.append("multiple_questions_collapsed")
            logger.info(
                "[FinalGuard] 一轮出现 %d 个问题 → 只保留 1 个（%s），其余进下一轮",
                len(question_indexes), kept[keep_index][:80],
            )
        result.questions_kept = [
            kept[index].strip() for index in question_indexes if kept[index].strip()
        ]

        text = " ".join(part.strip() for part in kept if part.strip()).strip()
        # 重复问同一项（上一轮刚问过、客户也答过）→ 交给上层决定，这里只记录
        result.changed = text != result.text
        result.text = text
        return result

    # ── 内部 ────────────────────────────────────────────────────────────
    def has_question(self, text: str) -> bool:
        return bool(_QUESTION_MARK_RE.search(str(text or "")))

    def strip_questions(self, text: str) -> str:
        """把问句整句去掉（用于"这个问题本轮不问了、留到下一轮"）。"""
        kept = [
            sentence for sentence in _sentences(text)
            if not _QUESTION_MARK_RE.search(sentence)
        ]
        return " ".join(sentence.strip() for sentence in kept if sentence.strip()).strip()

    def guard_extras(
        self,
        response: str,
        extras: Sequence[str],
        *,
        questions: Optional[Sequence[Any]] = None,
    ) -> List[str]:
        """附加气泡（`extra_messages`）也要遵守"一轮最多一个问题"。

        主回复已经带了一个问题 → 附加气泡里再有问题就把它去掉（内容保留），
        否则客户会看到两个气泡各问一个问题。
        """
        kept: List[str] = []
        budget = 0 if self.has_question(response) else 1
        for extra in extras or []:
            result = self.finalize(extra, questions=questions)
            text = result.text
            if not text:
                continue
            if self.has_question(text):
                if budget <= 0:
                    text = self.strip_questions(text)
                    if not text:
                        continue
                else:
                    budget -= 1
            kept.append(text)
        return kept

    @staticmethod
    def _normalize_candidates(questions: Optional[Iterable[Any]]) -> List[QuestionCandidate]:
        items: List[QuestionCandidate] = []
        for item in questions or []:
            if isinstance(item, QuestionCandidate):
                items.append(item)
            elif isinstance(item, dict):
                items.append(QuestionCandidate(
                    text=str(item.get("text") or item.get("question") or ""),
                    slot=str(item.get("slot") or item.get("question_slot") or ""),
                    priority=item.get("priority"),
                    source=str(item.get("source") or ""),
                ))
            elif item:
                items.append(QuestionCandidate(text=str(item)))
        return items

    @staticmethod
    def _priority_for(sentence: str, candidates: List[QuestionCandidate]) -> int:
        best: Optional[int] = None
        for candidate in candidates:
            if candidate.text and candidate.text.strip() and (
                candidate.text.strip() in sentence
                or sentence.strip().startswith(candidate.text.strip()[:20])
            ):
                value = candidate.effective_priority()
                best = value if best is None else min(best, value)
        return best if best is not None else 99

    @staticmethod
    def _slot_for(sentence: str, candidates: List[QuestionCandidate]) -> str:
        for candidate in candidates:
            if candidate.text and candidate.text.strip() in sentence:
                return candidate.slot
        return ""

    def _pick_question(
        self,
        kept: List[str],
        question_indexes: List[int],
        candidates: List[QuestionCandidate],
    ) -> int:
        """留优先级最高的那个问题（并列时留先出现的）。"""
        best_index = question_indexes[0]
        best_priority = self._priority_for(kept[best_index], candidates)
        for index in question_indexes[1:]:
            priority = self._priority_for(kept[index], candidates)
            if priority < best_priority:
                best_index, best_priority = index, priority
        return best_index


_guard = FinalResponseGuard()


def finalize_response(
    response: str,
    *,
    questions: Optional[Sequence[Any]] = None,
    conversation: Any = None,
    language: str = "en",
) -> GuardResult:
    """便捷函数：走全局 Guard 收口。"""
    return _guard.finalize(
        response, questions=questions, conversation=conversation, language=language
    )


__all__ = [
    "FinalResponseGuard",
    "GuardResult",
    "QuestionCandidate",
    "finalize_response",
]
