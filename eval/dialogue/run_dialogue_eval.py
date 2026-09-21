"""v2.7 Phase 17（§24/§25）：自然对话 Golden Dataset 离线评测。

不依赖 LLM：用对话层的纯函数复算"这一轮该做什么 / 问哪一项 / 说什么风格"，
并算出计划 §25 要求的话术指标：

    Question Repetition Rate        重复提问率（目标 0）
    Mechanical Phrase Rate          机械话术率（目标接近 0）
    Average Response Length         平均回复长度（普通采集 1~2 句）
    One-Question Compliance         一轮一问合规率（目标 100%）
    Unnecessary Confirmation Rate   多余确认率（目标 0）
    Context Continuation Rate       顺着客户话题推进的比例

用法：

    python -m eval.dialogue.run_dialogue_eval
    python -m eval.dialogue.run_dialogue_eval --json eval/dialogue/report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dialogue import (  # noqa: E402
    ANSWER,
    MINIMAL,
    build_natural_continuation,
    compute_answer_coverage,
    compute_momentum,
    count_questions,
    decide_response_density,
    decide_turn_action,
    find_mechanical_prefixes,
    next_question_plan,
    render_minimal,
    reset_conversation_state,
    strip_mechanical_phrases,
)
from src.dialogue.question_flow import previous_slot_blocked  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_dialogue.json")


@dataclass
class CaseResult:
    case_id: str
    action: str
    question_slot: str
    density: str
    response: str
    expected_action: str
    expected_slot: str
    expected_slot_any: List[str]
    expected_style: str
    repeated_question: bool
    mechanical: bool
    unnecessary_confirmation: bool
    question_count: int
    continued_context: bool

    @property
    def action_ok(self) -> bool:
        return self.action == self.expected_action

    @property
    def slot_ok(self) -> bool:
        if self.expected_slot_any:
            return self.question_slot in self.expected_slot_any
        return self.question_slot == self.expected_slot

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "action": self.action,
            "expected_action": self.expected_action,
            "question_slot": self.question_slot,
            "expected_question_slot": self.expected_slot,
            "expected_question_slot_any": list(self.expected_slot_any),
            "density": self.density,
            "expected_response_style": self.expected_style,
            "action_ok": self.action_ok,
            "slot_ok": self.slot_ok,
            "repeated_question": self.repeated_question,
            "mechanical": self.mechanical,
            "unnecessary_confirmation": self.unnecessary_confirmation,
            "question_count": self.question_count,
            "continued_context": self.continued_context,
            "response": self.response,
        }


def _profile(slots: Dict[str, Any]) -> RequirementProfile:
    return RequirementProfile.from_slots(slots, explicit_keys=set(slots))


def run_case(case: Dict[str, Any]) -> CaseResult:
    session_id = f"eval-dialogue-{case['id']}"
    reset_conversation_state(session_id)
    profile = _profile(case.get("requirement_slots") or {})
    previous_slot = str(case.get("previous_question_slot") or "")
    if previous_slot:
        # 复现"上一轮问的就是这一项"的状态
        profile.last_asked_slot = previous_slot
        profile.record_ask(previous_slot)

    plan = next_question_plan(profile, session_id=session_id)
    question_slot = str(getattr(plan, "slot", "") or "")
    question = str(getattr(plan, "question", "") or "")

    message = str(case.get("customer_message") or "")
    lowered = message.lower()
    customer_question = "?" in message or "how long" in lowered or "price" in lowered
    question_kind = ""
    if "delivery" in lowered or "how long" in lowered:
        question_kind = "DELIVERY_QUESTION"
    elif "price" in lowered or "cost" in lowered:
        question_kind = "PRICE_QUESTION"
    elif customer_question:
        question_kind = "PRODUCT_QUESTION"
    recommend_requested = "you decide" in lowered or "recommend" in lowered

    newly_filled = [key for key in (case.get("requirement_slots") or {}) if key != "display_type"]
    momentum = compute_momentum(
        newly_filled_slots=newly_filled, last_question_slot=previous_slot
    )
    coverage = compute_answer_coverage(
        asked_slot=previous_slot, newly_filled_slots=newly_filled
    )
    action = decide_turn_action(
        customer_question=customer_question,
        question_kind=question_kind,
        recommend_requested=recommend_requested,
        newly_filled_slots=newly_filled,
        missing_slots=[question_slot] if question_slot else [],
        question_candidates=[question_slot] if question_slot else [],
        momentum_slot=momentum.slot,
        previous_question_slot=previous_slot,
        blocked_slot=previous_slot if previous_slot == question_slot else "",
    )
    density = decide_response_density(
        customer_question=customer_question,
        question_kind=question_kind,
        newly_filled_slots=newly_filled,
        has_recommendation=recommend_requested,
    )
    continuation = build_natural_continuation(
        customer_message=message,
        newly_filled_slots=newly_filled,
        momentum=momentum,
        next_required_slot=question_slot,
        customer_question=customer_question,
        question_kind=question_kind,
        has_recommendation=recommend_requested,
    )
    # 这一轮真实会对客户说的话（MINIMAL → 只留那一句）
    sample = str(case.get("sample_response") or "")
    response = (
        render_minimal(continuation, question=question) if density == MINIMAL else sample
    )
    response, _removed = strip_mechanical_phrases(
        response, allow=density != MINIMAL
    )

    repeated = bool(previous_slot) and previous_slot == question_slot
    if repeated:
        repeated = previous_slot_blocked(profile, question_slot)
    mechanical = bool(find_mechanical_prefixes(response))
    # 动作不需要提问时，这一轮就没有 question_slot（不能报一个没问出去的槽位）
    reported_slot = question_slot if action.asks_question else ""
    return CaseResult(
        case_id=str(case.get("id") or ""),
        action=action.action,
        question_slot=reported_slot,
        density=density,
        response=response,
        expected_action=str(case.get("expected_action") or ""),
        expected_slot=str(case.get("expected_question_slot") or ""),
        expected_slot_any=[
            str(item) for item in (case.get("expected_question_slot_any") or [])
        ],
        expected_style=str(case.get("expected_response_style") or ""),
        repeated_question=repeated,
        mechanical=mechanical,
        unnecessary_confirmation=mechanical and density == MINIMAL,
        question_count=count_questions(response),
        continued_context=(
            not reported_slot
            or not momentum.slot
            or reported_slot in set(momentum.follow_ups) | {momentum.slot}
        ),
    )


def run(dataset_path: str = "") -> Dict[str, Any]:
    path = dataset_path or GOLDEN_PATH
    with open(path, "r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    cases: List[Dict[str, Any]] = list(dataset.get("cases") or [])
    results = [run_case(case) for case in cases]
    total = max(1, len(results))

    metrics = {
        "question_repetition_rate": round(
            sum(1 for item in results if item.repeated_question) / total, 4
        ),
        "mechanical_phrase_rate": round(
            sum(1 for item in results if item.mechanical) / total, 4
        ),
        "average_response_length": round(
            sum(len(item.response.split()) for item in results) / total, 2
        ),
        "one_question_compliance": round(
            sum(1 for item in results if item.question_count <= 1) / total, 4
        ),
        "unnecessary_confirmation_rate": round(
            sum(1 for item in results if item.unnecessary_confirmation) / total, 4
        ),
        "context_continuation_rate": round(
            sum(1 for item in results if item.continued_context) / total, 4
        ),
        "action_accuracy": round(sum(1 for item in results if item.action_ok) / total, 4),
        "question_slot_accuracy": round(
            sum(1 for item in results if item.slot_ok) / total, 4
        ),
    }
    return {
        "version": dataset.get("version", ""),
        "cases": len(results),
        "metrics": metrics,
        "results": [item.to_dict() for item in results],
    }


THRESHOLDS = {
    "question_repetition_rate": 0.0,
    "mechanical_phrase_rate": 0.0,
    "one_question_compliance": 1.0,
    "unnecessary_confirmation_rate": 0.0,
}


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="自然对话 Golden Dataset 评测（v2.7 §24/§25）")
    parser.add_argument("--dataset", default="", help="golden json 路径")
    parser.add_argument("--json", default="", help="把报告写到这个文件")
    args = parser.parse_args(argv)

    report = run(args.dataset)
    metrics = report["metrics"]
    print(f"[DialogueEval] cases={report['cases']} version={report['version']}")
    for name, value in metrics.items():
        print(f"  {name:32} {value}")

    failures = []
    for name, limit in THRESHOLDS.items():
        value = metrics.get(name, 0.0)
        if name == "one_question_compliance":
            if value < limit:
                failures.append(f"{name}={value} < {limit}")
        elif value > limit:
            failures.append(f"{name}={value} > {limit}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"[DialogueEval] 报告已写入 {args.json}")

    if failures:
        print("[DialogueEval] FAILED: " + "; ".join(failures))
        return 1
    print("[DialogueEval] PASSED")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["CaseResult", "THRESHOLDS", "main", "run", "run_case"]
