"""计划 v2.9 §二十二（Phase 14）：自然话术 Golden Dataset。

数据：`tests/dialogue/golden/natural_cases.json`

每个用例检查五件事（不只是"问对 slot"）：

    1. 问的是期望的那个 slot
    2. 只问一个问题（≤1）
    3. 不问已确认 / 不该问的 slot
    4. 不编造参数（P 值 / 型号）
    5. 不出现多个独立话术块（Validator 判定）
"""
import io
import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

GOLDEN = os.path.join(project_root, "tests", "dialogue", "golden", "natural_cases.json")


def _cases():
    with io.open(GOLDEN, encoding="utf-8") as handle:
        return json.load(handle)["cases"]


def _context_for(case):
    from src.dialogue import QuestionSpec, ResponseContext
    from src.rag.readiness import question_for

    slot = case["slot"]
    return ResponseContext(
        action=case["action"],
        customer_message=case["customer"],
        known_facts=list(case.get("known_facts") or []),
        business_goal=f"collect {slot}",
        question_spec=QuestionSpec(
            slot=slot,
            intent=f"collect {slot}",
            anchor=question_for(slot, "en", 0) or slot,
        ),
        restrictions=[
            "ask_only_one_question",
            "do_not_invent_facts",
            "do_not_repeat_known_facts",
        ],
    )


class TestNaturalResponseGolden:

    def test_every_case_keeps_the_selected_slot(self):
        from src.dialogue import generate_response
        from src.rag.readiness import question_keywords

        problems = []
        for case in _cases():
            context = _context_for(case)
            text = generate_response(context, llm=None, seed=0)
            lowered = text.lower()
            keywords = question_keywords(case["slot"]) or ()
            if keywords and not any(word.lower() in lowered for word in keywords):
                problems.append(f"{case['id']}: 没问到 {case['slot']} → {text!r}")
            if text.count("?") + text.count("？") != 1:
                problems.append(f"{case['id']}: 问句数 != 1 → {text!r}")
            for forbidden in case.get("must_not_ask") or []:
                words = question_keywords(forbidden) or ()
                if words and all(word.lower() in lowered for word in words):
                    problems.append(f"{case['id']}: 又问到了 {forbidden} → {text!r}")
            for token in case.get("must_not_mention") or []:
                if token.lower() in lowered:
                    problems.append(f"{case['id']}: 编造了 {token} → {text!r}")
        assert not problems, "\n".join(problems)

    def test_every_case_passes_the_validator(self):
        from src.dialogue import generate_response
        from src.dialogue.response_validator import validate_response

        problems = []
        for case in _cases():
            context = _context_for(case)
            text = generate_response(context, llm=None, seed=0)
            check = validate_response(
                text,
                question_slot=context.question_spec.slot,
                customer_message=case["customer"],
                required_question=context.question_spec.expression_anchor,
            )
            hard = [
                issue
                for issue in check.issues
                if issue
                not in {
                    "generic_ack",
                    "repeated_connector",
                    "repeated_question_phrasing",
                    "repeated_known_facts",
                }
            ]
            if hard:
                problems.append(f"{case['id']}: {hard} → {text!r}")
        assert not problems, "\n".join(problems)
