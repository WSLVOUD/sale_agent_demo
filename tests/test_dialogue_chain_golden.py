"""v2.5+++（计划 §14）：对话决策与输出链路 Golden Dataset。

数据集：`eval/dialogue_chain_golden.json`（7 个典型场景，验证**行为约束**）。
"""
import json
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.requirement_extractor import RequirementExtractor  # noqa: E402
from src.dialogue import (  # noqa: E402
    FinalResponseGuard,
    build_grounded_facts,
    next_question_plan,
)
from src.dialogue.response_validator import validate_response  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.pitch_resolution import resolve_pitch  # noqa: E402

GOLDEN_PATH = os.path.join(project_root, "eval", "dialogue_chain_golden.json")


def _cases():
    with open(GOLDEN_PATH, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["id"])
def test_dialogue_chain_golden(case):
    kind = case["kind"]

    # ── Case 1/2：环境首问 ────────────────────────────────────────────────
    if kind == "environment":
        profile = RequirementExtractor().extract(
            case["customer_message"], use_llm=False
        )
        plan = next_question_plan(profile, session_id=case["id"])
        if case.get("expect_first_question_slot"):
            assert plan is not None
            assert plan.slot == case["expect_first_question_slot"]
        if case.get("expect_first_question_slot_not"):
            assert plan is None or plan.slot != case["expect_first_question_slot_not"]
        return

    # ── Case 3/4：Pitch 精确 / 最近可用 ───────────────────────────────────
    if kind == "pitch":
        slots = {
            "environment": case["environment"],
            "installation": case["installation"],
            "pixel_pitch_mm": case["requested_pitch"],
        }
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))

        class _Model:
            model = case["model"]
            pixel_pitch_mm = None

        # 型号名里的标签才是客户看到的（TW11-IR-P2.9 → P2.9）
        _Model.pixel_pitch_mm = case.get("model_pitch") or case["requested_pitch"]
        resolution = resolve_pitch(
            profile,
            _Model,
            available_pitches=case.get("available_pitches") or [case["requested_pitch"]],
        )
        assert resolution.match_type == case["expect_match_type"]
        assert resolution.needs_explanation is case["expect_explanation"]
        if case.get("expect_explanation_mentions"):
            explanation = resolution.explain()
            for token in case["expect_explanation_mentions"]:
                assert token in explanation
        return

    # ── Case 5：Pitch → Viewing Distance（推断也允许）─────────────────────
    if kind == "grounded":
        profile = RequirementProfile()
        profile.pixel_pitch_mm = case["pitch"]
        profile.sources["pixel_pitch_mm"] = "inferred"
        facts = build_grounded_facts(profile=profile)
        by_field = {fact.field: fact.source for fact in facts}
        for field, expected_source in case["expect_fact_source"].items():
            assert by_field.get(field) == expected_source, by_field
        return

    # ── Case 6：一轮最多一个问题 ──────────────────────────────────────────
    if kind == "single_question":
        guard = FinalResponseGuard()
        result = guard.finalize(case["response"])
        assert result.text.count("?") == case["expect_question_count"]
        return

    # ── Case 7：LLM 自己加业务事实 → 必须被拦 ─────────────────────────────
    if kind == "grounded_block":
        profile = RequirementExtractor().extract(
            case["customer_message"], use_llm=False
        )
        facts = build_grounded_facts(profile=profile)
        result = validate_response(case["response"], grounded_facts=facts)
        for claim in case["expect_blocked_claims"]:
            assert claim in result.ungrounded_facts, result.ungrounded_facts
        assert "ungrounded_business_fact" in result.issues
        return

    pytest.fail(f"未知用例类型：{kind}")


