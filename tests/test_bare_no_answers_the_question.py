"""客户用 "no" 回答是非问句 ≠ 纠正（客户口径 2026-09-30）。

实测（session_1790752042108_cxlzhgoi1）：

    AI: Is this a tender project?
    客户: no,personal
    → speech_act=CORRECTION → Policy 走 clarify_only → LCD 决策层算好的
      "Do you need an OPS slot?" **被吞掉** → 正文里没有问题 → question_slot 被清空
    → API 以为"这一轮不是在问"，拿"放宽某个条件"兜底话术顶上：
      "If one of the LCD requirements can be relaxed (for example the panel size,
       the bezel width, or the resolution)…"
    → 客户："需求都是正确的为什么会冒出这句话？"

修法：客户用一个光秃秃的 no 回答我们上一轮的**是非问句**时，那是否定回答；
只有"no + 新说法"（actually / instead / 改成…）才算纠正。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import decide_speech_policy  # noqa: E402
from src.dialogue.speech_act import ANSWER_REQUIREMENT, CORRECTION, detect_speech_act  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402


def _lcd_profile(**slots) -> RequirementProfile:
    base = {"display_type": "LCD", "lcd_category": "conference_education"}
    base.update(slots)
    return RequirementProfile.from_slots(base, explicit_keys=set(base))


class TestBareNoIsAnAnswerNotACorrection:
    @pytest.mark.parametrize(
        "slot",
        ["lcd_tender", "lcd_ops", "lcd_camera", "lcd_touch", "lcd_handwriting", "lcd_splicing"],
    )
    def test_no_answering_a_yes_no_question(self, slot):
        result = detect_speech_act("no,personal", last_asked_slot=slot)

        assert result.speech_act == ANSWER_REQUIREMENT, result.speech_act
        assert result.field == slot
        assert result.value is False

    def test_plain_no_also_counts(self):
        result = detect_speech_act("no", last_asked_slot="lcd_ops")

        assert result.speech_act == ANSWER_REQUIREMENT

    def test_a_real_correction_is_still_a_correction(self):
        """"no + 新说法" 才是纠正，不能被当成否定回答吃掉。"""
        result = detect_speech_act(
            "no, actually we need two screens instead", last_asked_slot="lcd_ops"
        )

        assert result.speech_act == CORRECTION, result.speech_act

    def test_a_no_on_a_non_yes_no_slot_is_not_hijacked(self):
        """上一轮问的是室内外这种非是非题 → "no" 的含义不清楚，保持原判（纠正）。"""
        result = detect_speech_act("no,personal", last_asked_slot="environment")

        assert result.speech_act == CORRECTION, result.speech_act


class TestPolicyKeepsAskingAfterABareNo:
    def test_policy_picks_an_ask_action_not_clarify_only(self):
        profile = _lcd_profile(lcd_size_inch=65.0, lcd_handwriting_required=True)
        profile.last_asked_slot = "lcd_tender"

        bundle = decide_speech_policy(
            "no,personal", profile=profile, ready_to_recommend=False, session_id="bare-no"
        )

        assert bundle["speech_act"]["speech_act"] == ANSWER_REQUIREMENT
        assert bundle["dialogue_action"]["action"] == "ask_only", bundle["dialogue_action"]

    def test_clarify_only_is_still_used_for_real_corrections(self):
        profile = _lcd_profile()
        profile.last_asked_slot = "lcd_ops"

        bundle = decide_speech_policy(
            "no, that's not what I meant",
            profile=profile,
            ready_to_recommend=False,
            session_id="real-correction",
        )

        assert bundle["speech_act"]["speech_act"] == CORRECTION
        assert bundle["dialogue_action"]["action"] == "clarify_only", bundle["dialogue_action"]
