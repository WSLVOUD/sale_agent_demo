"""
提问话术回归测试：同一需求有多种自然问法（措辞可变），但问到的内容完全一致。

背景：之前每次追问都用同一句固定话术；现在按"会话 + 轮次"轮换问法，
内容（槽位）保持不变。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.readiness import (  # noqa: E402
    QUESTION_VARIANTS,
    check_recommendation_ready,
    question_for,
)
from src.agents.sales.question_planner import plan_next_question  # noqa: E402
from tests._cases import assert_all_cases  # noqa: E402


def _has_variants(slot: str) -> None:
    texts = {question_for(slot, "en", seed) for seed in range(12)}
    assert len(texts) >= 2, f"{slot} 只有一种问法"


def _keeps_the_content(case) -> None:
    slot, required = case
    for seed in range(12):
        text = (question_for(slot, "en", seed) or "").lower()
        assert any(token in text for token in required), (slot, seed, text)


class TestQuestionVariants:

    # 用例表（2026-09-22 计数瘦身：一条测试跑整张表，断言一条不少）
    CONTENT_CASES = [
        ("environment", ("indoor", "outdoor")),
        ("installation", ("fixed", "rental")),
        ("viewing_distance", ("how far", "distance", "away")),
        ("purpose", ("use", "application", "used")),
    ]

    def test_slot_has_multiple_phrasings(self):
        """每个槽位都要有 ≥2 种说法（不能只有一句固定话术）"""
        assert_all_cases(sorted(QUESTION_VARIANTS), _has_variants, label="slot")

    def test_content_stays_the_same(self):
        """同一槽位的所有说法都必须问到同一件事"""
        assert_all_cases(self.CONTENT_CASES, _keeps_the_content, label="slot")

    def test_seed_rotates_phrasing(self):
        """同槽位不同 seed 得到不同措辞，且只在有限集合内轮换"""
        seen = {question_for("viewing_distance", "en", seed) for seed in range(20)}
        assert 2 <= len(seen) <= len(QUESTION_VARIANTS["viewing_distance"]["en"])

    def test_purpose_question_has_no_examples(self):
        """客户口径：问场景/环境时**不举例**，直接问问题。

        实测反馈：追问"用在什么场景"时列了"会议室/教室/商场/广告"的例子，
        客户不要这种罗列。
        """
        from src.rag.readiness import EASIER_QUESTIONS, MISSING_LABELS

        banned = (
            "meeting room", "classroom", "retail", "advertising", "store", "shop",
            "会议室", "教室", "商场", "门店", "零售", "广告",
        )
        texts = list(QUESTION_VARIANTS["purpose"]["en"]) + list(
            QUESTION_VARIANTS["purpose"]["zh"]
        )
        texts += list(EASIER_QUESTIONS["purpose"]["en"]) + list(
            EASIER_QUESTIONS["purpose"]["zh"]
        )
        texts.append(MISSING_LABELS["purpose"])
        for text in texts:
            for word in banned:
                assert word not in text.lower(), (text, word)

    def test_chinese_variants(self):
        texts = {question_for("installation", "zh", seed) for seed in range(9)}
        assert len(texts) >= 2
        assert all("固装" in t or "固定" in t for t in texts)


class TestPlannerKeepsSlotVariesText:

    def test_slot_unchanged_but_wording_changes(self):
        profile = RequirementProfile.from_slots(
            {"environment": "indoor", "purpose": "conference"},
            explicit_keys={"environment", "purpose"},
        )
        slots = set()
        texts = set()
        for seed in range(6):
            plan = plan_next_question(profile, seed=seed)
            assert plan
            slots.add(plan["slot"])
            texts.add(plan["question"])
        assert slots == {"installation"}, "缺失槽位判定不能被措辞影响"
        assert len(texts) >= 2, "不同轮次应换一种说法"


class TestGateQuestionVaries:

    def test_gate_question_varies_by_seed(self):
        profile = RequirementProfile.from_slots(
            {"environment": "indoor", "purpose": "conference"},
            explicit_keys={"environment", "purpose"},
        )
        texts = {
            check_recommendation_ready(profile, variant_seed=seed).next_question
            for seed in range(6)
        }
        assert all(texts), "未就绪时必须给出追问"
        assert len(texts) >= 2

    def test_missing_fields_do_not_depend_on_seed(self):
        profile = RequirementProfile.from_slots(
            {"environment": "indoor", "purpose": "conference"},
            explicit_keys={"environment", "purpose"},
        )
        missing = {
            tuple(check_recommendation_ready(profile, variant_seed=seed).missing)
            for seed in range(6)
        }
        # 客户口径（2026-09-18）：只差硬性条件（固装租赁 / P值 / 尺寸）；
        # 场景 / 内容类型 / 价格取向都只记录、不阻塞推荐。
        assert missing == {(
            "installation", "pixel_pitch", "size",
        )}


class TestInstallationQuestionIsNaturalAndRotates:
    """实测反馈：追问安装方式的话术偏僵硬、而且感觉每次都一样。"""

    def test_variants_are_plentiful(self):
        texts = {question_for("installation", "en", seed) for seed in range(12)}
        assert len(texts) >= 6, texts

    def test_all_variants_ask_fixed_or_rental(self):
        for seed in range(12):
            text = (question_for("installation", "en", seed) or "").lower()
            assert "fixed" in text or "rental" in text, (seed, text)

    def test_consecutive_turns_never_repeat(self):
        """轮换步长为 1 → 连续 N 轮（N = 变体数）不重复同一句。"""
        variants = QUESTION_VARIANTS["installation"]["en"]
        texts = [question_for("installation", "en", seed) for seed in range(len(variants))]
        assert len(set(texts)) == len(texts), texts

    def test_variants_are_conversational(self):
        """不能每条都像书面条款（抽查：至少有带口语过渡的问法）。"""
        texts = [question_for("installation", "en", seed) or "" for seed in range(12)]
        assert any(
            text.startswith(("Just so", "Quick", "Should I"))
            for text in texts
        ), texts


class TestMultiTurnWordingChanges:
    """同一会话连续追问时，措辞应随轮次变化（内容不变）"""

    def test_consecutive_questions_differ(self, monkeypatch):
        import src.agents.sales.nodes.requirement as sales_req

        class _Response:
            content = '{"usage": "church"}'

        class _FakeChat:
            def __init__(self, *args, **kwargs):
                pass

            def invoke(self, *args, **kwargs):
                return _Response()

        monkeypatch.setattr(sales_req, "ChatOpenAI", _FakeChat)

        base_requirements = {
            "display_type": "LED", "location_type": "室内",
            "indoor": True, "outdoor": False, "usage": "church",
        }
        asked = []
        for turn in range(3):
            messages = [{"role": "user", "content": "church"} for _ in range(turn + 1)]
            state = {
                "messages": messages,
                "current_message": "church",
                "session_id": "wording-session",
                "requirements": dict(base_requirements),
                "additional_requirements": [],
                "intent": "need_query",
                "next_action": "ask",
                "should_generate_solution": False,
                "response": "",
            }
            result = sales_req.requirement_mining(state)
            asked.append(result.get("pending_question"))

        # 每轮都还缺 installation，所以问的是同一件事（内容一致）
        assert all(question for question in asked)
        assert len(set(asked)) >= 2, f"连续追问的措辞应当变化，实际: {asked}"
