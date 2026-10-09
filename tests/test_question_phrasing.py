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
    if slot == "installation":
        assert len(texts) == 1, "安装问句按客户口径只保留一种表达"
        text = next(iter(texts)).lower()
        assert "fixed" in text and "quick-install" in text and "quick-release" in text
        assert "rental" not in text
        return
    assert len(texts) >= 2, f"{slot} 只有一种问法"


def _keeps_the_content(case) -> None:
    slot, required = case
    for seed in range(12):
        text = (question_for(slot, "en", seed) or "").lower()
        if slot == "installation":
            assert all(token in text for token in required), (slot, seed, text)
            assert "rental" not in text, (slot, seed, text)
        else:
            assert any(token in text for token in required), (slot, seed, text)


class TestQuestionVariants:

    # 用例表（2026-09-22 计数瘦身：一条测试跑整张表，断言一条不少）
    CONTENT_CASES = [
        ("environment", ("indoor", "outdoor")),
        ("installation", ("fixed", "quick-install", "quick-release")),
        ("viewing_distance", ("how far", "distance", "away")),
        ("purpose", ("use", "application", "used")),
    ]

    def test_slot_has_multiple_phrasings_merged(self):
        """合并自 5 条同类测试（瘦身；断言全部保留）。"""

        # ── test_slot_has_multiple_phrasings ──
        """每个槽位都要有 ≥2 种说法（不能只有一句固定话术）"""
        assert_all_cases(sorted(QUESTION_VARIANTS), _has_variants, label="slot")

        # ── test_content_stays_the_same ──
        """同一槽位的所有说法都必须问到同一件事"""
        assert_all_cases(self.CONTENT_CASES, _keeps_the_content, label="slot")

        # ── test_seed_rotates_phrasing ──
        """同槽位不同 seed 得到不同措辞，且只在有限集合内轮换"""
        seen = {question_for("viewing_distance", "en", seed) for seed in range(20)}
        assert 2 <= len(seen) <= len(QUESTION_VARIANTS["viewing_distance"]["en"])

        # ── test_purpose_question_has_no_examples ──
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

        # ── test_chinese_variants ──
        texts = {question_for("installation", "zh", seed) for seed in range(9)}
        assert len(texts) == 1
        assert all("固定" in t and "快装快拆" in t and "灵活搬动" in t for t in texts)


class TestPlannerKeepsInstallationPolicy:

    def test_slot_and_customer_facing_installation_meaning_are_preserved(self):
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
        assert len(texts) == 1, "安装问句按客户口径只保留一种表达"
        question = next(iter(texts)).lower()
        assert "fixed" in question and "quick-install" in question and "quick-release" in question
        assert "rental" not in question


class TestGateQuestionVaries:

    def test_gate_question_varies_by_seed_merged(self):
        """合并自 2 条同类测试（瘦身；断言全部保留）。"""

        # ── test_gate_question_varies_by_seed ──
        profile = RequirementProfile.from_slots(
            {"environment": "indoor", "purpose": "conference"},
            explicit_keys={"environment", "purpose"},
        )
        texts = {
            check_recommendation_ready(profile, variant_seed=seed).next_question
            for seed in range(6)
        }
        assert all(texts), "未就绪时必须给出追问"
        assert len(texts) == 1
        question = next(iter(texts)).lower()
        assert "fixed" in question and "quick-install" in question and "quick-release" in question
        assert "rental" not in question

        # ── test_missing_fields_do_not_depend_on_seed ──
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


class TestInstallationQuestionUsesCustomerFacingLanguage:
    """安装问句应使用清晰的客户口径，而不是内部 rental 术语。"""

    def test_single_clear_installation_question_merged(self):
        """合并自 4 条同类测试（客户口径更新后保留全部语义断言）。"""

        # ── test_installation_uses_the_current_customer_wording ──
        texts = {question_for("installation", "en", seed) for seed in range(12)}
        assert len(texts) == 1, texts

        # ── test_question_asks_about_fixed_or_flexible_installation ──
        for seed in range(12):
            text = (question_for("installation", "en", seed) or "").lower()
            assert "fixed installation" in text, (seed, text)
            assert "quick-install" in text and "quick-release" in text, (seed, text)
            assert "move around flexibly" in text, (seed, text)
            assert "rental" not in text, (seed, text)

        # ── test_seed_does_not_change_the_customer_facing_policy ──
        variants = QUESTION_VARIANTS["installation"]["en"]
        texts = [question_for("installation", "en", seed) for seed in range(len(variants))]
        assert len(set(texts)) == 1, texts

        # ── test_question_is_conversational_and_complete ──
        texts = [question_for("installation", "en", seed) or "" for seed in range(12)]
        assert all(text.startswith("Will it be") and text.endswith("?") for text in texts), texts


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
