"""客户口径（2026-09-21）：不看关键词，理解整句话；但**不许说不存在的事**。

规则是：

    · 规则（正则）先抽；规则没抽到时，才允许 LLM 补
    · LLM 补的每个字段都必须带"客户原话证据"，证据必须在当前消息或最近对话里真实存在
    · 数值不由模型决定 —— 由确定性解析器从那段原话里再算一遍
    · 编造的字段一律丢弃（事实安全优先）
"""
import os
import sys
from types import SimpleNamespace

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.requirement_extractor import RequirementExtractor  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402

CONVERSATION = "\n".join(
    [
        "客户: we are doing a 5m x 10m wall for our church",
        "AI: Will it be indoors or outdoors?",
        "客户: indoor",
        "客户: about 5 meters away, people sit fairly close",
        "客户: keep the cost down if you can",
        "客户: it will mostly play videos",
    ]
)


def _extract(semantic, *, message="ok", context=CONVERSATION):
    extractor = RequirementExtractor()
    return extractor.extract(
        message,
        previous_profile=RequirementProfile(),
        semantic_override=semantic,
        use_llm=False,
        context_text=context,
    )


class TestEvidenceBasedUnderstanding:

    def test_fields_paraphrased_earlier_are_recorded(self):
        profile = _extract(
            {
                "size": "5m x 10m",
                "size_evidence": "5m x 10m wall",
                "viewing_distance": "about 5 meters",
                "viewing_distance_evidence": "about 5 meters away",
                "price_preference": "price",
                "price_preference_evidence": "keep the cost down",
                "content_type": "video",
                "content_type_evidence": "mostly play videos",
            }
        )
        assert profile.target_width_mm == 5000
        assert profile.target_height_mm == 10000
        assert profile.viewing_distance_m == 5.0
        assert profile.price_preference == "price"
        assert profile.content_type == "video"
        # 客户原话说过的字段 → 记成 explicit（不是"系统推断"）
        assert profile.sources.get("target_width_m") == "explicit"
        assert profile.sources.get("price_preference") == "explicit"

    def test_evidence_must_exist_in_the_conversation(self):
        profile = _extract(
            {
                "pixel_pitch": "P3",
                "pixel_pitch_evidence": "we need P3 screens",
            }
        )
        assert profile.pixel_pitch_mm is None, "编造的字段不能入档"

    def test_soft_values_must_be_in_the_allowed_set(self):
        profile = _extract(
            {
                "price_preference": "cheapest possible",
                "price_preference_evidence": "keep the cost down",
            }
        )
        # 证据原话能解析出 price；即便模型给了别的说法，也只能落到合法枚举
        assert profile.price_preference in ("price", None)

    def test_rules_win_when_the_current_message_already_gave_a_value(self):
        profile = _extract(
            {
                "size": "99 x 99 m",
                "size_evidence": "we are doing a 5m x 10m wall",
            },
            message="3 x 5 m",
            context=CONVERSATION,
        )
        # 当前这句规则抽到 3x5 → 不被 LLM 的证据字段覆盖
        assert profile.target_width_mm == 3000
        assert profile.target_height_mm == 5000


class TestClassifierAndPhrasingCarryTheMemory:

    def test_classifier_prompt_includes_the_conversation(self, monkeypatch):
        import importlib

        from src.memory.store import memory

        memory.clear("cls-history")
        memory.add("cls-history", "user", "we are doing a 5m x 10m wall")
        memory.add("cls-history", "assistant", "Is the installation going to be indoors or outdoors?")

        captured = {}

        class _FakeLLM:
            def invoke(self, messages):
                captured["prompt"] = "\n".join(str(getattr(m, "content", m)) for m in messages)
                return SimpleNamespace(content="need_query")

        module = importlib.import_module("src.agents.sales.nodes.classify")
        monkeypatch.setattr(module, "ChatOpenAI", lambda **_kwargs: _FakeLLM())
        state = {
            "current_message": "keep the cost down",
            "messages": [],
            "session_id": "cls-history",
            "requirements": {},
        }
        module.classify(state)
        prompt = captured.get("prompt", "")
        assert "5m x 10m wall" in prompt, "分类必须带会话记忆"
        assert "keep the cost down" in prompt
        memory.clear("cls-history")

    def test_phrasing_context_includes_recent_dialogue(self):
        from src.dialogue import build_context, generate_response

        captured = {}

        class _FakeLLM:
            def invoke(self, prompt):
                captured["prompt"] = str(prompt)
                return SimpleNamespace(content="Indoors or outdoors?")

        context = build_context(
            action="ASK",
            customer_message="3*5",
            question="Will the screen be installed indoors or outdoors?",
            question_slot="environment",
            required_question="environment",
            recent_dialogue="客户: i need a led display\nAI: What size?\n客户: 3*5",
            language="en",
        )
        generate_response(context, llm=_FakeLLM())
        assert "Recent conversation" in captured["prompt"]
        assert "i need a led display" in captured["prompt"]
