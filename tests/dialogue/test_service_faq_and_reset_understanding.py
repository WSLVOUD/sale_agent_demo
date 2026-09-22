"""客户口径（2026-09-21）两条实测反馈的回归：

  ① 重置（换产品/换项目）不能只看关键词 —— 要 AI 结合最近 50 条对话判断；
     客户在否认/纠正（"我没说我用租赁款呀"）时**绝不能**清空需求重来。
  ② 服务口径（安装）不能自相矛盾 —— 标准口径只能由 LLM 组织成一段话，
     不许"标准口径 + LLM 自己的说法"硬拼；答完再接一句期望交期。
"""
import os
import sys
from types import SimpleNamespace

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.service_faq import strip_contradictory_installation_claims  # noqa: E402
from src.rag.session_switch import detect_requirement_reset  # noqa: E402

CHURCH_PROFILE = RequirementProfile.from_slots(
    {
        "display_type": "LED",
        "environment": "indoor",
        "purpose": "church",
        "installation": "fixed",
        "target_width_mm": 3000,
        "target_height_mm": 5000,
        "viewing_distance_m": 5.0,
    },
    explicit_keys={"display_type", "environment", "purpose", "viewing_distance_m"},
)


def _stub_reset_llm(monkeypatch, intent, *, reason="stub"):
    import json

    import src.core.llm as core_llm

    class _FakeLLM:
        def invoke(self, _prompt):
            payload = {"intent": intent, "reason": reason, "evidence": "客户原话片段"}
            return SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))

    monkeypatch.setattr(core_llm, "get_llm", lambda **_kwargs: _FakeLLM())


class TestResetIsDecidedByUnderstandingNotKeywords:

    def test_negation_is_never_a_reset(self, monkeypatch):
        """'我没说我用租赁款呀' → 纠正/否认，不清空需求（旧实现会重置）。"""
        _stub_reset_llm(monkeypatch, "RESET")  # 就算 LLM 判错，否定句也应该先被排除
        decision = detect_requirement_reset(
            "我没说我用租赁款呀",
            profile=CHURCH_PROFILE,
            recommended=True,
            session_id="reset-negation",
        )
        assert decision.should_reset is False
        assert "纠正" in decision.detail or "否认" in decision.detail

    def test_llm_can_reject_a_rule_candidate(self, monkeypatch):
        """需求与已确认值冲突，但 AI 结合上下文判定只是补充/纠正 → 不重置。"""
        _stub_reset_llm(monkeypatch, "CORRECT", reason="客户在补充安装方式")
        decision = detect_requirement_reset(
            "我想换成另一款",
            profile=CHURCH_PROFILE,
            recommended=True,
            session_id="reset-correct",
        )
        assert decision.should_reset is False
        assert "AI" in decision.detail

    def test_llm_confirms_a_real_switch(self, monkeypatch):
        _stub_reset_llm(monkeypatch, "RESET", reason="客户要换一个项目")
        decision = detect_requirement_reset(
            "我想换个项目，另做一块户外的屏",
            profile=CHURCH_PROFILE,
            recommended=True,
            session_id="reset-switch",
        )
        assert decision.should_reset is True
        assert decision.reason in ("llm_confirmed_reset", "new_inquiry", "explicit_request")

    def test_plain_requirement_answer_is_never_a_reset(self, monkeypatch):
        _stub_reset_llm(monkeypatch, "RESET")  # 没有候选信号 → 不该走 LLM 判定
        decision = detect_requirement_reset(
            "indoor", profile=CHURCH_PROFILE, recommended=True, session_id="reset-plain"
        )
        assert decision.should_reset is False

    def test_display_type_switch_still_resets(self):
        decision = detect_requirement_reset(
            "改成 LCD 吧",
            profile=CHURCH_PROFILE,
            recommended=True,
            display_type_change=("LED", "LCD"),
        )
        assert decision.should_reset is True
        assert decision.reason == "display_type_switch"


class TestInstallationPolicyIsConsistent:

    CONTRADICTORY = (
        "We don't usually do on-site installation, since hiring a local installer is normally "
        "more cost-effective, but every order does come with an installation guide that ships "
        "together with your goods. Yes, we do provide installation. For a fixed indoor LED "
        "display like what you're planning for your church, our team handles the on-site setup "
        "as part of the project."
    )

    def test_contradicting_sentences_are_removed(self):
        cleaned = strip_contradictory_installation_claims(self.CONTRADICTORY)
        lowered = cleaned.lower()
        assert "do provide installation" not in lowered
        assert "handles the on-site setup" not in lowered
        # 标准口径（不提供现场安装 + 随货说明书）必须保留
        assert "local installer" in lowered or "on-site installation" in lowered
        assert "installation guide" in lowered

    def test_service_branch_composes_one_coherent_reply(self, monkeypatch):
        """安装问题 → 按标准口径答（不自相矛盾），并自然接一句期望交期。"""
        import importlib

        module = importlib.import_module("src.agents.sales.nodes.script_generator")
        # 不依赖真实 LLM：让它走结构化兜底（仍然必须包含事实 + 追问）
        monkeypatch.setattr(module, "_dialogue_llm", lambda: None)

        state = {
            "current_message": "do you provide installation?",
            "messages": [{"role": "user", "content": "do you provide installation?"}],
            "session_id": "faq-install",
            "requirements": {},
            "intent": "objection",
            "next_action": "ask",
            "should_generate_solution": False,
            "response": "",
            "pending_question": "",
            "pending_slot": "",
            "requirement_profile": RequirementProfile(),
            "requirement_met": True,
        }
        result = module.script_generator(state)
        text = str(result.get("response") or "")
        lowered = text.lower()
        assert "local installer" in lowered or "do not" in lowered, text
        assert "installation guide" in lowered, text
        assert "do provide installation" not in lowered
        # 追问：期望交期（由 LLM 组织，这里验证"要问什么"被带上）
        assert "deliver" in lowered or "lead time" in lowered, text
        assert result.get("service_faq_answered") == "installation"

    def test_coordinator_does_not_prepend_a_second_copy(self):
        from src.dialogue import ResponseCoordinator

        coordinator = ResponseCoordinator()
        text = coordinator.attach_service_faq(
            "We include an installation guide with every order.",
            "do you provide installation?",
            already_answered=True,
        )
        assert text.lower().count("installation guide") == 1
