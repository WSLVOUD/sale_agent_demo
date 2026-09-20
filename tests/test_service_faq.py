"""售后 / 服务类固定口径回归测试（客户口径 2026-09-18）：

  1. 说明书与图纸 → 下单后随货一并发给客户（我们也能提供安装图纸、说明书等技术资料）；
  2. 现场安装 → 一般不提供，建议当地找安装公司更省成本；每单随货提供安装指导说明书；
  3. 质保 → 默认 1 年、可付费延长；**客户不问就不提**。

要求：必须是英文、必须润色（不照抄原话）、绝不能编造事实/数字。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rag.service_faq import (  # noqa: E402
    FAQ_INSTALLATION,
    FAQ_MANUAL,
    FAQ_WARRANTY,
    detect_service_faq,
    service_faq_fact,
    service_faq_reply,
)


class _Resp:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, content="", error=None):
        self._content = content
        self._error = error

    def invoke(self, *args, **kwargs):
        if self._error:
            raise self._error
        return _Resp(self._content)


def _patch_llm(monkeypatch, **kwargs):
    import src.core.llm as llm_mod

    monkeypatch.setattr(llm_mod, "get_llm", lambda *a, **k: _FakeLLM(**kwargs))


class TestDetection:

    @pytest.mark.parametrize("message", [
        "你们会给我提供说明书和图纸吗",
        "有没有安装图纸",
        "do you provide the user manual?",
        "can you send the installation drawings?",
        "有技术资料吗",
    ])
    def test_manual_question(self, message):
        assert detect_service_faq(message) == FAQ_MANUAL

    @pytest.mark.parametrize("message", [
        "你们包安装吗？",
        "包安装吗",
        "上门安装吗",
        "do you also install it?",
        "is on-site installation included?",
    ])
    def test_installation_question(self, message):
        assert detect_service_faq(message) == FAQ_INSTALLATION

    @pytest.mark.parametrize("message", [
        "质保多久",
        "保修几年",
        "what about the warranty?",
        "do you offer a guarantee?",
    ])
    def test_warranty_question(self, message):
        assert detect_service_faq(message) == FAQ_WARRANTY

    @pytest.mark.parametrize("message", ["多少钱", "how much is it", "教堂室内 P3 5米x3米"])
    def test_unrelated_question(self, message):
        assert detect_service_faq(message) is None


class TestFacts:

    def test_manual_fact(self):
        fact = service_faq_fact("说明书和图纸").lower()
        assert "manual" in fact and "drawing" in fact
        assert "order" in fact and "goods" in fact
        assert "technical support" in fact

    def test_installation_fact(self):
        fact = service_faq_fact("你们包安装吗").lower()
        assert "do not provide on-site installation" in fact
        assert "local installation company" in fact
        assert "installation guide" in fact

    def test_warranty_fact(self):
        fact = service_faq_fact("质保多久").lower()
        assert "1-year warranty" in fact
        assert "extended" in fact and "fee" in fact

    def test_no_answer_when_not_asked(self):
        assert service_faq_reply("这个屏多少钱") is None
        assert service_faq_fact("教堂室内") is None


class TestPolishedEnglishReply:

    def test_reply_is_polished_english(self, monkeypatch):
        _patch_llm(
            monkeypatch,
            content="Yes, of course — the manual and the installation drawings go out with your "
                    "order, and we can also help with technical documentation.",
        )
        reply = service_faq_reply("你们会给我提供说明书和图纸吗")
        assert reply.startswith("Yes, of course")
        assert "manual" in reply

    def test_chinese_polish_is_rejected(self, monkeypatch):
        """润色结果夹带中文 → 直接退回英文标准回答（客户口径：必须英文）。"""
        _patch_llm(monkeypatch, content="当然可以，说明书和图纸会随货一起发。")
        reply = service_faq_reply("你们会给我提供说明书和图纸吗")
        assert reply == service_faq_fact("你们会给我提供说明书和图纸吗")

    def test_invented_numbers_are_rejected(self, monkeypatch):
        """润色时冒出标准回答里没有的数字（例如 3 年质保）→ 退回标准回答。"""
        _patch_llm(monkeypatch, content="All our screens come with a 3-year warranty included.")
        reply = service_faq_reply("质保多久")
        assert reply == service_faq_fact("质保多久")
        assert "3-year" not in reply

    def test_llm_failure_falls_back_to_fact(self, monkeypatch):
        _patch_llm(monkeypatch, error=RuntimeError("offline"))
        reply = service_faq_reply("你们包安装吗")
        assert reply == service_faq_fact("你们包安装吗")


class TestWarrantyIsNeverVolunteered:

    def test_catalog_warranty_is_one_year(self):
        from src.config import config
        from src.rag.json_loader import load_canonical_models

        models = load_canonical_models(config.DATA_DIR)
        assert models
        assert {m.warranty_years for m in models} == {1}, "质保统一为 1 年（客户口径）"

    def test_recommendation_prompt_never_mentions_warranty(self, monkeypatch):
        import src.agents.solution.nodes.recommend as recommend
        from src.models.requirement import RequirementProfile

        captured = {}

        class _Capture:
            def invoke(self, prompt, *args, **kwargs):
                captured["prompt"] = prompt if isinstance(prompt, str) else str(prompt)

                class _R:
                    content = "ok"
                return _R()

        monkeypatch.setattr(recommend, "get_llm", lambda *a, **k: _Capture())
        slots = {
            "display_type": "LED", "environment": "indoor", "purpose": "church",
            "content_type": "mixed", "installation": "fixed", "price_preference": "price",
            "pixel_pitch_mm": 3.0, "target_width_mm": 5000, "target_height_mm": 3000,
        }
        recommend._express_recommendation(
            recommendations=[{
                "model": "TW11-3216-P3.0", "pixel_pitch_mm": 3.0, "brightness_nit": 600,
                "cabinet_size_mm": "640mm*480mm", "modules_per_cabinet": 6,
                "price_tier": "low", "warranty_years": 1, "reasons": [],
                "installation": "fixed", "features": [],
            }],
            profile=RequirementProfile.from_slots(slots, explicit_keys=set(slots)),
            calculation=None,
            additional_requirements=[],
            customer_text="church screen",
        )
        assert "warranty" not in captured["prompt"].lower(), "客户没问质保就不能主动提"


class TestOrchestratorAttachesFaq:
    """客户问到售后口径时，无论这一轮是追问还是推荐，都要先回答。"""

    class _StubAgent:
        """编排器构造时会往 sales_agent 上注入 memory_store / solution_runner。"""

    def test_answer_is_prepended_to_the_turn_reply(self, monkeypatch):
        from src.orchestrator import DualAgentOrchestrator
        from src.rag.service_faq import service_faq_fact

        _patch_llm(monkeypatch, error=RuntimeError("offline"))
        orch = DualAgentOrchestrator(
            sales_agent=self._StubAgent(), solution_agent=self._StubAgent()
        )

        out = orch._attach_service_faq("TW11-3216-P3.0 looks like the best fit.", "你们包安装吗？")
        assert out.startswith(service_faq_fact("你们包安装吗"))
        assert out.endswith("TW11-3216-P3.0 looks like the best fit.")

    def test_unrelated_turn_is_untouched(self, monkeypatch):
        from src.orchestrator import DualAgentOrchestrator

        _patch_llm(monkeypatch, error=RuntimeError("offline"))
        orch = DualAgentOrchestrator(
            sales_agent=self._StubAgent(), solution_agent=self._StubAgent()
        )
        assert orch._attach_service_faq("Plain reply.", "教堂室内 P3") == "Plain reply."
