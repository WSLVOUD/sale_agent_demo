"""一个项目下多条屏体需求（客户口径 2026-09-18）回归测试。

客户："教堂里一块室内屏，门口再来一块室外屏" —— 一个项目、两块屏：
各自收集需求、各自推荐，最后给一份"两份推荐"的汇总。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.requirement import RequirementProfile  # noqa: E402
from src.memory.store import memory  # noqa: E402
from src.rag.project_items import (  # noqa: E402
    combined_summary,
    detect_new_item,
    product_model,
    screen_label,
)


def _profile(**slots):
    return RequirementProfile.from_slots(slots, explicit_keys=set(slots))


class TestDetectNewItem:

    def test_explicit_chinese_multi_item(self):
        profile = _profile(display_type="LED", environment="indoor", purpose="church")
        for message in (
            "教堂里这块是室内的，门口再来一块室外的",
            "另外还要一块屏",
            "还要再要一块",
            "第二块屏也帮我看看",
            "我们的门头也要装一块",
        ):
            should, reason = detect_new_item(message, profile, already_recommended=True)
            assert should is True, (message, reason)

    def test_explicit_english_multi_item(self):
        profile = _profile(display_type="LED", environment="indoor", purpose="church")
        for message in (
            "I need another screen for the entrance",
            "we also need one at the front door",
            "please quote two screens",
        ):
            should, reason = detect_new_item(message, profile, already_recommended=True)
            assert should is True, (message, reason)

    def test_environment_switch_needs_a_location_hint(self):
        profile = _profile(display_type="LED", environment="indoor", purpose="church")
        # 客户纠正自己说过的环境 → 这是改需求，不是第二块屏
        assert detect_new_item("actually make it outdoor", profile, already_recommended=True)[0] is False
        # 明确说是"门口/入口"的位置 → 第二块屏
        should, reason = detect_new_item(
            "the entrance one will be outdoor", profile, already_recommended=True
        )
        assert should is True and "environment_switch" in reason

    def test_no_signal_is_not_a_new_item(self):
        profile = _profile(display_type="LED", environment="indoor", purpose="church")
        for message in ("5 meters", "it's for a church", "I don't know"):
            assert detect_new_item(message, profile, already_recommended=True)[0] is False, message

class TestProjectItemStore:

    def test_product_model_reads_document_shape(self):
        """实测踩过：推荐结果是 Document 结构时，只认 ["model"] 会静默跳过。"""
        assert product_model({"model": "TW11-3216-P3.0"}) == "TW11-3216-P3.0"
        assert (
            product_model(
                {"id": "model-TW21-3216-P3.0", "metadata": {"model": "TW21-3216-P3.0"}}
            )
            == "TW21-3216-P3.0"
        )
        assert product_model({"id": "model-TW31-HOD-P5.7E", "metadata": {}}) == "TW31-HOD-P5.7E"
        assert product_model(None) == ""

    def test_records_and_summarises_two_items(self):
        session_id = "multi-item-store"
        memory.clear(session_id)
        try:
            memory.set_active_item_index(session_id, 0)
            memory.record_item_recommendation(
                session_id,
                {"model": "TW11-3216-P3.0", "profile": {"purpose": "church", "environment": "indoor"}},
            )
            memory.set_active_item_index(session_id, 1)
            memory.record_item_recommendation(
                session_id,
                {"model": "TW31-HOD-P5.7E", "profile": {"purpose": "advertising", "environment": "outdoor"}},
            )

            items = memory.get_project_items(session_id)
            assert [item["model"] for item in items] == ["TW11-3216-P3.0", "TW31-HOD-P5.7E"]

            summary = combined_summary(items, "en")
            assert summary
            assert summary.count("TW") == 2
            assert "2 screens" in summary
        finally:
            memory.clear(session_id)

    def test_summary_needs_two_models(self):
        assert combined_summary([], "en") is None
        assert combined_summary([{"model": "TW11-3216-P3.0"}], "en") is None

    def test_screen_label_is_language_aware(self):
        profile = {"purpose": "advertising", "environment": "outdoor"}
        assert screen_label(1, profile, "en").startswith("Screen 2 (outdoor")
        assert screen_label(1, profile, "zh").startswith("第 2 块屏（室外")


class _StubSales:
    """每轮都返回"推荐完成"的结果，并带上这一轮的产品。"""

    def __init__(self, products=None, response="TW11-3216-P3.0 fits well."):
        # 默认用真实结构（Document 形态），避免"只认 dict['model']"的回归漏掉
        self.products = products if products is not None else [
            {"id": "model-TW11-3216-P3.0", "metadata": {"model": "TW11-3216-P3.0"}}
        ]
        self.response = response
        self.seen = []

    def run(self, session_id, message, has_vision=False, **_kwargs):
        self.seen.append(message)
        return {
            "intent": "need_query",
            "next_action": "trigger_solution",
            "response": self.response,
            "requirements": {},
            "products": list(self.products),
        }


class _StubSolution:
    def run(self, *args, **kwargs):
        return {}


class TestMultiItemInOrchestrator:

    def _orchestrator(self, sales):
        from src.orchestrator import DualAgentOrchestrator

        return DualAgentOrchestrator(sales_agent=sales, solution_agent=_StubSolution())

    def _start(self, session_id):
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        slots = {"display_type": "LED", "environment": "indoor", "purpose": "church"}
        memory.set_requirement_profile(
            session_id, RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        )

    def test_first_recommendation_does_not_ask_for_more_screens(self):
        """客户口径：不要在推荐后追问 "is this the only screen in the project?"。"""
        session_id = "multi-item-no-ask"
        self._start(session_id)
        try:
            orch = self._orchestrator(_StubSales())
            result = orch.process_message("i need an led screen for a church", session_id)

            extras = result.get("extra_messages") or []
            assert extras == [], extras
            assert "only screen in the project" not in result["response"]
            items = memory.get_project_items(session_id)
            assert items and items[0]["model"] == "TW11-3216-P3.0"
        finally:
            memory.clear(session_id)

    def test_second_screen_starts_a_fresh_profile(self):
        session_id = "multi-item-second"
        self._start(session_id)
        try:
            orch = self._orchestrator(_StubSales())
            orch.process_message("i need an led screen for a church", session_id)

            # 客户说"门口再来一块室外的屏" → 开第二条需求档案
            orch.process_message("门口再来一块室外的屏", session_id)
            assert memory.get_active_item_index(session_id) == 1
            profile = memory.get_requirement_profile(session_id) or {}
            # 第一块的需求不能被带过来（环境/场景都要重新采集）
            assert profile.get("purpose") != "church" or profile.get("environment") != "indoor"
        finally:
            memory.clear(session_id)

    def test_second_recommendation_returns_combined_summary(self):
        session_id = "multi-item-summary"
        self._start(session_id)
        try:
            orch = self._orchestrator(_StubSales())
            orch.process_message("i need an led screen for a church", session_id)

            # 客户开了第二块屏，并给了它的需求
            orch.process_message("门口再来一块室外的屏", session_id)
            slots = {"display_type": "LED", "environment": "outdoor", "purpose": "advertising"}
            memory.set_requirement_profile(
                session_id, RequirementProfile.from_slots(slots, explicit_keys=set(slots))
            )

            orch = self._orchestrator(_StubSales(products=[{"model": "TW31-HOD-P5.7E"}]))
            result = orch.process_message("that one is for advertising", session_id)

            extras = result.get("extra_messages") or []
            assert len(extras) == 1, extras
            assert "TW11-3216-P3.0" in extras[0]
            assert "TW31-HOD-P5.7E" in extras[0]
            # 第二块屏的推荐要说清楚是哪一块，不能和第一块混在一起
            assert result["response"].startswith("Screen 2")
        finally:
            memory.clear(session_id)
