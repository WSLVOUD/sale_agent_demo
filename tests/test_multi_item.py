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
    """方案 Agent 的桩：多屏时每块屏会各调一次，返回该屏的型号。"""

    def __init__(self, model="TW31-HOD-P5.7E"):
        self.model = model

    def run(self, *args, **kwargs):
        return {
            "answer": f"{self.model} fits well.",
            "products": [{"model": self.model}],
        }


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

    def test_second_screen_starts_as_a_copy_when_unspecified(self):
        """客户口径：没说清"这块要什么、那块要什么"→ 两块记成一样的。"""
        session_id = "multi-item-second"
        self._start(session_id)
        try:
            orch = self._orchestrator(_StubSales())
            orch.process_message("i need an led screen for a church", session_id)

            # 客户说"门口再来一块室外的屏" → 开第二条需求档案
            orch.process_message("门口再来一块室外的屏", session_id)
            assert memory.get_active_item_index(session_id) == 1
            items = memory.get_project_items(session_id)
            first, second = items[0]["profile"], items[1]["profile"]
            # 客户没指明差异的字段 → 两块一致
            assert second["display_type"] == first["display_type"] == "LED"
            assert second["purpose"] == first["purpose"] == "church"
        finally:
            memory.clear(session_id)

    def test_two_specs_in_one_message_are_recorded_separately(self):
        """一句话给了两块屏的规格 → 分别记录（每块一条）"""
        session_id = "multi-item-split"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        try:
            orch = self._orchestrator(_StubSales())
            orch.process_message(
                "i need two led screens: 4m wide x 2.5 high for indoor "
                "and 3m x 2m for the outdoor",
                session_id,
            )
            items = memory.get_project_items(session_id)
            assert len(items) == 2, items
            by_env = {
                item["profile"].get("environment"): item["profile"] for item in items
            }
            assert set(by_env) == {"indoor", "outdoor"}, by_env
            # 每块屏各拿各的尺寸（不是两个都记成 4x2.5）
            assert (by_env["indoor"]["target_width_m"], by_env["indoor"]["target_height_m"]) == (4.0, 2.5)
            assert (by_env["outdoor"]["target_width_m"], by_env["outdoor"]["target_height_m"]) == (3.0, 2.0)
        finally:
            memory.clear(session_id)

    def test_edit_switches_to_the_named_screen_only(self):
        """客户说"把室内那块改成 5m x 3m" → 只切到那一块（另一块不动）"""
        session_id = "multi-item-edit"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        try:
            orch = self._orchestrator(_StubSales())
            orch._split_and_apply_screen_specs(
                session_id,
                "indoor 4m x 2.5m and outdoor 3m x 2m",
            )
            assert memory.get_active_item_index(session_id) == 1

            target = orch._maybe_target_screen(
                session_id, "change the indoor one to 5m x 3m"
            )
            assert target == 0
            assert memory.get_active_item_index(session_id) == 0
            # 室外那块没被动过
            items = memory.get_project_items(session_id)
            assert items[1]["profile"]["target_width_m"] == 3.0
        finally:
            memory.clear(session_id)

    def test_unknown_specs_are_copied_to_both_screens(self):
        """两块屏但只有一组参数（没指明归属）→ 两块记成一样的"""
        from src.rag.project_items import split_multi_screen_specs

        specs = split_multi_screen_specs("indoor and outdoor, 4m x 2.5m")
        assert len(specs) == 2
        assert {spec["environment"] for spec in specs} == {"indoor", "outdoor"}
        assert all(spec["width_m"] == 4.0 and spec["height_m"] == 2.5 for spec in specs)

    def test_multi_screen_reply_lists_one_model_per_screen(self):
        """客户口径：有几块屏就按量给几个型号（不再只报一块屏）。"""
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
            # 第二块屏还没推荐过 → 清掉它的推荐记录，模拟"两块屏同时要推"
            items = memory.get_project_items(session_id)
            items[1].pop("model", None)
            items[1].pop("reply", None)
            items[1]["profile"] = RequirementProfile.from_slots(
                slots, explicit_keys=set(slots)
            ).model_dump()
            memory.set_project_items(session_id, items)

            orch = self._orchestrator(
                _StubSales(
                    products=[{"model": "TW31-HOD-P5.7E"}],
                    response="TW31-HOD-P5.7E fits well.",
                )
            )
            result = orch.process_message("that one is for advertising", session_id)

            reply = result["response"]
            # 两块屏 → 两个型号都在回复里，而且分得清哪块是哪块
            assert "TW11-3216-P3.0" in reply, reply
            assert "TW31-HOD-P5.7E" in reply, reply
            assert "Screen 1" in reply and "Screen 2" in reply, reply
        finally:
            memory.clear(session_id)


class TestMultiItemKeepsAskingInstallation:
    """实测 bug：多屏拆规格时把"教堂默认固装"写成了"客户明说"，

    于是系统再也不问"固装还是租赁"，直接按固装推荐。
    """

    def _orchestrator(self):
        from src.orchestrator import DualAgentOrchestrator

        return DualAgentOrchestrator(sales_agent=_StubSales(), solution_agent=_StubSolution())

    def _start_with_scene_default_installation(self, session_id):
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        slots = {"display_type": "LED", "environment": "indoor", "purpose": "church"}
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        # 教堂默认固装（系统默认值，客户还没答过这一项）
        profile.installation = "fixed"
        profile.sources["installation"] = "default"
        memory.set_requirement_profile(session_id, profile)

    def test_scene_default_installation_is_not_promoted_to_explicit(self):
        from src.rag.readiness import check_recommendation_ready

        session_id = "multi-item-installation"
        self._start_with_scene_default_installation(session_id)
        try:
            self._orchestrator()._split_and_apply_screen_specs(
                session_id,
                "4m wide x2.5 high for indoor and 3m x2m for the outdoor",
            )
            items = memory.get_project_items(session_id)
            assert len(items) == 2
            for item in items:
                profile = RequirementProfile.model_validate(item["profile"])
                assert profile.installation == "fixed"
                assert profile.sources["installation"] == "default", (
                    "场景默认固装不能被当成客户明说"
                )
                decision = check_recommendation_ready(profile)
                assert decision.ready is False
                assert "installation" in decision.missing, decision.missing
                # 客户这句话里明说的那几项仍然是 explicit
                assert profile.sources["environment"] == "explicit"
                assert profile.sources["target_width_m"] == "explicit"
        finally:
            memory.clear(session_id)

    def test_explicit_answer_is_shared_to_the_other_screen(self):
        """客户答过的共有项要同步到另一块屏，不能换个屏再问一遍。"""
        session_id = "multi-item-share"
        self._start_with_scene_default_installation(session_id)
        try:
            orch = self._orchestrator()
            orch._split_and_apply_screen_specs(
                session_id,
                "4m wide x2.5 high for indoor and 3m x2m for the outdoor",
            )
            # 客户回答"permanent"（回答当前这块屏）→ 记为客户确认
            live = RequirementProfile.model_validate(memory.get_requirement_profile(session_id))
            live.installation = "fixed"
            live.sources["installation"] = "explicit"
            memory.set_requirement_profile(session_id, live)

            orch._share_common_facts(session_id)

            items = memory.get_project_items(session_id)
            other = RequirementProfile.model_validate(items[0]["profile"])
            assert other.sources["installation"] == "explicit", other.sources
        finally:
            memory.clear(session_id)


class TestMultiScreenReplySurvivesSanitizer:
    """实测 bug：多屏回复被"当前这块屏（室外）"整段过滤掉，只剩 Screen 2。

    ``sanitize_customer_response(outdoor=True)`` 会把出现室内型号的句子删掉；
    多屏回复里室内那块的型号本来就是室内型号，必须整段保留。
    """

    MULTI_REPLY = (
        "Screen 1 (indoor / church): TW11-3216-P3.0 is the right call for the indoor screen.\n"
        "Screen 2 (outdoor / church): TW11-OD-P5 is the right fit for the outdoor screen."
    )

    def test_api_keeps_both_screens_for_multi_screen_reply(self):
        import asyncio

        from src import api

        class _StubOrchestrator:
            def process_message(self, message, session_id, images=None):
                return {
                    "response": TestMultiScreenReplySurvivesSanitizer.MULTI_REPLY,
                    "requirements": {"outdoor": True, "location_type": "室外"},
                    "products": [{"model": "TW11-OD-P5"}],
                    "route": "agent",
                    "complexity": "simple",
                    "multi_screen": True,
                    "_perf": {},
                }

        original = api.orchestrator
        api.orchestrator = _StubOrchestrator()
        try:
            result = asyncio.run(
                api._chat_sync(api.ChatRequest(session_id="multi-http", question="price"))
            )
        finally:
            api.orchestrator = original

        assert "TW11-3216-P3.0" in result.answer, result.answer
        assert "TW11-OD-P5" in result.answer, result.answer

    def test_single_screen_still_filters_conflicting_environment(self):
        """单屏（没有 multi_screen 标记）时，室外会话仍要过滤掉室内型号。"""
        from src.rag.rerank import sanitize_customer_response

        assert "TW11-3216-P3.0" not in sanitize_customer_response(
            self.MULTI_REPLY, outdoor=True
        )
