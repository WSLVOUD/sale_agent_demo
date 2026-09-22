"""
视觉链路端到端测试（计划第十二~十九阶段 / 第二十四阶段用例）。

覆盖计划第 24 节的 Case：
  1  纯图片        → Vision → RequirementProfile → Sales 继续追问
  2  图片 + 文字   → 统一合并进同一个 RequirementProfile
  5  客户与图片冲突 → 保留客户信息 + 记录冲突
  7  Vision 失败   → 主流程不崩
  15 首轮就是图片  → 仍然走 First Contact，但图片信息要保存
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.memory.store import memory  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402
from src.orchestrator import DualAgentOrchestrator  # noqa: E402
from src.vision.extractor import VisionExtractor  # noqa: E402
from src.vision.integration import apply_vision_to_profile  # noqa: E402


class _StubSales:
    def __init__(self, result=None):
        self.result = result or {}
        self.calls = []

    def run(self, session_id, message, has_vision=False, **_kwargs):
        self.calls.append(
            {"session_id": session_id, "message": message, "has_vision": has_vision}
        )
        return dict(
            {
                "intent": "need_query",
                "next_action": "ask",
                "response": "Thanks for the photo.",
                "requirements": {},
                "products": [],
            },
            **self.result,
        )


class _StubSolution:
    def __init__(self, result=None):
        self.result = result or {}

    def run(self, message, history=None, session_id=None, **_kwargs):
        return dict(self.result)


class _FakeVisionExtractor:
    """返回预设的 VisionRequirement（走真实的 integration 合并逻辑）。"""

    def __init__(self, payloads, raise_error=False):
        self.payloads = payloads
        self.raise_error = raise_error
        self.calls = 0

    def extract_many(self, images, session_id="", customer_text="", **kwargs):
        self.calls += 1
        if self.raise_error:
            raise RuntimeError("vision down")
        return [VisionExtractor.from_payload(payload) for payload in self.payloads]


INDOOR_CONFERENCE = {
    "display_type": {"value": "LED", "source": "vision_explicit", "confidence": 0.95},
    "environment": {"value": "indoor", "source": "vision_explicit", "confidence": 0.9},
    "purpose": {"value": "conference", "source": "vision_explicit", "confidence": 0.8},
    "target_width_m": {"value": 5.0, "source": "vision_inferred", "confidence": 0.4},
    "target_height_m": {"value": 3.0, "source": "vision_inferred", "confidence": 0.4},
}


@pytest.fixture
def patch_vision(monkeypatch):
    import src.vision.integration as integration

    def _apply(extractor):
        monkeypatch.setattr(integration, "get_vision_extractor", lambda: extractor)
        return extractor

    return _apply


def _orchestrator(monkeypatch, sales_result=None, solution_result=None):
    sales = _StubSales(sales_result)
    solution = _StubSolution(solution_result)
    orchestrator = DualAgentOrchestrator(sales_agent=sales, solution_agent=solution)
    monkeypatch.setattr(orchestrator, "memory_store", memory)
    return orchestrator, sales


class TestImageOnlyTurn:
    def test_image_requirements_are_merged_before_sales(self, monkeypatch, patch_vision):
        patch_vision(_FakeVisionExtractor([INDOOR_CONFERENCE]))
        orchestrator, sales = _orchestrator(monkeypatch)
        session_id = "vision-image-only"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        try:
            orchestrator.process_message("", session_id, images=[b"image-bytes"])

            profile = memory.get_requirement_profile(session_id) or {}
            assert profile["environment"] == "indoor"
            assert profile["purpose"] == "conference"
            assert profile["sources"]["environment"] == "vision_explicit"
            # 图片尺寸只作为提示，不能进工程字段
            assert profile["target_width_m"] is None
            assert profile["vision_size_hint_mm"] == [5000.0, 3000.0]
            # 纯图片消息也要驱动 Sales 继续工作
            assert sales.calls and sales.calls[0]["message"]
        finally:
            memory.clear(session_id)

    def test_image_and_text_share_one_profile(self, monkeypatch, patch_vision):
        patch_vision(_FakeVisionExtractor([INDOOR_CONFERENCE]))
        orchestrator, sales = _orchestrator(monkeypatch)
        session_id = "vision-image-text"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        try:
            orchestrator.process_message(
                "Can you recommend something like this?", session_id, images=[b"img"]
            )
            profile = memory.get_requirement_profile(session_id) or {}
            assert profile["display_type"] == "LED"
            assert sales.calls[0]["message"] == "Can you recommend something like this?"
        finally:
            memory.clear(session_id)

    def test_no_images_means_no_vision_call(self, monkeypatch):
        import src.vision.integration as integration

        extractor = _FakeVisionExtractor([INDOOR_CONFERENCE])
        monkeypatch.setattr(integration, "get_vision_extractor", lambda: extractor)
        orchestrator, _ = _orchestrator(monkeypatch)
        session_id = "vision-none"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        try:
            orchestrator.process_message("indoor LED for a church", session_id)
            assert extractor.calls == 0
        finally:
            memory.clear(session_id)

    def test_vision_disabled_ignores_images(self, monkeypatch):
        import src.orchestrator as orchestrator_module

        monkeypatch.setattr(orchestrator_module, "_vision_enabled", lambda: False)
        orchestrator, _ = _orchestrator(monkeypatch)
        session_id = "vision-disabled"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        try:
            orchestrator.process_message("", session_id, images=[b"img"])
            assert memory.get_requirement_profile(session_id) is None
        finally:
            memory.clear(session_id)


class TestConflictBetweenCustomerAndImage:
    def test_customer_value_wins_and_conflict_is_recorded(self, monkeypatch, patch_vision):
        patch_vision(_FakeVisionExtractor([INDOOR_CONFERENCE]))
        orchestrator, _ = _orchestrator(monkeypatch)
        session_id = "vision-conflict"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        try:
            # 客户已经明确说了"室外"
            slots = {"environment": "outdoor", "display_type": "LED"}
            memory.set_requirement_profile(
                session_id, RequirementProfile.from_slots(slots, explicit_keys=set(slots))
            )

            orchestrator.process_message("here is the photo", session_id, images=[b"img"])

            profile = memory.get_requirement_profile(session_id) or {}
            assert profile["environment"] == "outdoor", "客户明确说的不能被图片覆盖"
            assert profile["conflicts"], "图片与客户说法冲突必须记录"
            assert "environment" in profile["conflict_slots"]
        finally:
            memory.clear(session_id)


class TestVisionFailureIsSafe:
    def test_vision_error_does_not_break_turn(self, monkeypatch, patch_vision):
        patch_vision(_FakeVisionExtractor([INDOOR_CONFERENCE], raise_error=True))
        orchestrator, sales = _orchestrator(monkeypatch, sales_result={"response": "How can I help?"})
        session_id = "vision-failure"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        try:
            result = orchestrator.process_message("hello", session_id, images=[b"img"])

            assert result["response"] == "How can I help?"
            assert sales.calls, "Sales Agent 必须照常工作"
            assert memory.get_requirement_profile(session_id) is None
            assert result["vision"]["vision_success"] is False
        finally:
            memory.clear(session_id)

    def test_broken_vision_metrics_are_reported(self, monkeypatch, patch_vision):
        patch_vision(_FakeVisionExtractor([INDOOR_CONFERENCE], raise_error=True))
        orchestrator, _ = _orchestrator(monkeypatch)
        session_id = "vision-failure-metrics"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        try:
            result = orchestrator.process_message("hi", session_id, images=[b"img"])
            assert "vision" in result
            assert result["vision"].get("error")
        finally:
            memory.clear(session_id)


class TestFirstContactWithImage:
    class _FakeFirstContact:
        class _Result:
            intro_text = "Hi, I'm your LED advisor."
            intro_success = True
            all_success = True

            def to_messages(self):
                return [{"role": "assistant", "content": self.intro_text}]

        def run(self, session_id, customer_message, language="en"):
            return self._Result()

    def test_first_contact_still_runs_but_image_is_saved(self, monkeypatch, patch_vision):
        import src.orchestrator as orchestrator_module

        monkeypatch.setattr(orchestrator_module, "first_contact_handler", self._FakeFirstContact())
        patch_vision(_FakeVisionExtractor([INDOOR_CONFERENCE]))
        orchestrator, sales = _orchestrator(monkeypatch)
        session_id = "vision-first-contact"
        memory.clear(session_id)
        try:
            result = orchestrator.process_message("", session_id, images=[b"img"])

            assert result["route"] == "first_contact"
            assert not sales.calls, "首次接待流程内不能跑 Sales Agent"
            assert memory.is_first_contact_done(session_id)
            profile = memory.get_requirement_profile(session_id) or {}
            assert profile["environment"] == "indoor"
            assert profile["purpose"] == "conference"
        finally:
            memory.clear(session_id)


class TestVisionConfirmation:
    """图片识别出的需求要**先跟客户确认**；客户纠正时以客户为准并记录（客户口径）。"""

    def _vision_profile(self):
        profile = RequirementProfile()
        vision = VisionExtractor.from_payload(INDOOR_CONFERENCE)
        merged, _ = apply_vision_to_profile(profile, vision)
        return merged

    def test_vision_merge_marks_fields_pending_confirmation(self):
        profile = self._vision_profile()
        assert "environment" in profile.vision_confirmation_pending
        assert "purpose" in profile.vision_confirmation_pending
        assert profile.vision_assertions["environment"] == "indoor"

    def test_image_turn_reply_confirms_with_customer(self):
        """带图那一轮的回复必须说出"图片里看到什么"，并请客户确认。"""
        from src.agents.sales.nodes.script_generator import script_generator

        profile = self._vision_profile()
        state = {
            "messages": [{"role": "user", "content": "i need a display like this"}],
            "current_message": "i need a display like this",
            "session_id": "vision-confirm-turn",
            "intent": "need_query",
            "next_action": "ask",
            "requirements": {},
            "additional_requirements": [],
            "should_generate_solution": False,
            "response": "",
            "pending_question": "Is it a permanent install, or is it for rental/events?",
            "pending_slot": "installation",
            "requirement_profile": profile,
            "vision_applied": True,
        }
        out = script_generator(state)
        reply = out["response"].lower()

        assert "indoor" in reply, reply          # 图片看到的环境
        assert "conference" in reply, reply      # 图片看到的场景
        assert "?" in reply, reply               # 请客户确认
        assert "installation" in reply or "rental" in reply, reply  # 仍然继续问缺失项

    def test_no_image_turn_has_no_confirmation(self):
        from src.agents.sales.nodes.script_generator import script_generator

        profile = self._vision_profile()
        profile.vision_confirmation_pending = []      # 已经确认过
        state = {
            "messages": [{"role": "user", "content": "permanent"}],
            "current_message": "permanent",
            "session_id": "vision-confirm-none",
            "intent": "need_query",
            "next_action": "ask",
            "requirements": {},
            "additional_requirements": [],
            "should_generate_solution": False,
            "response": "",
            "pending_question": "What is the typical viewing distance?",
            "pending_slot": "viewing_distance",
            "requirement_profile": profile,
            "vision_applied": False,
        }
        out = script_generator(state)
        assert "photo" not in out["response"].lower()
        assert "indoor" not in out["response"].lower()

    def test_offtopic_image_turn_still_confirms(self):
        """实测 bug（客户日志）：只发图片 + "i need this" 被判成 off-topic，

        回复里把"图片里看到什么"的确认句丢了 → 直接跳到问点间距。
        """
        from src.agents.sales.nodes.script_generator import script_generator

        profile = self._vision_profile()
        state = {
            "messages": [{"role": "user", "content": "i need this"}],
            "current_message": "i need this",
            "session_id": "vision-confirm-offtopic",
            "intent": "need_query",
            "next_action": "ask",
            "requirements": {},
            "additional_requirements": [],
            "should_generate_solution": False,
            "response": "",
            "offtopic_turn": True,
            "acknowledgement": "Got it, I hear you.",
            "pending_question": "What pixel pitch do you have in mind?",
            "pending_slot": "pixel_pitch",
            "requirement_profile": profile,
            "vision_applied": True,
        }
        out = script_generator(state)
        reply = out["response"].lower()
        assert "photo" in reply or "picture" in reply, reply
        assert "indoor" in reply, reply
        assert "conference" in reply, reply
        assert "pixel pitch" in reply, reply

    def test_orchestrator_prepends_confirmation_when_missing(self):
        """编排器兜底：任何一条路径漏了图片确认句，也要补在最前面。"""
        from src.orchestrator import DualAgentOrchestrator
        from src.memory.store import memory

        class _StubAgent:
            pass

        session_id = "vision-orch-confirm"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        memory.set_requirement_profile(session_id, self._vision_profile())
        try:
            orch = DualAgentOrchestrator(
                sales_agent=_StubAgent(), solution_agent=_StubAgent()
            )
            out = orch._attach_vision_confirmation(
                "Which pixel pitch are you thinking of?", session_id, "i need this"
            )
            assert "photo" in out.lower()
            assert "indoor" in out.lower() and "conference" in out.lower()
            assert out.lower().rstrip().endswith("which pixel pitch are you thinking of?")
            # 已经确认过的内容不重复
            assert orch._attach_vision_confirmation(out, session_id, "i need this") == out
        finally:
            memory.clear(session_id)

    def test_reply_never_asks_what_the_vision_confirmation_just_stated(self, monkeypatch):
        """实测 bug（客户日志）：回复里刚说 "it looks like … a fixed installation …,

        correct me if I've misread it"，紧接着又问 "is this a long-term installation,
        or do you need it for rental/events?" —— 自相矛盾。

        规则：图片识别结果里的那一项，本轮**不能再问**（确认句本身就是"在问这一项"）。
        """
        import importlib

        sales_req = importlib.import_module("src.agents.sales.nodes.requirement")
        extractor_mod = importlib.import_module("src.core.requirement_extractor")
        from src.agents.sales.nodes.script_generator import script_generator

        class _Response:
            content = '{"usage": null, "additional_requirements": [], "ack": ""}'

        class _FakeChat:
            def __init__(self, *args, **kwargs):
                pass

            def invoke(self, *args, **kwargs):
                return _Response()

        monkeypatch.setattr(sales_req, "ChatOpenAI", _FakeChat)
        monkeypatch.setattr(
            extractor_mod.RequirementExtractor,
            "_llm_semantic_extract",
            lambda self, message, rule_slots, session_id="": {},
        )
        extractor_mod.RequirementExtractor._semantic_cache.clear()

        # 图片：LED / indoor / advertising / installation=fixed（只是推测，所以必须核对）
        profile = RequirementProfile()
        vision = VisionExtractor.from_payload({
            "display_type": {"value": "LED", "source": "vision_explicit", "confidence": 0.9},
            "environment": {"value": "indoor", "source": "vision_explicit", "confidence": 0.9},
            "purpose": {"value": "advertising", "source": "vision_explicit", "confidence": 0.8},
            "installation": {"value": "fixed", "source": "vision_inferred", "confidence": 0.5},
        })
        merged, _ = apply_vision_to_profile(profile, vision)
        assert "installation" in merged.vision_confirmation_pending

        state = {
            "messages": [{"role": "user", "content": "i need this"}],
            "current_message": "i need this",
            "session_id": "vision-no-contradiction",
            "intent": "need_query",
            "next_action": "ask",
            "requirements": {},
            "additional_requirements": [],
            "should_generate_solution": False,
            "response": "",
            "pending_question": "",
            "pending_slot": "",
            "requirement_profile": merged,
            "vision_applied": True,
        }
        out = sales_req.requirement_mining(state)
        # 图片确认里已经说了 fixed → 本轮不能把 installation 当成待问项
        assert out["pending_slot"] != "installation", out["pending_question"]

        reply = script_generator(out)["response"]
        lowered = reply.lower()
        assert "photo" in lowered, reply                 # 先跟客户核对图片
        assert "fixed installation" in lowered, reply    # 说清看到的是固定安装
        assert "rental" not in lowered, reply            # 不能紧接着又问固装还是租赁

    def test_customer_confirmation_marks_fields_confirmed(self):
        from src.vision import resolve_vision_confirmation

        profile = self._vision_profile()
        stats = resolve_vision_confirmation(profile, "yes, that's right")

        assert "environment" in stats["confirmed"]
        assert "purpose" in stats["confirmed"]
        assert profile.sources["environment"] == "confirmed"
        assert profile.sources["purpose"] == "confirmed"
        assert profile.vision_confirmation_pending == []

    def test_customer_correction_uses_customer_value(self):
        """客户纠正 → 以客户说的为准，并留下纠正记录。"""
        from src.agents.sales.nodes.script_generator import script_generator  # noqa: F401
        from src.vision import resolve_vision_confirmation

        profile = self._vision_profile()
        # 客户直接用另一句话纠正（Extractor 会以"客户明说"覆盖图片值）
        slots = {"environment": "outdoor"}
        profile = profile.merge(
            RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        )
        stats = resolve_vision_confirmation(profile, "no, it is outdoor")

        assert profile.environment == "outdoor", "客户说的必须覆盖图片识别"
        assert profile.sources["environment"] == "explicit"
        assert "environment" in stats["corrected"]
        assert any("image said indoor" in note for note in profile.vision_corrections)
        assert profile.vision_confirmation_pending == []

    def test_unrelated_reply_is_accepted_and_not_reasked(self):
        """客户没纠正、也没说"对" → 视为"已核对过、没反对"，不再重复追问同一件事。

        （客户口径：识别结果已经摆在上一条回复里请他核对过了；再问一次就是重复。）
        """
        from src.rag.readiness import check_recommendation_ready
        from src.vision import resolve_vision_confirmation

        profile = self._vision_profile()
        stats = resolve_vision_confirmation(profile, "what is the price?")

        assert stats["confirmed"] == []                      # 客户并没有口头确认
        assert "environment" in stats["accepted"]            # 但也没纠正 → 接受
        assert profile.sources["environment"] == "vision_accepted"
        assert profile.vision_confirmation_pending == []
        # 已经核对过的项不再被 Gate 追问
        assert "environment" not in check_recommendation_ready(profile).missing

    def test_customer_affirmation_still_counts_as_confirmed(self):
        from src.vision import resolve_vision_confirmation

        profile = self._vision_profile()
        resolve_vision_confirmation(profile, "yes, that's right")
        assert profile.sources["environment"] == "confirmed"


class TestApiImagePayload:
    """API 层：图片入参规范化（计划第十五 / 二十一阶段）。"""

    def test_data_field_becomes_data_url(self):
        from src.api import ImageInput, _normalize_images

        result = _normalize_images([ImageInput(data="QUJD", mime_type="image/png")])
        assert result == ["data:image/png;base64,QUJD"]

    def test_url_is_passed_through(self):
        from src.api import ImageInput, _normalize_images

        result = _normalize_images([ImageInput(url="https://example.com/a.jpg")])
        assert result == ["https://example.com/a.jpg"]

    def test_empty_payload_is_ignored(self):
        from src.api import ImageInput, _normalize_images

        assert _normalize_images([ImageInput()]) == []

    def test_too_many_images_are_trimmed(self):
        from src.api import ImageInput, _normalize_images

        images = [ImageInput(data=f"img{index}") for index in range(10)]
        assert len(_normalize_images(images)) == 3

    def test_no_images_returns_empty(self):
        from src.api import _normalize_images

        assert _normalize_images(None) == []


class TestFrontendImageControls:
    """前端图片入口（按钮 / 粘贴 / 拖拽）必须真的存在并被服务出去。"""

    @pytest.fixture(scope="class")
    def static_client(self):
        from fastapi.testclient import TestClient

        from src.api import app

        # 不进入 context manager → 不触发 startup（不需要向量库/大模型）
        return TestClient(app)

    def test_index_has_attach_button(self, static_client):
        html = static_client.get("/static/index.html").text
        assert 'id="attach-btn"' in html, "页面必须有添加图片按钮"
        assert 'id="image-input"' in html
        assert 'id="image-preview"' in html

    def test_index_uses_cache_busted_assets(self, static_client):
        html = static_client.get("/static/index.html").text
        assert "app.js?v=" in html, "静态资源要带版本号，避免浏览器用旧 JS"
        assert "style.css?v=" in html

    def test_app_js_has_image_handlers(self, static_client):
        js = static_client.get("/static/app.js").text
        assert "bindImageInput" in js
        assert "handlePaste" in js, "粘贴图片必须有全局处理"
        assert "addImageUrl" in js
        assert "dragenter" in js, "拖拽图片要支持"


