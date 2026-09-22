"""实测（2026-09-22）：客户答了"只在意质量"，系统却卡住 —— 既不问缺的，也不推荐。

日志证据：

    [Understanding] {'price_preference': {'value': 'quality', 'adopted': True, …}}
    [RecommendationGate] status=CONTINUE_ASKING missing=['installation'] — asking: …
    [Continuation] 客户没答上一问（price_preference）→ 本轮先承接、不问问题（承接 3/3）
    final_response question_count=0

根因：`newly_filled_slots` 来自 `profile.to_facts()`，而它**不包含**
`price_preference` / `content_type` / `budget_level` → 系统以为"价格问题一直没回答"
→ 每轮都压掉问题（改成承接）→ Gate 缺 installation 也不问、推荐也不触发 → 卡死。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import get_conversation_state, reset_conversation_state  # noqa: E402
from src.memory.store import memory  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402
from src.orchestrator import DualAgentOrchestrator  # noqa: E402

CHURCH_SLOTS = {
    "display_type": "LED",
    "environment": "indoor",
    "purpose": "church",
    "installation": "fixed",
    "viewing_distance_m": 3.0,
    "target_width_mm": 3000,
    "target_height_mm": 5000,
    "price_preference": "quality",
    "content_type": "mixed",
}


class _StubSales:
    """Gate 只差 installation → 这一轮要问 installation。"""

    def __init__(
        self,
        question="Is it fixed in place for good, or portable?",
        slot="installation",
        offtopic=False,
    ):
        self.pending_question = question
        self.pending_slot = slot
        # 客户这句话是不是"与需求无关的闲聊"（真实链路由 LLM + 规则在语境里判定）
        self.offtopic = offtopic

    def run(self, session_id, message, has_vision=False, **_kwargs):
        # 真实链路里"待问的那句话"就在回复正文里（这里照做，方便断言问题个数）
        text = (
            "For a church screen at three meters I can line up the right option. "
            f"{self.pending_question}"
        )
        return {
            "intent": "need_query",
            "next_action": "ask",
            "response": text,
            "requirements": {},
            "products": [],
            "pending_question": self.pending_question,
            "pending_slot": self.pending_slot,
            "acknowledgement": "Quality first, that tells me a lot.",
            "offtopic_turn": self.offtopic,
            "speech_act": {"speech_act": "NEW_REQUIREMENT"},
            "dialogue_action": {"action": "ask_only", "priority": 3},
        }


class _StubSolution:
    def run(self, *args, **kwargs):
        return {"answer": "Sure.", "products": []}


def _prepare(session_id: str, *, ack_streak: int = 1):
    memory.clear(session_id)
    memory.mark_first_contact_done(session_id)
    profile = RequirementProfile.from_slots(CHURCH_SLOTS, explicit_keys=set(CHURCH_SLOTS))
    memory.set_requirement_profile(session_id, profile)
    reset_conversation_state(session_id)
    state = get_conversation_state(session_id)
    # 上一轮 AI 问的是"价格还是质量"，客户已经答过（档案里已有 quality）
    state.note_ai_turn(
        action="ask_only",
        question="Should I optimise for the best price, or for the best quality?",
        slot="price_preference",
        response="Should I optimise for the best price, or for the best quality?",
        turn_id="prev-turn",
    )
    state.ack_streak = ack_streak
    return profile


class TestProfileSlotMapCoversAllFields:

    def test_price_preference_and_content_type_are_included(self):
        session_id = "stall-slot-map"
        _prepare(session_id)
        try:
            orch = DualAgentOrchestrator(sales_agent=_StubSales(), solution_agent=_StubSolution())
            slots = orch._profile_slot_map(session_id)
            assert slots.get("price_preference") == "quality"
            assert slots.get("content_type") == "mixed"
            assert slots.get("installation") == "fixed"
            assert slots.get("size"), "尺寸也要映射进来"
        finally:
            memory.clear(session_id)


class TestNoStall:

    def test_answered_price_question_does_not_suppress_the_next_question(self):
        session_id = "stall-recover-1"
        _prepare(session_id)
        try:
            orch = DualAgentOrchestrator(sales_agent=_StubSales(), solution_agent=_StubSolution())
            result = orch.process_message("给我推荐", session_id)
            # 关键：不能只承接（那是卡住的症状），必须问缺的那一项
            assert result["question_count"] == 1, result.get("response")
            assert result["question_slot"] == "installation"
            assert "?" in result["response"]
            assert result.get("suppressed_question") in (None, {})
            assert result["continuation"]["suppress_question"] is False
        finally:
            memory.clear(session_id)

    def test_registry_is_synced_with_the_profile(self):
        session_id = "stall-recover-2"
        _prepare(session_id)
        try:
            orch = DualAgentOrchestrator(sales_agent=_StubSales(), solution_agent=_StubSolution())
            orch.process_message("给我推荐", session_id)
            registry = get_conversation_state(session_id).registry
            assert registry.is_answered("price_preference") is True
            assert registry.is_answered("purpose") is True
        finally:
            memory.clear(session_id)

    def test_unanswered_question_still_chats_first(self):
        """反向用例：客户这轮说的是与需求无关的话 → 先承接（不立刻重复问）。"""
        session_id = "stall-recover-3"
        # 承接额度还没用完（ack_streak=0，上限 1 条）→ 应该先承接一轮
        _prepare(session_id, ack_streak=0)
        # 把价格取向从档案里拿掉 → 客户确实没答过
        slots = dict(CHURCH_SLOTS)
        slots.pop("price_preference")
        memory.set_requirement_profile(
            session_id, RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        )
        try:
            orch = DualAgentOrchestrator(
                sales_agent=_StubSales(offtopic=True), solution_agent=_StubSolution()
            )
            result = orch.process_message("haha i am in nairobi", session_id)
            assert result["question_count"] == 0, result.get("response")
            assert result["continuation"]["suppress_question"] is True
            assert result["continuation"]["reason"] == "customer_off_topic_chat_first"
        finally:
            memory.clear(session_id)
