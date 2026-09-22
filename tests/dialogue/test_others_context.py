"""客户口径（2026-09-22）：自由问答（others / product_question）也要看得见语境。

实测记录：

    🤖 … Shall I prepare the quotation?
    👤 yew
    🤖 Sorry, I'm not sure I caught that…          ← 打错字，没认出来
    👤 yes
    🤖 Sure, go ahead, what would you like to know? ← 没绑到"要不要出报价"
    👤 give me quatatio
    🤖 … A few things I'd need from you … The exact installation type … The final dimensions …
                                                     ↑ 又问已经答过的安装方式 / 尺寸

根因：这条路径的回答节点（`others_node`）拿不到任何历史，只拿到"这一句话 +
一份残缺的已确认需求清单 + 检索片段"：

  · 提示词里根本没有对话历史 → 客户回 "yes" 时它不知道上一句在问什么；
  · 已确认需求来自 legacy 投影字典（indoor / is_rental / size），而护栏只认
    environment / installation / target_width_mm → 环境、安装方式、尺寸全丢。

这里把两件事锁住：① 最近 50 条对话必须进提示词；② 已确认需求（含 legacy 来源）
必须完整可见。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.solution.nodes import others as others_module  # noqa: E402
from src.memory.store import memory  # noqa: E402
from src.models.legacy_adapter import profile_to_solution_requirement  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.model_guard import (  # noqa: E402
    requirement_lines,
    requirement_summary,
    retrieval_filters,
)

SLOTS = {
    "display_type": "LED",
    "environment": "indoor",
    "installation": "fixed",
    "purpose": "church",
    "viewing_distance_m": 5,
    "target_width_mm": 3000,
    "target_height_mm": 5000,
}

DIALOGUE = [
    {"role": "user", "content": "i need a led display"},
    {"role": "assistant", "content": "An LED display gives us plenty to work with. Is this indoors or outdoors?"},
    {"role": "user", "content": "indoor, 3*5, permanent"},
    {
        "role": "assistant",
        "content": "For your 3m x 5m indoor permanent screen, TW11-3216-P3.0 fits well. Shall I prepare the quotation?",
    },
    {"role": "user", "content": "yew"},
    {"role": "assistant", "content": "Sorry, I'm not sure I caught that, could you say a bit more?"},
]


class _Captured:
    def __init__(self):
        self.prompts = []

    def __call__(self, temperature: float = 0.7):  # pragma: no cover - 简单工厂
        captured = self

        class _LLM:
            def invoke(self, prompt):
                content = getattr(prompt, "content", None)
                if content is None:
                    content = str(prompt)
                captured.prompts.append(str(content))

                class _Response:
                    content = "Sure — I'll put the quotation together right away."

                return _Response()

        return _LLM()


def _profile() -> RequirementProfile:
    return RequirementProfile.from_slots(SLOTS, explicit_keys=set(SLOTS))


def _run_node(monkeypatch, session_id: str, message: str, *, profile=None):
    captured = _Captured()
    monkeypatch.setattr(others_module, "get_llm", captured)
    state = {
        "session_id": session_id,
        "messages": [{"role": "user", "content": message}],
        "requirement": profile_to_solution_requirement(profile) if profile else {},
        "requirement_profile": profile,
        "hybrid_search": None,
    }
    others_module.others_node(state)
    assert captured.prompts, "others_node 必须真的调用 LLM"
    return captured.prompts[-1]


class TestOthersNodeCarriesContext:

    def test_prompt_includes_the_recent_dialogue(self, monkeypatch):
        session_id = "others-ctx-1"
        memory.clear(session_id)
        memory.extend(session_id, DIALOGUE)
        try:
            prompt = _run_node(monkeypatch, session_id, "yes", profile=_profile())
            assert "Recent conversation" in prompt
            # 上一句 AI 的问话（客户的 "yes" 是在回答它）必须看得见
            assert "Shall I prepare the quotation?" in prompt
            # 更早的上下文（尺寸 / 室内 / 固装）也要在
            assert "indoor, 3*5, permanent" in prompt
            assert "i need a led display" in prompt
            # 顺序：旧的在前、新的在后（模型据此判断"最后一句在问什么"）
            assert prompt.index("i need a led display") < prompt.index("Shall I prepare the quotation?")
        finally:
            memory.clear(session_id)

    def test_prompt_has_confirmed_requirements_from_profile(self, monkeypatch):
        session_id = "others-ctx-2"
        memory.clear(session_id)
        try:
            prompt = _run_node(monkeypatch, session_id, "give me quatatio", profile=_profile())
            confirmed = prompt.split("Confirmed customer requirements")[1]
            assert "indoor" in confirmed
            assert "fixed" in confirmed, "安装方式必须可见（否则又会问 fixed 还是 rental）"
            assert "3000" in confirmed and "5000" in confirmed, "尺寸必须可见"
            assert "5.0" in confirmed, "观看距离必须可见"
        finally:
            memory.clear(session_id)

    def test_prompt_tells_the_model_not_to_reask_known_facts(self, monkeypatch):
        session_id = "others-ctx-3"
        memory.clear(session_id)
        try:
            prompt = _run_node(monkeypatch, session_id, "yes", profile=_profile())
            assert "never re-ask" in prompt.lower() or "Never re-ask" in prompt
            assert "do not ask the\n  customer for it again" in prompt
        finally:
            memory.clear(session_id)


class TestLegacyRequirementsAreUnderstood:
    """legacy 投影字典（indoor / is_rental / size）也必须被护栏认出来。"""

    def test_summary_covers_environment_installation_and_size(self):
        summary = requirement_summary(profile_to_solution_requirement(_profile()))
        assert summary["environment"] == "indoor"
        assert summary["installation"] == "fixed"
        assert summary["size"]
        assert summary["viewing_distance"]

    def test_requirement_lines_are_complete(self):
        lines = "\n".join(requirement_lines(profile_to_solution_requirement(_profile())))
        assert "indoor" in lines and "fixed" in lines
        assert "3000" in lines or "3" in lines

    def test_retrieval_filters_keep_environment_and_installation(self):
        filters = retrieval_filters(profile_to_solution_requirement(_profile()))
        assert filters.get("indoor") is True
        assert filters.get("is_rental") is False
        assert filters.get("display_type") == "LED"


class _StubSalesOthers:
    """Sales 把这一轮交给 Solution（next_action=others），只留一个占位符。"""

    def run(self, session_id, message, has_vision=False, **_kwargs):
        return {
            "intent": "others",
            "next_action": "others",
            "response": "Sure.",
            "requirements": {},
            "products": [],
            "pending_question": "",
            "pending_slot": "",
            "speech_act": {},
            "dialogue_action": {},
        }


class _StubSolutionAnswer:
    def run(self, *args, **kwargs):
        return {
            "answer": "Sure — I'll put the quotation together and send it right away.",
            "products": [],
        }


class TestPlaceholderHistoryIsReplaced:
    """历史里不能留 Sales 的占位符（"Sure."）——那是下一轮语境的污染源。"""

    def test_solution_answer_replaces_the_placeholder_in_history(self):
        from src.orchestrator import DualAgentOrchestrator

        session_id = "others-history-1"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        # 模拟真实链路：Sales runner 已经把这一轮的"占位符回复"写进 memory
        memory.add(session_id, "user", "yes")
        memory.add(session_id, "assistant", "Sure.")
        orch = DualAgentOrchestrator(
            sales_agent=_StubSalesOthers(), solution_agent=_StubSolutionAnswer()
        )
        try:
            orch.process_message("yes", session_id)
            assistants = [
                str(item.get("content") or "")
                for item in memory.get_history(session_id)
                if str(item.get("role") or "") == "assistant"
            ]
            assert assistants, "历史里必须有 assistant 消息"
            assert assistants[-1] != "Sure.", "占位符要被真正的答复替换"
            assert "quotation" in assistants[-1].lower()
        finally:
            memory.clear(session_id)

    def test_store_helper_only_touches_the_last_assistant_message(self):
        session_id = "others-history-2"
        memory.clear(session_id)
        memory.extend(
            session_id,
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": "Sure."},
            ],
        )
        try:
            assert memory.replace_last_assistant(session_id, "Here is the quotation.") is True
            history = [item["content"] for item in memory.get_history(session_id)]
            assert history == ["hi", "Hello!", "yes", "Here is the quotation."]
            # 已经一样 → 不再重复替换
            assert memory.replace_last_assistant(session_id, "Here is the quotation.") is False
            # 空文本不动历史
            assert memory.replace_last_assistant(session_id, "   ") is False
        finally:
            memory.clear(session_id)
