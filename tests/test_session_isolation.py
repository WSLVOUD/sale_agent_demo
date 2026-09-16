"""
跨会话隔离测试（客户端反馈：不重启服务器时，新对话会聊到别的对话框的内容）。

根因：`RequirementExtractor._semantic_cache` 是**进程级**缓存，而且只用消息文字做键。
但语义理解是"结合当前对话上下文 + 已收集需求"由 LLM 得出的 ——
同一条 "5m" 在 A 对话里是"教堂室内"，在 B 对话里可能什么都不是。
于是新会话说了一句和旧会话相同的话，就会凭空继承旧会话的 purpose / environment，
而且只有重启进程才会消失。

本文件锁定三条规则：
  1. 语义缓存必须按 session_id 隔离（新会话不会被别的会话污染）；
  2. 同一会话内仍然允许复用（否则会多调一次 LLM）；
  3. 没有 session_id 时不缓存、不复用（宁可不复用，也不能串台）。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.requirement_extractor import RequirementExtractor  # noqa: E402
from src.memory.store import MAX_SESSIONS, MemoryStore  # noqa: E402


@pytest.fixture(autouse=True)
def clean_semantic_cache():
    """每条用例前清空进程级语义缓存，保证用例之间互不影响。"""
    RequirementExtractor._semantic_cache.clear()
    yield
    RequirementExtractor._semantic_cache.clear()


class TestSemanticCacheIsolation:
    """语义缓存不能跨会话复用。"""

    def test_new_session_does_not_inherit_previous_session_context(self):
        extractor = RequirementExtractor()

        # 会话 A：Sales 的 LLM 结合上下文，把 "5m" 理解成"教堂 / 室内"
        extractor.extract(
            "5m",
            semantic_override={"purpose": "church", "environment": "indoor"},
            session_id="session-A",
        )

        # 会话 B：全新对话，同样一句 "5m"（LLM 不可用时走缓存分支）
        profile_b = extractor.extract("5m", use_llm=False, session_id="session-B")

        assert profile_b.purpose is None, "新会话不应继承别的会话的 purpose"
        assert profile_b.environment is None, "新会话不应继承别的会话的 environment"

    def test_same_session_still_reuses_cache(self):
        extractor = RequirementExtractor()
        extractor.extract(
            "5m",
            semantic_override={"purpose": "church"},
            session_id="session-A",
        )

        profile = extractor.extract("5m", use_llm=False, session_id="session-A")

        assert profile.purpose == "church", "同一会话内仍应复用本轮语义结果"

    def test_without_session_id_nothing_is_cached(self):
        extractor = RequirementExtractor()
        extractor.extract("5m", semantic_override={"purpose": "church"})

        profile = extractor.extract("5m", use_llm=False)

        assert profile.purpose is None, "没有 session_id 时不应读写缓存"
        assert RequirementExtractor._semantic_cache == {}

    def test_clear_session_semantics_only_affects_one_session(self):
        extractor = RequirementExtractor()
        extractor.extract("5m", semantic_override={"purpose": "church"}, session_id="A")
        extractor.extract("5m", semantic_override={"purpose": "stadium"}, session_id="B")

        extractor.clear_session_semantics("A")

        assert extractor.extract("5m", use_llm=False, session_id="A").purpose is None
        assert extractor.extract("5m", use_llm=False, session_id="B").purpose == "stadium"

    def test_cache_key_includes_session(self):
        assert RequirementExtractor._cache_key("5m", "A") == "A::5m"
        assert RequirementExtractor._cache_key("5m", "A") != RequirementExtractor._cache_key("5m", "B")
        assert RequirementExtractor._cache_key("5m", "") == ""


class TestMemorySessionCap:
    """长时间运行（不重启）时，会话不能无限堆积。"""

    def test_old_sessions_are_evicted(self):
        store = MemoryStore()
        for index in range(MAX_SESSIONS + 10):
            store.add(f"session-{index}", "user", "hello")

        assert len(store._sessions) <= MAX_SESSIONS
        assert "session-0" not in store._sessions, "最久未使用的会话应被淘汰"
        assert f"session-{MAX_SESSIONS + 9}" in store._sessions

    def test_active_session_is_kept(self):
        store = MemoryStore()
        store.add("hot", "user", "hello")
        for index in range(MAX_SESSIONS):
            store.add(f"cold-{index}", "user", "hello")
            store.add("hot", "assistant", "hi")   # 持续活跃 → 不会被淘汰

        assert "hot" in store._sessions


class _ContextAwareSalesLLM:
    """模拟真实 LLM 的行为：语义结果**取决于当前对话上下文**（这正是串台的来源）。"""

    class _Response:
        def __init__(self, content: str):
            self.content = content

    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, messages, *args, **kwargs):
        # 只看**最后一条**（客户消息 + 已收集需求 + 当前对话），
        # 系统提示词里本来就有 church/会议室 这些示例词，不能当上下文用
        parts = messages if isinstance(messages, list) else [messages]
        text = str(getattr(parts[-1], "content", parts[-1])).lower()
        if "church" in text:
            return self._Response(
                '{"usage": "church", "purpose": "church", '
                '"additional_requirements": [], "ack": ""}'
            )
        return self._Response(
            '{"usage": null, "additional_requirements": [], "ack": ""}'
        )


class _EmptyExtractorLLM:
    """Extractor 自己的语义补全（缓存未命中时）——返回空，避免真的调网络。"""

    class _Response:
        content = "{}"

    def invoke(self, *args, **kwargs):
        return self._Response()


class TestSalesTurnCrossSessionIsolation:
    """端到端（Sales 节点级）：A 会话聊教堂，B 会话说同一句话，不能继承教堂。"""

    @pytest.fixture
    def sales_node(self, monkeypatch):
        import importlib

        module = importlib.import_module("src.agents.sales.nodes.requirement")
        monkeypatch.setattr(module, "ChatOpenAI", _ContextAwareSalesLLM)
        # 让 Extractor 走真实的"查缓存 → 未命中再调 LLM"逻辑，只把 LLM 换成假的
        extractor = RequirementExtractor()
        monkeypatch.setattr(extractor, "llm", _EmptyExtractorLLM(), raising=False)
        return module

    def _turn(self, module, session_id, message, requirements=None, profile=None):
        state = {
            "messages": [{"role": "user", "content": message}],
            "current_message": message,
            "session_id": session_id,
            "intent": "need_query",
            "next_action": "ask",
            "requirements": dict(requirements or {}),
            "additional_requirements": [],
            "should_generate_solution": False,
            "response": "",
            "pending_question": "",
            "pending_slot": "",
        }
        if profile is not None:
            state["requirement_profile"] = profile
        return module.requirement_mining(state)

    def test_second_session_does_not_inherit_first_session_purpose(self, sales_node):
        # 会话 A：先聊教堂，再报视距
        first = self._turn(sales_node, "session-A", "indoor LED screen for a church")
        second = self._turn(
            sales_node,
            "session-A",
            "5m",
            requirements=first["requirements"],
            profile=first["requirement_profile"],
        )
        profile_a = second["requirement_profile"]
        assert profile_a.purpose == "church", "同一会话内应当复用上下文语义结果"

        # 会话 B：全新对话，同样一句 "5m"
        other = self._turn(sales_node, "session-B", "5m")
        profile_b = other["requirement_profile"]

        assert profile_b.purpose is None, "新会话不能继承别的会话的 purpose"
        assert profile_b.environment is None

    def test_cache_entries_are_namespaced_by_session(self, sales_node):
        first = self._turn(sales_node, "session-A", "indoor LED screen for a church")
        self._turn(
            sales_node,
            "session-A",
            "5m",
            requirements=first["requirements"],
            profile=first["requirement_profile"],
        )

        keys = list(RequirementExtractor._semantic_cache)
        assert keys, "同一会话内应当写入语义缓存"
        assert all(key.startswith("session-A::") for key in keys), keys
