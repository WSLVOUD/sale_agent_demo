"""v2.7 Phase 2（§5.2）：MessageStore 必须可替换（内存 / SQLite）且业务接口不变。"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.input.message_deduplicator import MessageDeduplicator  # noqa: E402
from src.input.message_store import (  # noqa: E402
    KIND_MESSAGE_ID,
    InMemoryMessageStore,
    SQLiteMessageStore,
    build_message_store,
)


class TestInMemoryMessageStore:

    def test_add_is_idempotent(self):
        store = InMemoryMessageStore()
        assert store.add("m1", kind=KIND_MESSAGE_ID) is True
        assert store.add("m1", kind=KIND_MESSAGE_ID) is False
        assert store.is_seen("m1") is True

    def test_forget_allows_retry(self):
        store = InMemoryMessageStore()
        store.add("m1")
        store.forget("m1")
        assert store.is_seen("m1") is False

    def test_ttl_expires(self):
        import time

        store = InMemoryMessageStore()
        store.add("fp", kind="fingerprint")
        assert store.is_seen("fp", kind="fingerprint", ttl=0.0) is True
        time.sleep(0.02)
        assert store.is_seen("fp", kind="fingerprint", ttl=0.001) is False


class TestSQLiteMessageStore:

    def test_survives_a_new_instance(self, tmp_path):
        """§5.2：换持久化实现不能改业务接口 —— 重启后仍然记得处理过的消息。"""
        path = str(tmp_path / "messages.sqlite")
        first = SQLiteMessageStore(path)
        assert first.add("m100", kind=KIND_MESSAGE_ID) is True

        second = SQLiteMessageStore(path)
        assert second.is_seen("m100") is True
        assert second.add("m100") is False
        second.forget("m100")
        assert second.is_seen("m100") is False

    def test_build_message_store_falls_back_to_memory(self, monkeypatch):
        monkeypatch.delenv("LED_RAG_MESSAGE_STORE_PATH", raising=False)
        assert isinstance(build_message_store(), InMemoryMessageStore)


class TestDeduplicatorUsesTheStore:

    def test_deduplicator_works_against_any_store(self, tmp_path):
        store = SQLiteMessageStore(str(tmp_path / "dedup.sqlite"))
        dedup = MessageDeduplicator(store=store)
        assert dedup.check("s1", message_id="m1", text="hi").is_duplicate is False
        dedup.remember("s1", message_id="m1", text="hi")
        assert dedup.check("s1", message_id="m1", text="hi").is_duplicate is True
        dedup.forget(message_id="m1")
        assert dedup.check("s1", message_id="m1", text="hi").is_duplicate is True, (
            "指纹兜底仍然会命中（同内容重发）"
        )
