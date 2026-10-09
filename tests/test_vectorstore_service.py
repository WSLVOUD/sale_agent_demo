from __future__ import annotations

from pathlib import Path

from src.rag.loader import ENVIRONMENT_METADATA_VERSION
from src.vectorstore_service import load_or_rebuild_vectorstore, validate_vectorstore


def _metadata() -> dict:
    return {
        "indoor": True,
        "outdoor": False,
        "display_type": "LED",
        "environment_metadata_version": ENVIRONMENT_METADATA_VERSION,
        "product_category": "led",
        "gob": False,
        "flexible": False,
        "modules_per_cabinet": 1,
        "level": "model",
    }


class _Collection:
    def __init__(self, metadatas):
        self.metadatas = metadatas

    def count(self):
        return len(self.metadatas)

    def get(self, *, include):
        assert include == ["metadatas"]
        return {"metadatas": self.metadatas}


class _VectorStore:
    def __init__(self, metadatas):
        self._collection = _Collection(metadatas)


def _persisted_store(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "chroma.sqlite3").write_bytes(b"store")


def test_validate_vectorstore_rejects_outdated_metadata(monkeypatch, tmp_path):
    import src.vectorstore_service as service

    monkeypatch.setattr(
        service, "load_vectorstore", lambda **_kwargs: _VectorStore([{"level": "chunk"}])
    )

    loaded, count, reasons = validate_vectorstore(tmp_path, expected_count=1)

    assert loaded is not None
    assert count == 1
    assert any("metadata 是旧版本" in reason for reason in reasons)


def test_load_or_rebuild_reuses_valid_fallback_store(monkeypatch, tmp_path):
    import src.core.embeddings as embeddings
    import src.vectorstore_service as service

    primary = tmp_path / "vectorstore"
    fallback = tmp_path / "vectorstore_rebuilt"
    _persisted_store(primary)
    _persisted_store(fallback)
    primary_store = _VectorStore([{"level": "old"}])
    fallback_store = _VectorStore([_metadata()])
    released = []

    monkeypatch.setattr(
        service,
        "load_vectorstore",
        lambda persist_dir: {
            str(primary): primary_store,
            str(fallback): fallback_store,
        }[persist_dir],
    )
    monkeypatch.setattr(
        embeddings, "_release_vectorstore_handles", lambda: released.append(True)
    )
    monkeypatch.setattr(
        service,
        "recreate_vectorstore",
        lambda _documents: (_ for _ in ()).throw(AssertionError("must reuse fallback")),
    )

    vectorstore, active_dir = load_or_rebuild_vectorstore(
        primary, documents=[object()], expected_count=1
    )

    assert vectorstore is fallback_store
    assert active_dir == str(fallback)
    assert released == [True]


def test_load_or_rebuild_rebuilds_when_no_candidate_exists(monkeypatch, tmp_path):
    import src.vectorstore_service as service

    primary = tmp_path / "vectorstore"
    rebuilt_store = _VectorStore([_metadata()])
    rebuilt_store._active_persist_dir = str(tmp_path / "active-vectorstore")
    documents = [object()]
    rebuild_calls = []
    load_calls = []

    def rebuild(received_documents):
        rebuild_calls.append(received_documents)
        return rebuilt_store

    def load(persist_dir):
        load_calls.append(persist_dir)
        return rebuilt_store

    monkeypatch.setattr(service, "recreate_vectorstore", rebuild)
    monkeypatch.setattr(service, "load_vectorstore", load)

    vectorstore, active_dir = load_or_rebuild_vectorstore(
        primary, documents=documents, expected_count=1
    )

    assert vectorstore is rebuilt_store
    assert active_dir == rebuilt_store._active_persist_dir
    assert rebuild_calls == [documents]
    assert load_calls == [rebuilt_store._active_persist_dir]
