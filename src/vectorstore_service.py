"""Validate, reuse, or rebuild the active vector-store directory."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Sequence

from src.core.embeddings import load_vectorstore, recreate_vectorstore
from src.rag.loader import ENVIRONMENT_METADATA_VERSION

logger = logging.getLogger(__name__)

_REQUIRED_METADATA = {
    "indoor",
    "outdoor",
    "display_type",
    "environment_metadata_version",
    "product_category",
    "gob",
    "flexible",
    "modules_per_cabinet",
}


def validate_vectorstore(store_dir: Path, expected_count: int):
    """Return the loaded store, record count, and reasons it should be rebuilt."""
    try:
        loaded = load_vectorstore(persist_dir=str(store_dir))
    except Exception as error:
        return None, 0, [f"加载失败: {error}"]

    try:
        collection = loaded._collection
        count = collection.count()
        metadatas = collection.get(include=["metadatas"]).get("metadatas", [])
    except Exception as error:
        return loaded, 0, [f"读取失败: {error}"]

    reasons: list[str] = []
    if count == 0:
        reasons.append("没有任何记录")
    if len(metadatas) != count:
        reasons.append(f"metadata 数量({len(metadatas)})与记录数({count})不一致")
    if count != expected_count:
        reasons.append(f"记录数 {count} != 期望 {expected_count}")
    invalid = [
        metadata
        for metadata in metadatas
        if not metadata
        or not _REQUIRED_METADATA.issubset(metadata)
        or not isinstance(metadata.get("indoor"), bool)
        or not isinstance(metadata.get("outdoor"), bool)
        or metadata.get("display_type") not in {"LED", "LCD", "IFP"}
        or metadata.get("environment_metadata_version")
        != ENVIRONMENT_METADATA_VERSION
        or metadata.get("level") != "model"
    ]
    if invalid:
        reasons.append(f"{len(invalid)} 条记录的 metadata 是旧版本")
    return loaded, count, reasons


def load_or_rebuild_vectorstore(
    primary_dir: str | Path,
    documents: Sequence[Any],
    expected_count: int,
) -> tuple[Any, str]:
    """Reuse the first valid primary/fallback store, otherwise rebuild it."""
    primary = Path(primary_dir)
    candidate_dirs = [primary, Path(str(primary) + "_rebuilt")]
    vectorstore = None
    collection_count = 0

    for candidate in candidate_dirs:
        sqlite_path = candidate / "chroma.sqlite3"
        if not sqlite_path.exists() or os.path.getsize(sqlite_path) == 0:
            continue

        loaded, collection_count, reasons = validate_vectorstore(
            candidate, expected_count
        )
        if loaded is not None and not reasons:
            vectorstore = loaded
            active_dir = str(candidate)
            if candidate != primary:
                logger.warning(
                    "主向量库不可用 → 直接复用已重建好的向量库：%s（%d 条记录）",
                    candidate,
                    collection_count,
                )
            else:
                logger.info(
                    "Existing vector store has valid v%s metadata: %d records",
                    ENVIRONMENT_METADATA_VERSION,
                    collection_count,
                )
            break

        logger.warning(
            "向量库 %s 需要重建：%s", candidate, "; ".join(reasons) or "未知原因"
        )
        from src.core.embeddings import _release_vectorstore_handles

        _release_vectorstore_handles()

    if vectorstore is None:
        logger.warning("Vector store needs rebuild: expected=%d chunks", expected_count)
        vectorstore = recreate_vectorstore(list(documents))
        active_dir = getattr(vectorstore, "_active_persist_dir", str(primary))
        logger.info("Vector store rebuilt and persisted at %s", active_dir)
        try:
            vectorstore = load_vectorstore(persist_dir=active_dir)
        except Exception as error:
            logger.warning("Could not reload vectorstore after rebuild: %s", error)
    return vectorstore, str(active_dir)
