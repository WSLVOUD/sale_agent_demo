"""
检索语料构建（Phase 2：Model 级）。

所有需要构建向量库 / 稀疏索引 / BM25 索引的入口都统一走这里，避免
"有的入口用 Series 文本、有的入口用 Model 数据" 的不一致。

原则（来自计划文档 Phase 2）：
    一个实际销售 Model = 一个主要 RAG Document，Series 作为 series_id metadata。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def build_retrieval_corpus(data_dir: str | None = None) -> List[Document]:
    """构建 Model 级检索语料（Phase 2 起为唯一产品语料来源）。"""
    from src.config import config
    from src.rag.json_loader import load_canonical_models, load_model_documents

    data_dir = data_dir or config.DATA_DIR
    documents = load_model_documents(data_dir)
    if not documents:
        # 数据缺失时给出明确错误，避免静默退回 Series 级语料造成"看起来还能跑"
        raise RuntimeError(
            f"未能在 {data_dir} 构建 Model 级语料；请先运行 "
            f"`python -m src.rag.json_loader --validate` 检查产品 JSON"
        )
    # Series 数按语料里的 metadata 统计（LED + LCD + IFP 都算）
    series_count = len(
        {
            doc.metadata.get("series_id")
            for doc in documents
            if doc.metadata.get("series_id")
        }
    )
    by_type: Dict[str, int] = {}
    for doc in documents:
        key = str(doc.metadata.get("display_type") or "?")
        by_type[key] = by_type.get(key, 0) + 1
    logger.info(
        "检索语料就绪：%d 个 Model / %d 个 Series（Model 级）%s",
        len(documents), series_count, by_type,
    )
    return documents


def corpus_as_dicts(documents: List[Document]) -> List[Dict[str, Any]]:
    """转成 sparse / BM25 需要的 ``{"id", "text", "metadata"}`` 结构。"""
    items: List[Dict[str, Any]] = []
    for index, doc in enumerate(documents or []):
        metadata = dict(getattr(doc, "metadata", {}) or {})
        items.append({
            "id": metadata.get("chunk_id") or str(index),
            "text": getattr(doc, "page_content", ""),
            "metadata": metadata,
        })
    return items


def corpus_summary(documents: List[Document]) -> Dict[str, Any]:
    """语料概况，用于启动日志、/health 与诊断接口。"""
    metadatas = [getattr(doc, "metadata", {}) or {} for doc in documents or []]
    return {
        "documents": len(metadatas),
        "series": len({m.get("series_id") for m in metadatas if m.get("series_id")}),
        "indoor": sum(1 for m in metadatas if m.get("indoor") is True),
        "outdoor": sum(1 for m in metadatas if m.get("outdoor") is True),
        "rental": sum(1 for m in metadatas if m.get("is_rental") is True),
        "fixed": sum(1 for m in metadatas if m.get("is_rental") is False),
        "level": "model" if all(m.get("level") == "model" for m in metadatas) else "mixed",
    }
