"""
Embedding 和向量存储模块
处理向量存储和检索
"""
import shutil
import gc
import logging
import threading
from pathlib import Path
import os
from typing import List, Optional, Dict, Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.config import config

logger = logging.getLogger(__name__)

# Phase 11: 全局缓存（线程安全）— 避免每次请求都重新加载 Embedding 模型
_embeddings_singleton: Optional[HuggingFaceEmbeddings] = None
_embeddings_lock = threading.Lock()

# Phase 11: 向量库 + 产品 JSON 常驻内存的全局缓存
_vectorstore_singleton: Optional[Chroma] = None
_vectorstore_lock = threading.Lock()


def get_embeddings():
    """
    Load BGE-M3 from the local models directory. Never hits HuggingFace Hub.

    Phase 11: 使用全局缓存，只在第一次调用时真正加载模型。
    """
    global _embeddings_singleton
    if _embeddings_singleton is not None:
        return _embeddings_singleton

    with _embeddings_lock:
        if _embeddings_singleton is not None:
            # Double-check after acquiring lock
            return _embeddings_singleton

        model_path = config.EMBEDDING_MODEL_PATH
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        logger.info("Loading local embedding model from %s", model_path)
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )
        _embeddings_singleton = embeddings
        return _embeddings_singleton


def reset_embeddings_cache():
    """清除 Embedding 模型缓存（用于测试或模型热更新）。"""
    global _embeddings_singleton
    with _embeddings_lock:
        _embeddings_singleton = None


def get_cached_vectorstore(persist_dir: str = None) -> Optional[Chroma]:
    """Phase 11: 获取缓存的向量库（避免每次请求都重新连接 ChromaDB）。"""
    global _vectorstore_singleton
    return _vectorstore_singleton


def set_cached_vectorstore(vectorstore: Chroma):
    """Phase 11: 设置缓存的向量库。"""
    global _vectorstore_singleton
    with _vectorstore_lock:
        _vectorstore_singleton = vectorstore


def reset_vectorstore_cache():
    """Phase 11: 清除向量库缓存（例如：向量库重新构建后）。"""
    global _vectorstore_singleton
    with _vectorstore_lock:
        _vectorstore_singleton = None


def create_vectorstore(documents: List[Document], persist_dir: str = None, collection_name: Optional[str] = None) -> Chroma:
    """
    Create a new vector store from documents.

    Args:
        documents: List of documents to embed
        persist_dir: Directory to persist the vector store
        collection_name: Optional explicit Chroma collection name.

    Returns:
        Chroma vector store
    """
    persist_dir = persist_dir or config.VECTORSTORE_DIR
    logger.info(f"Creating vector store at {persist_dir} collection={collection_name}")

    embeddings = get_embeddings()

    chroma_kwargs: Dict[str, Any] = {"embedding": embeddings, "persist_directory": persist_dir}
    if collection_name:
        chroma_kwargs["collection_name"] = collection_name

    vectorstore = Chroma.from_documents(documents=documents, **chroma_kwargs)

    vectorstore.persist()
    logger.info(f"Vector store created with {len(documents)} documents")

    return vectorstore


def load_vectorstore(persist_dir: str = None, collection_name: Optional[str] = None, use_cache: bool = True) -> Chroma:
    """
    Load an existing vector store.

    Args:
        persist_dir: Directory of the vector store
        collection_name: Optional explicit collection name.
        use_cache: 是否复用进程内的 Chroma 实例（Phase 11，默认 True）

    Returns:
        Chroma vector store
    """
    persist_dir = persist_dir or config.VECTORSTORE_DIR

    # Phase 11: 命中缓存则直接复用
    if use_cache:
        cached = get_cached_vectorstore(persist_dir)
        if cached is not None and getattr(cached, "_active_persist_dir", persist_dir) == persist_dir:
            logger.debug("Vector store loaded from cache")
            return cached

    logger.info(f"Loading vector store from {persist_dir} collection={collection_name}")

    embeddings = get_embeddings()

    chroma_kwargs: Dict[str, Any] = {"embedding_function": embeddings, "persist_directory": persist_dir}
    if collection_name:
        chroma_kwargs["collection_name"] = collection_name

    vectorstore = Chroma(**chroma_kwargs)
    setattr(vectorstore, "_active_persist_dir", persist_dir)
    set_cached_vectorstore(vectorstore)

    logger.info("Vector store loaded successfully")
    return vectorstore


def get_retriever(vectorstore: Chroma, top_k: int = None) -> any:
    """
    Get a retriever from the vector store.
    
    Args:
        vectorstore: Chroma vector store
        top_k: Number of documents to retrieve
        
    Returns:
        Retriever instance
    """
    top_k = top_k or config.TOP_K
    return vectorstore.as_retriever(
        search_kwargs={"k": top_k}
    )


def recreate_vectorstore(documents: List[Document], persist_dir: str = None, collection_name: Optional[str] = None) -> Chroma:
    """
    Recreate the vector store on disk.
    """
    persist_dir = persist_dir or config.VECTORSTORE_DIR
    target = Path(persist_dir)
    target_parent = target.parent
    target_parent.mkdir(parents=True, exist_ok=True)
    logger.info("Recreating vector store at %s with %d documents", persist_dir, len(documents))

    fallback = target_parent / f"{target.name}_rebuilt"
    if fallback.exists():
        shutil.rmtree(fallback, ignore_errors=True)

    sqlite_path = target / "chroma.sqlite3"
    can_reuse_target = (not sqlite_path.exists()) or _try_remove(sqlite_path)

    if can_reuse_target:
        if target.exists():
            for entry in target.iterdir():
                if entry == sqlite_path:
                    continue
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    try:
                        entry.unlink()
                    except OSError:
                        pass
        gc.collect()
        instance = _build_vectorstore(documents, str(target), collection_name)
        setattr(instance, "_active_persist_dir", str(target))
        return instance

    # Fallback for locked collection
    persist_path = Path(str(target) + "_rebuilt")
    logger.warning("Vector store directory %s is locked; building rebuilt collection at %s", target, persist_path)
    gc.collect()
    if persist_path.exists():
        shutil.rmtree(persist_path, ignore_errors=True)
    instance = _build_vectorstore(documents, str(persist_path), collection_name)
    setattr(instance, "_active_persist_dir", str(persist_path))
    return instance


def _try_remove(path: Path) -> bool:
    """Try to delete ``path``, swallowing Windows file-lock errors."""
    try:
        path.unlink()
        return True
    except (OSError, PermissionError) as error:
        logger.warning("Could not delete locked file %s: %s", path, error)
        return False


def _build_vectorstore(documents: List[Document], persist_dir: str, collection_name: Optional[str] = None) -> Chroma:
    """Create a Chroma collection at ``persist_dir`` and persist it eagerly."""
    embeddings = get_embeddings()
    chroma_kwargs: Dict[str, Any] = {"embedding": embeddings, "persist_directory": persist_dir}
    if collection_name:
        chroma_kwargs["collection_name"] = collection_name
    vectorstore = Chroma.from_documents(documents=documents, **chroma_kwargs)
    vectorstore.persist()
    return vectorstore
