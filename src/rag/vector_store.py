"""向量存储模块"""
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Optional
import logging

from src.core.embeddings import get_embeddings, recreate_vectorstore

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """向量存储管理器 - 封装Chroma向量数据库操作"""

    def __init__(self, persist_dir: str, collection_name: Optional[str] = None):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._vectorstore: Optional[Chroma] = None

    @property
    def vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            self._vectorstore = load_vectorstore(self.persist_dir, self.collection_name)
        return self._vectorstore

    def as_retriever(self, top_k: int = 5):
        return get_retriever(self.vectorstore, top_k)

    @classmethod
    def create(cls, documents: List[Document], persist_dir: str, collection_name: Optional[str] = None) -> "VectorStoreManager":
        """Create a new vector store and return the manager."""
        create_vectorstore(documents, persist_dir, collection_name)
        return cls(persist_dir, collection_name)


def create_vectorstore(
    documents: List[Document],
    persist_dir: str,
    collection_name: Optional[str] = None
) -> Chroma:
    """Create a new vector store from documents."""
    logger.info(f"Creating vector store at {persist_dir}")
    return recreate_vectorstore(documents, persist_dir, collection_name)


def load_vectorstore(
    persist_dir: str,
    collection_name: Optional[str] = None
) -> Chroma:
    """Load an existing vector store."""
    logger.info(f"Loading vector store from {persist_dir}")
    embeddings = get_embeddings()
    chroma_kwargs = {"embedding_function": embeddings, "persist_directory": persist_dir}
    if collection_name:
        chroma_kwargs["collection_name"] = collection_name
    return Chroma(**chroma_kwargs)


def get_retriever(vectorstore: Chroma, top_k: int = 5):
    """Get a retriever from the vector store."""
    return vectorstore.as_retriever(search_kwargs={"k": top_k})
