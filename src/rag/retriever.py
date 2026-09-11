"""
向量检索模块
基于向量相似度进行检索
"""
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
import logging

from src.core.embeddings import get_embeddings

logger = logging.getLogger(__name__)


def retrieve(
    vectorstore: Chroma,
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve documents using vector similarity.
    
    Args:
        vectorstore: Chroma vector store
        query: Search query
        top_k: Number of results
        filters: Metadata filters
        
    Returns:
        List of retrieved documents with scores
    """
    if not vectorstore:
        return []
    
    try:
        # Build filter clause
        # Handle both simple values {"key": value} and comparison dicts {"key": {"$eq"/"$lte"/"$gte": value}}
        if filters:
            clauses = []
            for key, value in filters.items():
                if value is None:
                    continue
                # Check if value is already a comparison dict
                if isinstance(value, dict) and any(op in value for op in ["$eq", "$lte", "$gte", "$lt", "$gt"]):
                    clauses.append({key: value})
                else:
                    clauses.append({key: {"$eq": value}})
            where_clause = clauses[0] if len(clauses) == 1 else {"$and": clauses} if clauses else None
        else:
            where_clause = None
        
        # Query
        results = vectorstore.similarity_search_with_score(query, k=top_k, filter=where_clause)
        
        documents = []
        for doc, score in results:
            similarity = 1 / (1 + score)
            documents.append({
                "id": doc.metadata.get("chunk_id", ""),
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": float(similarity)
            })
        
        return documents
        
    except Exception as e:
        logger.error(f"Vector retrieval error: {e}")
        return []
