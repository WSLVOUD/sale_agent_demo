"""
稀疏检索模块 - 基于 BGE-M3 的 SPLADE 风格稀疏向量
替代传统 BM25 检索
"""
from typing import List, Dict, Any, Optional
import logging
import torch
import os
import json
import hashlib
from pathlib import Path

from FlagEmbedding import BGEM3FlagModel

from src.config import config

logger = logging.getLogger(__name__)

# 全局缓存
_sparse_cache: Dict[str, "SparseSearch"] = {}


class SparseSearch:
    """
    基于 BGE-M3 稀疏向量的关键词检索。
    BGE-M3 的 sparse向量 类似 BM25，但由模型学习得到，效果更好。
    """

    def __init__(self, documents: List[Dict[str, Any]], cache_dir: Optional[str] = None):
        """
        初始化 BGE-M3 稀疏检索索引。

        Args:
            documents: 文档列表，每项包含 text 和 metadata
            cache_dir: 缓存目录，用于存储编码后的索引
        """
        self.documents = documents
        self.doc_ids = [
            doc.get("metadata", {}).get("chunk_id") or doc.get("id", str(i))
            for i, doc in enumerate(documents)
        ]
        self.model = None
        self.doc_sparse_vectors = []
        
        # 计算缓存键
        self.cache_dir = Path(cache_dir or config.VECTORSTORE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        doc_hash = hashlib.md5(
            json.dumps([d.get("text", "")[:200] for d in documents], sort_keys=True).encode()
        ).hexdigest()[:12]
        self.cache_file = self.cache_dir / f"sparse_index_{doc_hash}.json"

        if not documents:
            logger.warning("No documents provided for SparseSearch")
            return

        # 尝试从缓存加载
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                
                # 加载 vectors - 保持 dict 格式 {token: weight}
                vectors_raw = cache_data.get("vectors", [])
                self.doc_sparse_vectors = [
                    {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in vec.items()}
                    for vec in vectors_raw
                ]
                logger.info(f"Loaded sparse index from cache: {self.cache_file} ({len(self.doc_sparse_vectors)} docs)")
                return  # 直接返回，不加载模型，不重新编码
            except Exception as e:
                logger.warning(f"Failed to load sparse cache: {e}")

        # 缓存不存在或不匹配，需要重新编码
        logger.info(f"Initializing BGE-M3 sparse retriever with {len(documents)} documents...")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        model_path = config.EMBEDDING_MODEL_PATH
        logger.info("Loading local BGE-M3 sparse model from %s", model_path)
        self.model = BGEM3FlagModel(
            model_path,
            use_fp16=torch.cuda.is_available(),
            devices="cuda" if torch.cuda.is_available() else "cpu",
        )

        texts = [doc.get("text", "") for doc in documents]
        logger.info("Encoding documents for sparse retrieval (first time or cache miss)...")
        results = self.model.encode(
            texts,
            return_sparse=True,
            return_dense=False,
            return_colbert_vecs=False,
            batch_size=32,
        )
        self.doc_sparse_vectors = results["lexical_weights"]
        
        # 保存到缓存
        try:
            cache_data = {
                "vectors": [
                    {str(k): float(v) for k, v in vec.items()}
                    for vec in self.doc_sparse_vectors
                ],
                "doc_ids": self.doc_ids,
            }
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f)
            logger.info(f"Saved sparse index to cache: {self.cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save sparse cache: {e}")
        
        logger.info(f"BGE-M3 sparse index built with {len(documents)} documents")

    def _load_model(self):
        """延迟加载模型（仅在需要时）"""
        if self.model is not None:
            return
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        model_path = config.EMBEDDING_MODEL_PATH
        self.model = BGEM3FlagModel(
            model_path,
            use_fp16=torch.cuda.is_available(),
            devices="cuda" if torch.cuda.is_available() else "cpu",
        )

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        使用 BGE-M3 稀疏向量搜索。

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            按得分排序的文档列表
        """
        if not self.documents:
            return []
        
        # 如果没有 sparse 向量（从缓存加载但有问题），回退到 BM25
        if not self.doc_sparse_vectors:
            logger.warning("No sparse vectors available, using fallback search")
            return self._fallback_search(query, top_k)
        
        # 如果模型未加载且没有预计算的查询向量，使用简单的文本匹配
        if not self.model:
            logger.debug("Model not loaded, using simple token matching")
            return self._simple_search(query, top_k)

        # 编码查询
        query_result = self.model.encode(
            [query],
            return_sparse=True,
            return_dense=False,
            return_colbert_vecs=False,
            batch_size=1,
        )
        query_sparse = query_result["lexical_weights"][0]

        # 计算每个文档与查询的稀疏得分（SPLADE 方式：取交集的 max）
        scores: Dict[int, float] = {}
        import math
        for token, q_weight in query_sparse.items():
            for doc_idx, doc_weights in enumerate(self.doc_sparse_vectors):
                if token in doc_weights:
                    # 累加 log(1 + doc_weight) * query_weight
                    score = math.log(1 + doc_weights[token]) * q_weight
                    scores[doc_idx] = scores.get(doc_idx, 0.0) + score

        # 排序
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_idx, score in ranked[:top_k]:
            doc = self.documents[doc_idx].copy()
            doc["sparse_score"] = float(score)
            doc["sparse_rank"] = doc_idx + 1
            doc["id"] = self.doc_ids[doc_idx]
            results.append(doc)

        return results

    def _simple_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """简单的 token 匹配搜索（当模型不可用时）"""
        import math
        query_tokens = set(query.lower().split())
        scores: Dict[int, float] = {}
        
        for doc_idx, doc in enumerate(self.documents):
            doc_tokens = set(doc.get("text", "").lower().split())
            common = query_tokens & doc_tokens
            if common:
                # 简单的 Jaccard 相似度
                scores[doc_idx] = len(common) / max(len(query_tokens | doc_tokens), 1)
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_idx, score in ranked[:top_k]:
            doc = self.documents[doc_idx].copy()
            doc["sparse_score"] = float(score)
            doc["sparse_rank"] = doc_idx + 1
            doc["id"] = self.doc_ids[doc_idx]
            results.append(doc)
        return results

    def _fallback_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """回退搜索：当没有任何索引时"""
        return self._simple_search(query, top_k)


def get_sparse_search(documents: List[Dict[str, Any]], cache_dir: Optional[str] = None) -> SparseSearch:
    """
    获取或创建 SparseSearch 实例（带缓存）。
    
    Args:
        documents: 文档列表
        cache_dir: 缓存目录
        
    Returns:
        SparseSearch 实例
    """
    cache_key = hashlib.md5(
        json.dumps([d.get("text", "")[:100] for d in documents], sort_keys=True).encode()
    ).hexdigest()[:16]
    
    if cache_key not in _sparse_cache:
        logger.info(f"Creating new SparseSearch (cache_key={cache_key})")
        _sparse_cache[cache_key] = SparseSearch(documents, cache_dir)
    
    return _sparse_cache[cache_key]
