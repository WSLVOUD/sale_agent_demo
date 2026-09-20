"""
混合检索融合模块
结合向量检索、BGE-M3 Sparse 检索、BM25 检索（可选同时启用）
"""
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ReciprocalRankFusion:
    """Reciprocal Rank Fusion for combining search results."""
    
    def __init__(self, k: int = 60):
        self.k = k
    
    def fuse(
        self,
        ranked_lists: List[List[Tuple[str, float]]],
        top_k: int = 5,
        weights: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Fuse multiple ranked lists using RRF."""
        if weights is None:
            weights = [1.0] * len(ranked_lists)

        rrf_scores: Dict[str, float] = {}
        doc_sources: Dict[str, List[str]] = {}

        for list_idx, ranked_list in enumerate(ranked_lists):
            w = weights[list_idx] if list_idx < len(weights) else 1.0
            for rank, (doc_id, original_score) in enumerate(ranked_list, 1):
                rrf_score = w * (1 / (self.k + rank))
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0
                    doc_sources[doc_id] = []
                rrf_scores[doc_id] += rrf_score
                doc_sources[doc_id].append(original_score)

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, rrf_score in sorted_docs[:top_k]:
            results.append({
                "id": doc_id,
                "rrf_score": rrf_score,
                "source_scores": doc_sources[doc_id]
            })

        return results


class HybridSearch:
    """
    混合检索：向量 + BGE-M3 Sparse + BM25（可同时启用，也可单独使用）。
    
    优先级：sparse > bm25，当两者都存在时，Sparse 权重更高。
    """
    
    def __init__(
        self,
        vectorstore,
        sparse=None,
        bm25=None,
        sparse_weight=1.5,
        bm25_weight=1.0,
        max_per_series: int = 2,
    ):
        """
        初始化混合检索器。

        Args:
            vectorstore: ChromaDB 向量库
            sparse: BGE-M3 SparseSearch 实例（可选）
            bm25: BM25Search 实例（可选）
            sparse_weight: Sparse 检索在 RRF 中的权重（默认 1.5）
            bm25_weight: BM25 检索在 RRF 中的权重（默认 1.0）
            max_per_series: 同一个系列在 Top-K 里最多出现几条（默认 2）
        """
        self.vectorstore = vectorstore
        self.sparse = sparse
        self.bm25 = bm25
        self.sparse_weight = sparse_weight
        self.bm25_weight = bm25_weight
        self.max_per_series = int(max_per_series or 0)
        self.fusion = ReciprocalRankFusion(k=60)
        from .retriever import retrieve as vector_retrieve
        self._vector_retrieve = vector_retrieve
        
        active = [n for n, v in [("sparse", sparse), ("bm25", bm25)] if v]
        logger.info(f"HybridSearch initialized with: vector + {', '.join(active) or 'none'}")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        brightness_min: Optional[int] = None,
        brightness_max: Optional[int] = None,
        pitch_max: Optional[float] = None,
        pitch_min: Optional[float] = None,
        is_rental: Optional[bool] = None,
        display_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining all available retrievers."""
        # Build metadata filter
        meta_filters = dict(filters) if filters else {}
        if brightness_min is not None:
            meta_filters["brightness_min_cd"] = {"$gte": brightness_min}
        if brightness_max is not None:
            pass
        if pitch_max is not None:
            meta_filters["pixel_pitch_max_mm"] = {"$lte": pitch_max}
        if pitch_min is not None:
            meta_filters["pixel_pitch_min_mm"] = {"$gte": pitch_min}
        if is_rental is not None:
            meta_filters["is_rental"] = is_rental

        final_filters = meta_filters if meta_filters else None

        ranked_lists = []
        list_weights = []

        # ── Vector search ──────────────────────────────────────
        if display_type:
            logger.info(f"Skipping vector search (display_type={display_type}), using keyword retrieval + filter")
            vector_results = []
        else:
            vector_results = self._vector_retrieve(self.vectorstore, query, top_k * 2, final_filters)
        vector_ranked = [(doc.get("id", str(i)), 0.0) for i, doc in enumerate(vector_results)]
        ranked_lists.append(vector_ranked)
        list_weights.append(1.0)

        # ── Helper: build ranked list from a search instance ───
        def _build_ranked(searcher, weight: float):
            results = searcher.search(query, top_k * 2)
            def _matches_filters(doc):
                meta = doc.get("metadata") or {}
                if final_filters:
                    for k, v in final_filters.items():
                        if isinstance(v, dict):
                            val = meta.get(k)
                            if v.get("$gte") is not None and (val is None or val < v["$gte"]):
                                return False
                            if v.get("$lte") is not None and (val is None or val > v["$lte"]):
                                return False
                        elif meta.get(k) != v:
                            return False
                if display_type and meta.get("display_type") != display_type:
                    return False
                return True
            results = [r for r in results if _matches_filters(r)]
            ranked = [(doc.get("id", str(i)), 0.0) for i, doc in enumerate(results)]
            if ranked:
                ranked_lists.append(ranked)
                list_weights.append(weight)

        # ── BGE-M3 Sparse search ────────────────────────────────
        if self.sparse:
            _build_ranked(self.sparse, self.sparse_weight)

        # ── BM25 search ────────────────────────────────────────
        if self.bm25:
            _build_ranked(self.bm25, self.bm25_weight)

        if not ranked_lists:
            return []

        # 多取一些候选：下面要做"同系列去重"（一个系列有很多型号时，
        # 5 个槽位会被同系列的相邻点间距占满，别的系列一个都进不来 ——
        # 实测检索召回下降、客户也会看到 5 个几乎一样的型号）。
        fused = self.fusion.fuse(ranked_lists, top_k=max(top_k * 3, top_k), weights=list_weights)

        # Rehydrate results
        doc_map = {doc.get("id"): doc for doc in vector_results}
        for searcher in [self.sparse, self.bm25]:
            if searcher:
                for doc in searcher.search(query, top_k * 2):
                    if doc.get("id"):
                        doc_map[doc["id"]] = doc

        max_per_series = int(getattr(self, "max_per_series", 2) or 0)
        results: List[Dict[str, Any]] = []
        overflow: List[Dict[str, Any]] = []
        per_series: Dict[str, int] = {}
        for item in fused:
            original = doc_map.get(item["id"])
            if not original:
                continue
            entry = original.copy()
            entry["rrf_score"] = item["rrf_score"]
            entry["source_scores"] = item["source_scores"]
            if brightness_max is not None:
                brightness = original.get("metadata", {}).get("brightness_max_cd")
                if brightness is not None and brightness > brightness_max:
                    continue
            series = str((original.get("metadata") or {}).get("series_id") or "")
            if max_per_series and series and per_series.get(series, 0) >= max_per_series:
                # 同一系列已经占了上限 → 先放一边，最后不够 top_k 时再补上
                overflow.append(entry)
                continue
            if series:
                per_series[series] = per_series.get(series, 0) + 1
            results.append(entry)
            if len(results) >= top_k:
                break

        if len(results) < top_k and overflow:
            results.extend(overflow[: top_k - len(results)])
        return results
