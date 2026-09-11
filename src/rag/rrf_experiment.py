"""
RRF 参数网格搜索实验
Phase 2: RRF 参数实验

对 RRF 权重进行小规模网格搜索，对比不同权重组合：
- Recall
- MRR
- Latency

使用 Golden Dataset 验证最优参数。
"""
from __future__ import annotations

import itertools
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# RRF k 参数的候选值
RRF_K_VALUES = [30, 60, 100]

# 权重搜索空间
# 格式: (vector_weight, sparse_weight, bm25_weight)
WEIGHT_GRID = [
    # 基础权重
    (1.0, 0.0, 0.0),      # 仅向量
    (0.0, 1.0, 0.0),      # 仅 Sparse
    (0.0, 0.0, 1.0),      # 仅 BM25
    
    # 双路组合
    (1.0, 1.0, 0.0),      # 向量 + Sparse
    (1.0, 0.0, 1.0),      # 向量 + BM25
    (0.0, 1.0, 1.0),      # Sparse + BM25
    
    # 三路组合
    (1.0, 1.0, 1.0),      # 全部等权重
    (2.0, 1.0, 1.0),      # 向量优先
    (1.0, 2.0, 1.0),      # Sparse 优先
    (1.0, 1.0, 2.0),      # BM25 优先
    
    # 细粒度权重
    (1.5, 1.0, 0.5),      # 向量主导
    (1.0, 1.5, 0.5),      # Sparse 主导
    (1.0, 0.5, 1.5),      # BM25 主导
]


@dataclass
class RetrievalResult:
    """单次检索结果"""
    query_id: str
    query: str
    expected_products: List[str]
    retrieved_products: List[str]
    latency_ms: float
    weights: Tuple[float, float, float]
    rrf_k: int


@dataclass
class RRFExperimentResult:
    """实验结果"""
    weights: Tuple[float, float, float]
    rrf_k: int
    
    # 检索指标
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    
    # 性能指标
    avg_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    
    # 详细结果
    results: List[RetrievalResult] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "weights": self.weights,
            "rrf_k": self.rrf_k,
            "recall@5": round(self.recall_at_5, 4),
            "recall@10": round(self.recall_at_10, 4),
            "mrr": round(self.mrr, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
        }


class RRFGridSearcher:
    """
    RRF 参数网格搜索器
    
    使用 Golden Dataset 验证不同 RRF 权重组合的效果。
    """
    
    def __init__(self, hybrid_search, eval_dataset_path: str = None):
        """
        初始化搜索器
        
        Args:
            hybrid_search: HybridSearch 实例
            eval_dataset_path: 评估数据集路径
        """
        self.hybrid_search = hybrid_search
        self.eval_dataset_path = eval_dataset_path
        self._dataset = None
    
    def load_dataset(self) -> List[dict]:
        """加载评估数据集"""
        if self._dataset is not None:
            return self._dataset
        
        if not self.eval_dataset_path or not Path(self.eval_dataset_path).exists():
            logger.warning(f"Eval dataset not found: {self.eval_dataset_path}")
            return []
        
        import json
        with open(self.eval_dataset_path, "r", encoding="utf-8") as f:
            self._dataset = json.load(f)
        
        logger.info(f"Loaded {len(self._dataset)} eval queries")
        return self._dataset
    
    def _calculate_recall(self, expected: List[str], retrieved: List[str], k: int) -> float:
        """计算 Recall@k"""
        if not expected:
            return 1.0  # 无期望时默认为 1
        
        retrieved_k = retrieved[:k]
        hits = sum(1 for p in expected if p in retrieved_k)
        return hits / len(expected)
    
    def _calculate_mrr(self, expected: List[str], retrieved: List[str]) -> float:
        """计算 MRR"""
        if not expected:
            return 1.0
        
        for i, product in enumerate(retrieved, 1):
            if product in expected:
                return 1.0 / i
        return 0.0
    
    def run_single_experiment(
        self,
        weights: Tuple[float, float, float],
        rrf_k: int,
        queries: List[dict],
    ) -> RRFExperimentResult:
        """
        运行单组参数实验
        
        Args:
            weights: (vector, sparse, bm25) 权重
            rrf_k: RRF k 参数
            queries: 查询列表
            
        Returns:
            实验结果
        """
        result = RRFExperimentResult(
            weights=weights,
            rrf_k=rrf_k,
        )
        
        total_latency = 0.0
        
        for query_item in queries:
            query_id = query_item.get("id", "")
            query = query_item.get("query", "")
            expected = query_item.get("expected_products", [])
            
            start_time = time.time()
            
            # 执行检索
            try:
                # 临时修改权重
                old_vector_weight = self.hybrid_search.vector_weight if hasattr(self.hybrid_search, 'vector_weight') else 1.0
                old_sparse_weight = self.hybrid_search.sparse_weight
                old_bm25_weight = self.hybrid_search.bm25_weight
                old_rrf_k = self.hybrid_search.fusion.k
                
                # 设置新权重
                if hasattr(self.hybrid_search, 'vector_weight'):
                    self.hybrid_search.vector_weight = weights[0]
                self.hybrid_search.sparse_weight = weights[1]
                self.hybrid_search.bm25_weight = weights[2]
                self.hybrid_search.fusion.k = rrf_k
                
                # 检索
                search_results = self.hybrid_search.search(query, top_k=10)
                
                # 恢复权重
                if hasattr(self.hybrid_search, 'vector_weight'):
                    self.hybrid_search.vector_weight = old_vector_weight
                self.hybrid_search.sparse_weight = old_sparse_weight
                self.hybrid_search.bm25_weight = old_bm25_weight
                self.hybrid_search.fusion.k = old_rrf_k
                
            except Exception as e:
                logger.warning(f"Search failed for query {query_id}: {e}")
                search_results = []
            
            latency_ms = (time.time() - start_time) * 1000
            total_latency += latency_ms
            
            # 提取检索到的产品
            retrieved = [
                r.get("metadata", {}).get("model", "")
                or r.get("metadata", {}).get("product_id", "")
                or str(r.get("id", ""))
                for r in search_results
            ]
            retrieved = [p for p in retrieved if p]
            
            # 计算指标
            recall_5 = self._calculate_recall(expected, retrieved, 5)
            recall_10 = self._calculate_recall(expected, retrieved, 10)
            mrr = self._calculate_mrr(expected, retrieved)
            
            # 记录结果
            result.results.append(RetrievalResult(
                query_id=query_id,
                query=query,
                expected_products=expected,
                retrieved_products=retrieved,
                latency_ms=latency_ms,
                weights=weights,
                rrf_k=rrf_k,
            ))
            
            result.recall_at_5 += recall_5
            result.recall_at_10 += recall_10
            result.mrr += mrr
        
        # 计算平均值
        n = len(queries) if queries else 1
        result.recall_at_5 /= n
        result.recall_at_10 /= n
        result.mrr /= n
        result.avg_latency_ms = total_latency / n
        result.total_latency_ms = total_latency
        
        return result
    
    def run_grid_search(
        self,
        weight_grid: List[Tuple[float, float, float]] = None,
        k_values: List[int] = None,
    ) -> List[RRFExperimentResult]:
        """
        运行网格搜索
        
        Args:
            weight_grid: 权重组合列表
            k_values: RRF k 参数列表
            
        Returns:
            所有实验结果
        """
        if weight_grid is None:
            weight_grid = WEIGHT_GRID
        if k_values is None:
            k_values = RRF_K_VALUES
        
        queries = self.load_dataset()
        if not queries:
            logger.error("No eval queries loaded, aborting grid search")
            return []
        
        all_results = []
        total_experiments = len(weight_grid) * len(k_values)
        current = 0
        
        logger.info(f"Starting grid search: {len(weight_grid)} weight combos × {len(k_values)} k values = {total_experiments} experiments")
        
        for weights in weight_grid:
            for k in k_values:
                current += 1
                logger.info(f"[{current}/{total_experiments}] Testing weights={weights}, k={k}")
                
                result = self.run_single_experiment(weights, k, queries)
                all_results.append(result)
                
                logger.info(
                    f"  Results: Recall@5={result.recall_at_5:.4f}, "
                    f"Recall@10={result.recall_at_10:.4f}, "
                    f"MRR={result.mrr:.4f}, "
                    f"Latency={result.avg_latency_ms:.2f}ms"
                )
        
        return all_results
    
    def get_best_config(
        self,
        results: List[RRFExperimentResult],
        metric: str = "recall@10",
    ) -> RRFExperimentResult:
        """
        获取最佳配置
        
        Args:
            results: 实验结果列表
            metric: 优化指标 (recall@5, recall@10, mrr, latency)
            
        Returns:
            最佳配置
        """
        if not results:
            raise ValueError("No results to compare")
        
        metric_map = {
            "recall@5": lambda r: r.recall_at_5,
            "recall@10": lambda r: r.recall_at_10,
            "mrr": lambda r: r.mrr,
            "latency": lambda r: -r.avg_latency_ms,  # 负数表示越小越好
        }
        
        if metric not in metric_map:
            raise ValueError(f"Unknown metric: {metric}")
        
        scorer = metric_map[metric]
        return max(results, key=scorer)
    
    def generate_report(
        self,
        results: List[RRFExperimentResult],
        output_path: str = None,
    ) -> str:
        """
        生成实验报告
        
        Args:
            results: 实验结果列表
            output_path: 输出路径
            
        Returns:
            报告文本
        """
        lines = [
            "# RRF 参数网格搜索实验报告",
            "",
            f"实验数量: {len(results)}",
            "",
            "## 完整结果",
            "",
            "| Weights (V,S,B) | k | Recall@5 | Recall@10 | MRR | Latency(ms) |",
            "|---|---|---|---|---|---|",
        ]
        
        for r in sorted(results, key=lambda x: x.recall_at_10, reverse=True):
            lines.append(
                f"| {r.weights} | {r.rrf_k} | "
                f"{r.recall_at_5:.4f} | {r.recall_at_10:.4f} | "
                f"{r.mrr:.4f} | {r.avg_latency_ms:.2f} |"
            )
        
        # 最佳配置
        best_recall = self.get_best_config(results, "recall@10")
        best_mrr = self.get_best_config(results, "mrr")
        best_latency = self.get_best_config(results, "latency")
        
        lines.extend([
            "",
            "## 最佳配置",
            "",
            f"**最佳 Recall@10**: {best_recall.weights}, k={best_recall.rrf_k} "
            f"(Recall@10={best_recall.recall_at_10:.4f})",
            "",
            f"**最佳 MRR**: {best_mrr.weights}, k={best_mrr.rrf_k} "
            f"(MRR={best_mrr.mrr:.4f})",
            "",
            f"**最低延迟**: {best_latency.weights}, k={best_latency.rrf_k} "
            f"(Latency={best_latency.avg_latency_ms:.2f}ms)",
            "",
            "## 建议",
            "",
            "根据业务需求选择：",
            "- 高召回优先：使用 Recall@10 最佳配置",
            "- 排序质量优先：使用 MRR 最佳配置",
            "- 低延迟优先：使用延迟最低配置",
            "- 综合最优：考虑权重 (1.0, 1.0, 1.0), k=60 作为默认配置",
        ])
        
        report = "\n".join(lines)
        
        if output_path:
            Path(output_path).write_text(report, encoding="utf-8")
            logger.info(f"Report saved to {output_path}")
        
        return report


def run_rrf_experiment(
    hybrid_search,
    eval_dataset_path: str = "eval/dataset.json",
    output_path: str = "eval/rrf_experiment_report.md",
) -> RRFExperimentResult:
    """
    运行 RRF 参数实验的便捷函数
    
    Returns:
        最佳配置
    """
    searcher = RRFGridSearcher(hybrid_search, eval_dataset_path)
    results = searcher.run_grid_search()
    
    if results:
        report = searcher.generate_report(results, output_path)
        best = searcher.get_best_config(results, "recall@10")
        print(report)
        return best
    
    return None
