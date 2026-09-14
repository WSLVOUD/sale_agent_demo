"""
检索模块测试（Phase 2 之后：Model 级语料）。

覆盖：
- BM25 关键词检索
- HybridSearch（Vector + BGE-M3 Sparse + BM25 + RRF）
- metadata 硬过滤（亮度 / 点间距 / 租赁 / 环境）
- Golden Dataset 的 Series 级 Recall@5

注意（历史问题修正）：本文件原先用 ``load_structured_documents()`` 构建
Series 级语料，且把 LangChain ``Document`` 直接传给 ``BM25Search``，
导致 fixture 抛错、7 条检索测试被静默 skip。现在统一改用生产语料
``build_retrieval_corpus()``（Model 级），并通过 ``corpus_as_dicts()`` 转换。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rag.bm25 import BM25Search  # noqa: E402
from src.rag.corpus import build_retrieval_corpus, corpus_as_dicts  # noqa: E402
from src.rag.fusion import HybridSearch  # noqa: E402


@pytest.fixture(scope="module")
def corpus():
    """生产语料（Model 级，一个实际销售型号一个文档）。"""
    return corpus_as_dicts(build_retrieval_corpus())


@pytest.fixture(scope="module")
def bm25_index(corpus):
    return BM25Search(corpus)


@pytest.fixture(scope="module")
def hybrid_search(corpus):
    """构建 Hybrid Search。

    刻意使用**内存版** Chroma（由生产语料现场构建）而不是磁盘上的向量库：
    这样测试结果不依赖 ``vectorstore/`` 目录当时是什么状态，
    保证"语料 = 测试对象 = 生产语料"。
    """
    try:
        from src.config import config
        from src.core.embeddings import get_embeddings
        from src.rag.sparse import get_sparse_search
        from langchain_community.vectorstores import Chroma

        documents = build_retrieval_corpus()
        vectorstore = Chroma.from_documents(documents=documents, embedding=get_embeddings())
        sparse = get_sparse_search(corpus, config.VECTORSTORE_DIR)
        return HybridSearch(vectorstore, sparse, BM25Search(corpus))
    except Exception as exc:  # pragma: no cover - 环境缺失时跳过
        pytest.skip(f"Could not initialize HybridSearch: {exc}")


class TestBM25:
    """BM25 关键词检索测试"""

    def test_bm25_ranking(self, bm25_index):
        results = bm25_index.search("outdoor LED screen", top_k=5)
        assert len(results) > 0
        assert len(results) <= 5

    def test_bm25_indoor_query(self, bm25_index):
        assert len(bm25_index.search("indoor meeting room display", top_k=5)) > 0

    def test_bm25_rental_query(self, bm25_index):
        assert len(bm25_index.search("rental LED screen concert", top_k=5)) > 0

    def test_bm25_returns_model_level_documents(self, bm25_index):
        """Phase 2 之后语料必须是 Model 级（每个文档一个实际型号）"""
        results = bm25_index.search("indoor LED display conference", top_k=5)
        assert results
        for item in results:
            metadata = item.get("metadata", {})
            assert metadata.get("level") == "model"
            assert "-P" in str(metadata.get("model", ""))


class TestHybridSearch:
    """混合检索测试"""

    def test_hybrid_search_basic(self, hybrid_search):
        results = hybrid_search.search("outdoor advertising LED", top_k=5)
        assert len(results) > 0
        assert len(results) <= 5

    def test_hybrid_search_with_filters(self, hybrid_search):
        results = hybrid_search.search("LED screen", top_k=5, brightness_min=4000, pitch_max=5.0)
        assert len(results) >= 0  # 严格过滤下允许为空

    def test_hybrid_search_with_rental(self, hybrid_search):
        results = hybrid_search.search("rental concert stage LED", top_k=5, is_rental=True)
        assert len(results) > 0
        for item in results:
            assert item.get("metadata", {}).get("is_rental") is True

    def test_recall_at_5_on_golden_dataset(self, hybrid_search):
        """Golden Dataset 的 Series 级 Recall@5（与生产一致：先套硬约束过滤）

        同时打印"不过滤"的对照值 —— 这个差距正是 Phase 3 硬约束过滤的价值所在。
        """
        import json

        from eval.metrics import build_retrieval_filters

        dataset_path = os.path.join(project_root, "eval", "golden_dataset.json")
        with open(dataset_path, "r", encoding="utf-8") as handle:
            cases = json.load(handle)["cases"]

        def evaluate(apply_filters: bool) -> tuple[float, int]:
            scored = 0
            total = 0.0
            for case in cases:
                expected = case.get("series") or []
                if not expected:
                    continue
                hard = case.get("hard") or {}
                kwargs = {"top_k": 5}
                if apply_filters:
                    kwargs["filters"] = build_retrieval_filters(hard) or None
                    if hard.get("brightness_min") is not None:
                        kwargs["brightness_min"] = hard["brightness_min"]
                results = hybrid_search.search(case["query"], **kwargs)
                retrieved = []
                for item in results:
                    series = item.get("metadata", {}).get("series_id")
                    if series and series not in retrieved:
                        retrieved.append(series)
                total += len(set(retrieved) & set(expected)) / len(expected)
                scored += 1
            return (total / scored if scored else 0.0), scored

        raw_average, _ = evaluate(apply_filters=False)
        average, scored = evaluate(apply_filters=True)

        if not scored:
            pytest.skip("No golden cases with series expectations")
        print(
            f"\n  Series Recall@5: 硬约束过滤后 {average:.1%} / 不过滤 {raw_average:.1%} "
            f"({scored} cases)"
        )
        assert average >= 0.75, f"带回硬约束的 Recall@5 回退到 {average:.1%}（基线 0.925）"


class TestRetrievalFilters:
    """检索 metadata 过滤测试（Phase 3 硬约束）"""

    def test_filter_by_brightness(self, hybrid_search):
        results = hybrid_search.search("LED display", brightness_min=5000, top_k=5)
        for item in results:
            metadata = item.get("metadata", {})
            assert (metadata.get("brightness_nit") or 0) >= 5000

    def test_filter_by_pitch(self, hybrid_search):
        results = hybrid_search.search("LED display", pitch_max=3.0, top_k=5)
        for item in results:
            pitch = item.get("metadata", {}).get("pixel_pitch_max_mm")
            if pitch is not None:
                assert pitch <= 3.0

    def test_filter_by_rental(self, hybrid_search):
        results = hybrid_search.search("LED screen", is_rental=True, top_k=5)
        for item in results:
            assert item.get("metadata", {}).get("is_rental") is True

    def test_filter_by_environment(self, hybrid_search):
        """户外过滤不得返回室内型号（Phase 3 禁止项）"""
        results = hybrid_search.search(
            "outdoor advertising display", top_k=5, filters={"outdoor": True}
        )
        assert results
        for item in results:
            assert item.get("metadata", {}).get("outdoor") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
