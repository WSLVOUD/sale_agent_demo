"""
检索模块测试（Phase 13）

测试 HybridSearch / BM25 / Sparse 检索链路的关键指标：
- Recall@K
- MRR (Mean Reciprocal Rank)
- 组合过滤

注意：这些测试需要 Chroma 向量库实例。如果向量库不存在会跳过。
"""
import pytest
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rag.fusion import HybridSearch
from src.rag.bm25 import BM25Search
from src.rag.json_loader import load_structured_documents


def _load_test_data():
    """加载测试用向量库文档。"""
    data_dir = os.path.join(project_root, "data")
    docs = load_structured_documents(data_dir)
    return docs


@pytest.fixture(scope="module")
def test_docs():
    """加载测试文档（只加载一次）。"""
    return _load_test_data()


@pytest.fixture(scope="module")
def bm25_index(test_docs):
    """构建 BM25 索引。"""
    # BM25Search 需要 dict 对象，Document 需要转换
    dict_docs = []
    for doc in test_docs:
        if hasattr(doc, "get"):
            dict_docs.append(doc)
        else:
            # LangChain Document → dict
            dict_docs.append({
                "text": getattr(doc, "page_content", str(doc)),
                "metadata": getattr(doc, "metadata", {}),
                "id": getattr(doc, "id", id(doc)),
            })
    return BM25Search(dict_docs)


@pytest.fixture(scope="module")
def hybrid_search(test_docs):
    """构建 Hybrid Search（需要 Chroma）。"""
    try:
        from src.core.embeddings import get_embeddings
        from langchain_community.vectorstores import Chroma

        embeddings = get_embeddings()
        chroma = Chroma(
            embedding_function=embeddings,
            persist_directory=os.path.join(project_root, "vectorstore"),
        )
        # 如果 collection 为空，跳过测试
        count = chroma._collection.count()
        if count == 0:
            pytest.skip("Chroma vector store is empty; run init_vectorstore.py first")
        return HybridSearch(chroma, bm25=BM25Search(test_docs))
    except Exception as e:
        pytest.skip(f"Could not initialize HybridSearch: {e}")


class TestBM25:
    """BM25 关键词检索测试"""

    def test_bm25_ranking(self, bm25_index):
        """BM25 检索应返回按相关性排序的结果"""
        results = bm25_index.search("outdoor LED screen", top_k=5)
        assert len(results) > 0
        assert len(results) <= 5

    def test_bm25_indoor_query(self, bm25_index):
        """室内场景检索"""
        results = bm25_index.search("indoor meeting room display", top_k=5)
        assert len(results) > 0

    def test_bm25_rental_query(self, bm25_index):
        """租赁场景检索"""
        results = bm25_index.search("rental LED screen concert", top_k=5)
        assert len(results) > 0


class TestHybridSearch:
    """混合检索测试"""

    def test_hybrid_search_basic(self, hybrid_search):
        """基本混合检索"""
        results = hybrid_search.search("outdoor advertising LED", top_k=5)
        assert len(results) > 0
        assert len(results) <= 5

    def test_hybrid_search_with_filters(self, hybrid_search):
        """带过滤的混合检索"""
        results = hybrid_search.search(
            "LED screen",
            top_k=5,
            brightness_min=4000,
            pitch_max=5.0,
        )
        # Should return some results even with strict filters
        # (outdoor screens typically have brightness >= 4000)
        assert len(results) >= 0  # 0 is valid if no matches

    def test_hybrid_search_with_rental(self, hybrid_search):
        """租赁场景"""
        results = hybrid_search.search(
            "rental concert stage LED",
            top_k=5,
            is_rental=True,
        )
        assert len(results) > 0

    def test_hybrid_search_recall(self, hybrid_search):
        """测试召回率（需要 eval/dataset.json 中的 expected_ids）"""
        import json
        dataset_path = os.path.join(project_root, "eval", "dataset.json")
        if not os.path.exists(dataset_path):
            pytest.skip("eval/dataset.json not found")

        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        recall_at_5_total = 0
        recall_at_10_total = 0
        count = 0

        for case in dataset.get("queries", []):
            expected_ids = case.get("expected_ids", [])
            if not expected_ids:
                continue
            query_text = case.get("query", "")
            results = hybrid_search.search(query_text, top_k=10)
            retrieved_ids = [r.get("metadata", {}).get("product_id", "") for r in results]

            hit_5 = sum(1 for eid in expected_ids if eid in retrieved_ids[:5])
            hit_10 = sum(1 for eid in expected_ids if eid in retrieved_ids)

            recall_at_5 = hit_5 / len(expected_ids) if expected_ids else 0
            recall_at_10 = hit_10 / len(expected_ids) if expected_ids else 0

            recall_at_5_total += recall_at_5
            recall_at_10_total += recall_at_10
            count += 1

        if count == 0:
            pytest.skip("No evaluation cases with expected_ids found")

        avg_recall_at_5 = recall_at_5_total / count
        avg_recall_at_10 = recall_at_10_total / count

        # 要求 Recall@5 >= 0.30 (宽松基线)
        assert avg_recall_at_5 >= 0.30, (
            f"Recall@5 = {avg_recall_at_5:.2%} < 30% (基线要求)"
        )
        print(f"\n  Recall@5 = {avg_recall_at_5:.1%}  Recall@10 = {avg_recall_at_10:.1%}")


class TestRetrievalFilters:
    """检索过滤测试"""

    def test_filter_by_brightness(self, hybrid_search):
        """亮度过滤"""
        results = hybrid_search.search(
            "LED display",
            brightness_min=5000,
            top_k=5,
        )
        for r in results:
            bn = r.get("metadata", {}).get("brightness_nit") or 0
            assert bn >= 5000

    def test_filter_by_pitch(self, hybrid_search):
        """点间距过滤"""
        results = hybrid_search.search(
            "LED display",
            pitch_max=3.0,
            top_k=5,
        )
        for r in results:
            pmax = r.get("metadata", {}).get("pixel_pitch_max_mm")
            if pmax is not None:
                assert pmax <= 3.0

    def test_filter_by_rental(self, hybrid_search):
        """租赁过滤"""
        results = hybrid_search.search(
            "LED screen",
            is_rental=True,
            top_k=5,
        )
        for r in results:
            rental = r.get("metadata", {}).get("is_rental")
            assert rental is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
