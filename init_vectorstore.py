"""Initialize vector database from product data (Phase 2: Model 级语料)."""
import logging
import sys

from src.config import config
from src.rag.corpus import build_retrieval_corpus
from src.core.embeddings import recreate_vectorstore

logger = logging.getLogger(__name__)


def init_product_vectorstore():
    """Build the product vector store from the canonical Model-level corpus."""
    print(f"Loading Model-level products from: {config.DATA_DIR}")

    docs = build_retrieval_corpus(config.DATA_DIR)
    series = {doc.metadata.get("series_id") for doc in docs}
    print(f"Loaded {len(docs)} model documents ({len(series)} series)")

    print(f"Creating product vector store at: {config.VECTORSTORE_DIR}")
    vectorstore = recreate_vectorstore(docs, persist_dir=config.VECTORSTORE_DIR)
    print("Product vector store created successfully!")
    return vectorstore


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    use_sample = "--sample" in sys.argv

    if use_sample:
        print("Sample mode not available - use main data files")
    else:
        init_product_vectorstore()
