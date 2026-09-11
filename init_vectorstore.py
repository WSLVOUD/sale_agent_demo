"""Initialize vector database from product data."""
import logging
import sys

from src.config import config
from src.rag.loader import load_all_product_files
from src.core.embeddings import recreate_vectorstore

logger = logging.getLogger(__name__)


def init_product_vectorstore():
    """Build the product vector store from all product files in data directory."""
    print(f"Loading products from: {config.DATA_DIR}")

    docs = load_all_product_files(config.DATA_DIR)
    print(f"Loaded {len(docs)} product documents")

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
