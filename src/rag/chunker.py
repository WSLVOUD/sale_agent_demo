"""
文档切片模块
将文档切分为小块以优化检索
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1500,
    chunk_overlap: int = 150,
    separators: Optional[List[str]] = None
) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval.
    
    Args:
        documents: List of documents to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        separators: Custom separators for splitting
        
    Returns:
        List of chunked documents
    """
    if separators is None:
        separators = ["\n==========", "\n|", "Model:", "====", "\n\n", "\n"]
    
    logger.info(f"Splitting {len(documents)} documents")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    
    return chunks
