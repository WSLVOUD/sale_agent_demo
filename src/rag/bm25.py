"""
BM25检索模块
基于关键词的BM25算法检索
"""
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import re
import logging

logger = logging.getLogger(__name__)


class BM25Search:
    """BM25 keyword search for LED products."""
    
    def __init__(self, documents: List[Dict[str, Any]]):
        """Initialize BM25 index."""
        self.documents = documents
        self.doc_ids = [
            doc.get("metadata", {}).get("chunk_id") or doc.get("id", str(i))
            for i, doc in enumerate(documents)
        ]
        self.tokenized_corpus = [self._tokenize(doc.get("text", "")) for doc in documents]
        
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            logger.info(f"BM25 index built with {len(documents)} documents")
        else:
            self.bm25 = None
            logger.warning("No documents to index for BM25")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer - lowercase and split."""
        if not text:
            return []
        return re.findall(r'\w+', text.lower())
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using BM25."""
        if not self.bm25 or not self.documents:
            return []
        
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []
        
        scores = self.bm25.get_scores(tokenized_query)
        
        results = []
        for i, score in enumerate(scores):
            if score > 0:
                doc = self.documents[i].copy()
                doc["bm25_score"] = float(score)
                doc["bm25_rank"] = i + 1
                doc["id"] = self.doc_ids[i]
                results.append(doc)
        
        results.sort(key=lambda x: x["bm25_score"], reverse=True)
        return results[:top_k]
