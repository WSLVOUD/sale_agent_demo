"""General search tool for agents."""
import logging
from typing import List, Dict, Any

from ..rag.retriever import HybridSearch

logger = logging.getLogger(__name__)


class SearchTool:
    """General-purpose search tool wrapping hybrid search.
    
    This tool provides a simpler interface for agents that need
    to search the product knowledge base without parameter filtering.
    """
    
    def __init__(self, hybrid_search: HybridSearch):
        """Initialize with a hybrid search instance.
        
        Args:
            hybrid_search: Initialized HybridSearch instance
        """
        self.hybrid_search = hybrid_search
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform a basic search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries
        """
        try:
            results = self.hybrid_search.search(query=query, top_k=top_k)
            logger.info(f"SearchTool: found {len(results)} results for: {query[:50]}")
            return results
        except Exception as e:
            logger.error(f"SearchTool error: {e}")
            return []
    
    def search_with_context(
        self,
        query: str,
        context: str = "",
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search with additional context for better results.
        
        Args:
            query: Primary search query
            context: Additional context to include
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries
        """
        combined_query = f"{context} {query}".strip() if context else query
        return self.search(combined_query, top_k)
    
    def extract_models(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract model identifiers from search results.
        
        Args:
            results: Search results
            
        Returns:
            List of unique model names
        """
        import re
        models = set()
        model_pattern = re.compile(
            r"\b(TW\d+[\w.-]*|T\d{2}Omni[\w.-]+)\b",
            re.IGNORECASE
        )
        
        for result in results:
            text = result.get("text", "")
            for match in model_pattern.finditer(text):
                models.add(match.group(0))
        
        return list(models)
