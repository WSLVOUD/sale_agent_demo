"""Product search tool for agents."""
import logging
from typing import Any, Dict, List, Optional

from ..rag.fusion import HybridSearch

logger = logging.getLogger(__name__)


class ProductSearchTool:
    """Tool for searching products in the vector store.
    
    This tool wraps the hybrid search functionality and provides
    a consistent interface for agents.
    """
    
    def __init__(self, hybrid_search: HybridSearch):
        """Initialize with a hybrid search instance.
        
        Args:
            hybrid_search: Initialized HybridSearch instance
        """
        self.hybrid_search = hybrid_search
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        brightness_min: Optional[int] = None,
        brightness_max: Optional[int] = None,
        pitch_max: Optional[float] = None,
        pitch_min: Optional[float] = None,
        is_rental: Optional[bool] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for products matching the query and filters.
        
        Args:
            query: Search query string
            top_k: Maximum number of results
            brightness_min: Minimum brightness in nits
            brightness_max: Maximum brightness in nits
            pitch_max: Maximum pixel pitch in mm
            pitch_min: Minimum pixel pitch in mm
            is_rental: Filter for rental-friendly products
            
        Returns:
            List of matching product dictionaries
        """
        try:
            results = self.hybrid_search.search(
                query=query,
                top_k=top_k,
                brightness_min=brightness_min,
                brightness_max=brightness_max,
                pitch_max=pitch_max,
                pitch_min=pitch_min,
                is_rental=is_rental,
                **kwargs
            )
            logger.info(f"ProductSearchTool: found {len(results)} products for query: {query[:50]}")
            return results
        except Exception as e:
            logger.error(f"ProductSearchTool search error: {e}")
            return []
    
    def get_product_by_model(self, model: str) -> Optional[Dict[str, Any]]:
        """Get a specific product by model name.
        
        Args:
            model: Model identifier (e.g., "TW21-COB-P0.9")
            
        Returns:
            Product dict or None if not found
        """
        results = self.search(f"model:{model}", top_k=5)
        for product in results:
            text = product.get("text", "") or ""
            if model.upper() in text.upper():
                return product
        return None
    
    def compare_products(self, models: List[str]) -> List[Dict[str, Any]]:
        """Get multiple products for comparison.
        
        Args:
            models: List of model identifiers
            
        Returns:
            List of product dictionaries
        """
        products = []
        for model in models:
            product = self.get_product_by_model(model)
            if product:
                products.append(product)
        return products
