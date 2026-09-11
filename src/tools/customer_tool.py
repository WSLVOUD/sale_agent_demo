"""Customer information tool for agents."""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CustomerInfoTool:
    """Tool for managing customer session information.
    
    This tool provides access to customer context across agent sessions,
    including conversation history and accumulated requirements.
    """
    
    def __init__(self, memory_store=None):
        """Initialize with an optional memory store.
        
        Args:
            memory_store: Memory store instance for persistence
        """
        self.memory_store = memory_store
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get all context for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict with history, requirements, and metadata
        """
        if not self.memory_store:
            return {"history": [], "requirements": {}}
        
        if hasattr(self.memory_store, "get_history"):
            history = self.memory_store.get_history(session_id)
            requirements = self.memory_store.get_requirements(session_id)
        else:
            session_data = self.memory_store.get(session_id, {})
            history = session_data.get("messages", [])
            requirements = session_data.get("requirements", {})
        
        return {
            "history": history,
            "requirements": requirements,
        }
    
    def get_requirement(self, session_id: str, key: str) -> Optional[Any]:
        """Get a specific requirement value.
        
        Args:
            session_id: Session identifier
            key: Requirement key (e.g., "usage", "indoor")
            
        Returns:
            Requirement value or None
        """
        context = self.get_session_context(session_id)
        return context.get("requirements", {}).get(key)
    
    def update_requirement(self, session_id: str, key: str, value: Any) -> None:
        """Update a specific requirement value.
        
        Args:
            session_id: Session identifier
            key: Requirement key
            value: New value
        """
        if not self.memory_store:
            return
        
        context = self.get_session_context(session_id)
        requirements = context.get("requirements", {}).copy()
        requirements[key] = value
        
        if hasattr(self.memory_store, "set_requirements"):
            self.memory_store.set_requirements(session_id, requirements)
    
    def clear_requirements(self, session_id: str) -> None:
        """Clear all requirements for a session.
        
        Args:
            session_id: Session identifier
        """
        if not self.memory_store:
            return
        
        if hasattr(self.memory_store, "set_requirements"):
            self.memory_store.set_requirements(session_id, {})
    
    def get_conversation_turns(self, session_id: str, limit: int = 5) -> list:
        """Get recent conversation turns.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of turns to return
            
        Returns:
            List of recent messages
        """
        context = self.get_session_context(session_id)
        history = context.get("history", [])
        return history[-limit:] if history else []
