"""Message normalization utilities."""
from typing import Dict, List, Any


def _normalize_role(role: str) -> str:
    """Map message types to canonical role names."""
    mapping = {"human": "user", "ai": "assistant", "system": "system"}
    return mapping.get(role, role)


def normalize_message(msg: Any) -> Dict[str, str]:
    """Normalize any message format to a standard {role, content} dict."""
    if isinstance(msg, dict):
        return {
            "role": _normalize_role(msg.get("role", "")),
            "content": msg.get("content", ""),
        }
    # LangChain message objects: HumanMessage / AIMessage / SystemMessage
    return {
        "role": _normalize_role(getattr(msg, "type", "unknown")),
        "content": getattr(msg, "content", str(msg)),
    }


def normalize_history(messages: List[Any]) -> List[Dict[str, str]]:
    """Normalize a list of messages to role/content dicts."""
    result = []
    for entry in messages or []:
        normalized = normalize_message(entry)
        if normalized.get("content"):  # Skip empty messages
            result.append(normalized)
    return result


def dedupe_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove consecutive duplicate (role, content) entries."""
    deduped = []
    for msg in messages or []:
        if deduped and deduped[-1].get("role") == msg.get("role") and deduped[-1].get("content") == msg.get("content"):
            continue
        deduped.append(msg)
    return deduped
