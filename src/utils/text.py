"""Text processing utilities."""
import re


def strip_markdown(text: str) -> str:
    """Remove common markdown artifacts so the chat reply is plain text."""
    if not text:
        return text
    cleaned = text.replace("**", "").replace("__", "")
    # Strip backtick code spans (single or triple)
    cleaned = re.sub(r"`+([^`]*?)`+", r"\1", cleaned)
    # Strip leading heading hashes
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)
    # Strip bullet markers at line start
    cleaned = re.sub(r"(?m)^\s*[-*+]\s+", "", cleaned)
    # Strip code block markers
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n?```$", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving line breaks."""
    if not text:
        return text
    # Replace multiple spaces with single space
    cleaned = re.sub(r"[ \t]+", " ", text)
    # Remove trailing whitespace from each line
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines())
    return cleaned.strip()


def truncate_text(text: str, max_length: int = 800, sentence_boundary: bool = True) -> str:
    """Truncate text to max_length, optionally at sentence boundary."""
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length]
    
    if sentence_boundary:
        # Find last sentence boundary
        last_period = max(
            truncated.rfind('。'),
            truncated.rfind('！'),
            truncated.rfind('？'),
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?'),
        )
        if last_period > max_length * 0.5:  # Only use if it's in the latter half
            return text[:last_period + 1]
    
    return truncated + "..."


def remove_internal_phrasing(text: str) -> str:
    """Remove internal implementation wording from customer-facing responses."""
    internal_phrases = [
        "根据资料",
        "查询结果",
        "检索到",
        "数据库中",
        "从知识库",
        "我手头的资料",
        "产品资料里",
    ]
    for phrase in internal_phrases:
        text = text.replace(phrase, "我了解")
    return text
