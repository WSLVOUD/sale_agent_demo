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


# ── 内部标记块（绝不能发给客户）────────────────────────────────────────────
# 实测（2026-09-28）：自由问答的兜底把检索片段原样拼给了客户：
#     An LCD display, noted … [产品资料 1]
#     H4930LN-B | B Series (LCD Video Wall) | LCD | indoor | size=49" | …
# 这些是给 LLM 当"事实"用的内部块（含内部字段名），必须从客户可见文本里清掉。
_INTERNAL_BLOCK_RE = re.compile(
    r"\[\s*(?:产品资料|产品信息|Product\s*Info|Product\s*Data)\s*\d*\s*\]"
    r"[\s\S]*?(?=\[\s*(?:产品资料|产品信息|Product\s*Info|Product\s*Data)\s*\d*\s*\]|$)",
    re.IGNORECASE,
)


def strip_internal_blocks(text: str) -> str:
    """删掉内部标记块（[产品资料 N] …）与"未检索到"这类内部提示。"""
    source = str(text or "")
    if not source:
        return ""
    cleaned = _INTERNAL_BLOCK_RE.sub("", source)
    cleaned = cleaned.replace("（未检索到相关产品资料）", "")
    cleaned = cleaned.replace("(no matching product data)", "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


# ── "没有事实"的收尾评论句（客户口径 2026-09-28）────────────────────────────
# 实测客户可见文本：
#   "Both options use the same TW11-3216-P3.0 cabinets and suit a permanent church
#    install, so the choice mainly comes down to whether you prefer the slightly
#    taller horizontal layout or the slightly wider vertical one."
# 这种句子不含任何排布事实（箱体数 / 实际尺寸 / 模组数），只是"怎么选"的评论 ——
# 客户明确要求"话术少点，直接推荐就行"，所以一律删掉。
# 带排布数字的句子（"5 x 11 = 55 cabinets" / "330 modules"）一定保留。
_COMMENTARY_OPENER_RE = re.compile(
    r"^\s*(?:both|either)\s+(?:options?|layouts?|configurations?|ways?|setups?)\b",
    re.IGNORECASE,
)
# 注意：型号里也有数字（"TW11-3216-P3.0 cabinets"）—— 那种不算排布事实，
# 所以数字前面不允许紧邻字母 / 点 / 连字符。
_COMMENTARY_FACT_RE = re.compile(
    r"(?<![\w.\-])\d+\s*(?:[x×*]\s*\d+|(?:cabinets?|modules?)\b)", re.IGNORECASE
)
_COMMENTARY_ASK_RE = re.compile(
    r"[?？]|\bplease\b|\bshall i\b|\bcould you\b|\bwould you\b|\blet me know\b|\bkindly\b",
    re.IGNORECASE,
)
# 句子边界 = 标点**之后**紧跟空白 / 行尾。不能用"[^.!?]+[.!?]"那种切法：
# 型号里也有点（"TW11-3216-P3.0 cabinets"），会把一句话切碎（实测踩过：
# 切碎后 "0 cabinets and suit …" 被当成评论句，正文被削掉半句）。
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？])(?:\s+|$)")


def sentence_spans(text: str):
    """返回 [(start, end, 句子原文), …]（边界按上面的规则，型号不会被切碎）。"""
    source = str(text or "")
    spans = []
    start = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(source):
        chunk = source[start:match.end()]
        if chunk.strip():
            spans.append((start, match.end(), chunk))
        start = match.end()
    tail = source[start:]
    if tail.strip():
        spans.append((start, len(source), tail))
    return spans


def is_fact_free_commentary(sentence: str, *, max_words: int = 40) -> bool:
    """这句是不是"没有事实"的收尾评论（可以安全删掉）。"""
    body = str(sentence or "").strip()
    if not body or not _COMMENTARY_OPENER_RE.match(body):
        return False
    if _COMMENTARY_FACT_RE.search(body):     # 带排布数字 = 带事实
        return False
    if _COMMENTARY_ASK_RE.search(body):      # 问句 / 请求句归收口逻辑管
        return False
    return len(body.split()) <= int(max_words or 40)


def strip_fact_free_commentary(text: str) -> str:
    """删掉正文里"没有事实"的收尾评论句；删空了就原样返回。"""
    source = str(text or "")
    spans = [
        (start, end)
        for start, end, chunk in sentence_spans(source)
        if is_fact_free_commentary(chunk)
    ]
    if not spans:
        return source
    pieces, cursor = [], 0
    for start, end in sorted(spans):
        pieces.append(source[cursor:start])
        cursor = max(cursor, end)
    pieces.append(source[cursor:])
    cleaned = re.sub(r"[ \t]{2,}", " ", "".join(pieces))
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned or source


# ── "复述客户需求"的前置从句（客户口径 2026-09-28）──────────────────────────
# 实测客户可见文本：
#   "For a permanent installation at roughly 3m by 5m with a viewing distance
#    around 5m, the model I recommend is TW11-3216-P3.0."
# 前半句全是客户已经说过的需求 —— 客户要求"不要重复客户需求，直接推荐"。
# 只在这种"复述从句 + 后半句含型号"的形态下裁掉从句；后半句不含型号时一律不动
# （避免把 "Since your wall is 3m x 5m, we can lay it out two ways." 这类带信息的句子改坏）。
_ECHO_PREFIX_RE = re.compile(r"^\s*(?:for|since|because|as)\b([^,]{0,140}),\s*", re.IGNORECASE)
_MODEL_CODE_RE = re.compile(r"\bTW\s?\d{2}\s*[-\s]\s*[A-Za-z0-9.-]{2,}", re.IGNORECASE)


def strip_requirement_echo_prefix(text: str) -> str:
    """裁掉"复述客户需求"的前置从句（仅在从句后面紧跟型号时）。"""
    source = str(text or "")
    spans = sentence_spans(source)
    if not spans:
        return source
    pieces, cursor, changed = [], 0, False
    for start, end, chunk in spans:
        body = chunk.strip()
        rewritten = body
        match = _ECHO_PREFIX_RE.match(body)
        if match:
            clause = match.group(1)
            rest = body[match.end():].strip()
            if (
                rest
                and _MODEL_CODE_RE.search(rest)
                and not _MODEL_CODE_RE.search(clause)
            ):
                rewritten = rest[0].upper() + rest[1:]
        if rewritten != body:
            changed = True
            trailing = chunk[len(chunk.rstrip()):]
            pieces.append(source[cursor:start])
            pieces.append(rewritten + trailing)
            cursor = end
    if not changed:
        return source
    pieces.append(source[cursor:])
    return "".join(pieces)
