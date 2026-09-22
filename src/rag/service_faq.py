"""售后 / 服务类固定口径（客户口径 2026-09-18）：

  1. 说明书与图纸：下单后随货一并发给客户，我们也可以提供安装图纸、说明书等技术资料；
  2. 现场安装：一般不提供现场安装（建议当地找安装公司，更省成本），
     但每个订单都提供安装指导说明书，随货一并发给客户；
  3. 质保：默认 1 年，可付费延长。

原则（和公司信息、交期一致）：
  - **事实由代码给定**，LLM 只做英文润色：不许增删事实、不许报价、不许承诺标准回答
    里没有的内容；
  - **客户没问就不主动提**（尤其是质保）；
  - 回复必须是英文（策略=en 时），中文润色结果直接丢弃、退回英文标准回答。
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

FAQ_MANUAL = "manual"
FAQ_INSTALLATION = "installation"
FAQ_WARRANTY = "warranty"

# ── 客户在问哪一类 ─────────────────────────────────────────────────────────
# 注意顺序：先看"现场安装"，再看"说明书/图纸"，最后才是"质保"。
_INSTALLATION_QUESTION_RE = re.compile(
    r"包安装|安装服务|上门安装|现场安装|派人安装|提供安装|负责安装|你们安装|谁来安装|"
    r"安装(?:吗|么|不)|安装费|"
    r"\b(?:do you (?:also )?install|on-?site install(?:ation)?|installation service|"
    r"provide installation|install it for (?:me|us)|who installs?)\b",
    re.IGNORECASE,
)
_MANUAL_QUESTION_RE = re.compile(
    r"说明书|安装图|图纸|技术资料|产品手册|手册|资料|"
    r"\b(?:user manual|manual|drawings?|installation drawings?|datasheet|documentation|"
    r"technical (?:documents?|files?|support))\b",
    re.IGNORECASE,
)
_WARRANTY_QUESTION_RE = re.compile(
    r"质保|保修|售后保障|保几年|几年质保|"
    r"\b(?:warranty|guarantee|guaranty)\b",
    re.IGNORECASE,
)

# ── 标准回答（事实，英文）─────────────────────────────────────────────────
FACTS: dict[str, str] = {
    FAQ_MANUAL: (
        "Yes — the user manual and installation drawings ship together with your goods once the "
        "order is placed. We can also provide installation drawings, manuals and other technical "
        "support."
    ),
    FAQ_INSTALLATION: (
        "In general we do not provide on-site installation — it is usually more cost-effective to "
        "use a local installation company. That said, we include an installation guide with every "
        "order, and it ships together with your goods."
    ),
    FAQ_WARRANTY: (
        "Our products come with a 1-year warranty by default, and the warranty can be extended for "
        "an additional fee."
    ),
}

_FAQ_POLISH_PROMPT = """你是 LED 显示屏产品的销售，正在微信上和客户聊天。
下面这段是公司对客户问题的**标准回答（事实不能改）**，请用你自己的话自然地说一遍。

硬性规则：
1. 意思必须和标准回答**完全一致**：不能增加、不能删减、不能改动任何事实或数字；
   不要承诺标准回答里没有的内容（不提价格、不提交期、不提额外承诺）。
   特别注意：标准回答里"随货提供安装指导说明书"**不等于**提供现场安装服务，
   绝不能说成 "installation is included" / "we can install it for you"。
2. 只回答客户问的这件事，**不要反问**客户的需求（系统会另外接需求问题）。
3. 1~2 句话，口语化，像真人销售说话；不要客套话堆砌。
4. 不要用 "—"，不要用 markdown，不要用引号，不要换行。
5. 语言要求：{language_rule}
6. 直接输出这一句话，不要 JSON、不要解释。

客户问：{message}
标准回答：{fact}"""


def detect_service_faq(message: str) -> Optional[str]:
    """客户这句话在问哪一类售后 / 服务问题（都不是则返回 None）。"""
    text = str(message or "")
    if not text.strip():
        return None
    if _INSTALLATION_QUESTION_RE.search(text):
        return FAQ_INSTALLATION
    if _MANUAL_QUESTION_RE.search(text):
        return FAQ_MANUAL
    if _WARRANTY_QUESTION_RE.search(text):
        return FAQ_WARRANTY
    return None


def service_faq_fact(message: str) -> Optional[str]:
    """标准回答（英文事实句）；客户没问这类问题则返回 None。"""
    kind = detect_service_faq(message)
    return FACTS.get(kind) if kind else None


def _numbers(text: str) -> set:
    return set(re.findall(r"\d+(?:[.,]\d+)?", str(text or "")))


# ── 与"不提供现场安装"相矛盾的说法（实测 bug）──────────────────────────────
# 客户日志：回复里同时出现 "we do not provide on-site installation" 和
# "Yes, installation is included." —— 后者是 LLM 把"随货说明书"说成了"包安装"。
_INSTALLATION_CONTRADICTIONS = re.compile(
    r"\byes\b[^.!?]{0,40}\binstallation\b"
    r"(?!\s*(?:guide|manual|drawings?|instructions?|documents?))"
    r"[^.!?]{0,20}\b(?:included|provided|covered)\b|"
    # 实测（2026-09-21）："Yes, we do provide installation." / "our team handles the
    # on-site setup as part of the project." 都没被旧规则拦住 → 补上这些说法。
    r"\bwe\s+(?:do\s+)?(?:provide|offer|arrange|handle|cover|include)\b"
    r"[^.!?]{0,30}\binstallation\b"
    r"(?!\s*(?:guide|manual|drawings?|instructions?|documents?))|"
    r"\b(?:our team|we)\b[^.!?]{0,40}\b(?:handle|handles|take care of|takes care of|do|does)\b"
    r"[^.!?]{0,30}\bon-?site\b|"
    r"\bon-?site\s+(?:setup|install(?:ation)?)\b[^.!?]{0,30}"
    r"\b(?:as part of|included|covered|provided)\b|"
    r"\binstallation\b(?!\s*(?:guide|manual|drawings?|instructions?|documents?))"
    r"[^.!?]{0,20}\b(?:is|will be|would be)\b[^.!?]{0,20}"
    r"\b(?:included|provided|covered|part of the (?:order|price|package))\b|"
    r"\bwe\b(?![^.!?]{0,40}\b(?:not|never|no)\b)[^.!?]{0,40}"
    r"\b(?:provide|offer|arrange|handle|include)\b[^.!?]{0,30}"
    r"\bon-?site installation\b|"
    r"\b(?:we|i)(?:'ll| will| can)?\s+install\s+(?:it|the screen|the display|the wall)\b|"
    r"包安装|含安装|提供安装服务|上门安装",
    re.IGNORECASE,
)

# 已经说清"不提供现场安装"的句子 → 不能当成矛盾删掉
_INSTALLATION_NEGATION = re.compile(
    r"(?:\bdo(?:es)?\s+not\b|\bdo(?:n'?t|esn'?t)\b|\bcannot\b|\bcan'?t\b|\bnever\b|\bno\b"
    r"|不提供|不含|不包|没有)",
    re.IGNORECASE,
)


def strip_contradictory_installation_claims(text: str) -> str:
    """删掉"包安装 / installation is included"这类与标准口径矛盾的句子。"""
    source = str(text or "")
    if not source:
        return ""
    # 只在"句末 + 空白"处切句，避免把 "TW11-3216-P3.0" 这种型号切断
    sentences = re.split(r"(?<=[.!?。！？])(?=\s)", source)
    kept = [
        sentence for sentence in sentences
        if sentence.strip() and not (
            _INSTALLATION_CONTRADICTIONS.search(sentence)
            and not _INSTALLATION_NEGATION.search(sentence)
        )
    ]
    dropped = len([s for s in sentences if s.strip()]) - len(kept)
    if dropped:
        logger.warning("ServiceFAQ: dropped %d contradictory installation claim(s)", dropped)
    return " ".join(sentence.strip() for sentence in kept).strip()


def sanitize_service_reply(response: str, kind: Optional[str]) -> str:
    """按服务口径清掉回复里自相矛盾的说法（目前只有"现场安装"这一类）。"""
    if kind == FAQ_INSTALLATION:
        return strip_contradictory_installation_claims(response)
    return str(response or "")


def service_faq_reply(message: str, *, language: str = "en") -> Optional[str]:
    """客户问了这类问题 → 返回**英文润色**后的回答；没问则返回 None。

    润色失败 / 结果夹带中文 / 冒出标准回答里没有的数字时，直接退回标准回答
    （客户口径：必须是英文，但不照抄原话）。
    """
    fact = service_faq_fact(message)
    if not fact:
        return None
    try:
        from src.core.llm import get_llm
        from src.rag.query_understanding import response_language_rule
        from src.rag.reply_composer import contains_cjk

        prompt = _FAQ_POLISH_PROMPT.format(
            message=str(message)[:200],
            fact=fact,
            language_rule=response_language_rule(language),
        )
        response = get_llm(temperature=0.3).invoke(prompt)
        text = str(getattr(response, "content", response) or "").strip()
        text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "").strip().strip("\"'“”")
        text = " ".join(text.split())
        if not text or len(text) > 400:
            raise ValueError("polished reply empty or too long")
        if contains_cjk(text):
            raise ValueError("polished reply contains CJK")
        if not _numbers(text).issubset(_numbers(fact)):
            raise ValueError("polished reply invented numbers")
        if detect_service_faq(message) == FAQ_INSTALLATION and (
            _INSTALLATION_CONTRADICTIONS.search(text)
        ):
            # 把"随货说明书"说成"包安装"这类自相矛盾 → 直接用标准回答
            raise ValueError("polished reply contradicts the installation policy")
        return text
    except Exception as exc:  # pragma: no cover - 网络/额度问题
        logger.warning("Service FAQ polish failed, using standard answer: %s", exc)
        return fact


__all__ = [
    "FACTS",
    "FAQ_INSTALLATION",
    "FAQ_MANUAL",
    "FAQ_WARRANTY",
    "detect_service_faq",
    "service_faq_fact",
    "service_faq_reply",
    "sanitize_service_reply",
    "strip_contradictory_installation_claims",
]
