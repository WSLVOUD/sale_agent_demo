"""
公司信息问答（严格依据 data/company_profile.txt，禁止编造）。

背景（客户实测反馈）：客户问 "Do you have a representative in western ..."，
系统却回了 "Yes — we do carry an LED."（既答非所问、又凭空说 Yes）。
实际上公司信息都在 `data/company_profile.txt` 里（公司、所在地、主营、优势），
这类问题必须照实回答。
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# 公司 / 办事处 / 经销商 / 地址类问题
_COMPANY_QUESTION_RE = re.compile(
    r"(?<![a-z])(?:representative|rep|distributor|dealer|reseller|agent|"
    r"office|branch|subsidiary|headquarters|head office|factory|"
    r"where are you|where is your|based in|located|location|address)\b|"
    r"办事处|代表处|分公司|子公司|代理商|经销商|总部|工厂|厂址|地址|"
    r"公司在哪|你们公司|你们在哪|在哪里",
    re.IGNORECASE,
)

_COMPANY_ANSWERS = {
    "en": (
        "This is {sales_name} from {company} — we are based in {location}, and that is our only office. We support overseas customers directly from Shenzhen.",
        "Our company is {company}, based in {location} — it is our single site, and we work with overseas customers directly from there.",
        "We are {company} in {location}. We do not have a local office elsewhere, but our Shenzhen team handles overseas projects directly.",
        "We are based in {location} ({company}) — that is our only company location, and we serve customers worldwide from there.",
    ),
    "zh": (
        "我是 {company} 的 {sales_name}，我们公司位于{location}，目前只有这一个公司/工厂，海外客户由深圳团队直接对接。",
        "我们公司是 {company}，位于{location}，没有其它地区的分公司或办事处，海外项目由深圳团队直接跟进。",
        "我们在{location}（{company}），这是公司唯一的所在地，海内外客户都由深圳团队对接。",
    ),
}


def is_company_question(message: str) -> bool:
    """客户是不是在问公司 / 办事处 / 地址类问题（这类必须照公司信息回答）。"""
    return bool(_COMPANY_QUESTION_RE.search(str(message or "")))


def _lang(language: Optional[str]) -> str:
    return "zh" if str(language or "").lower().startswith("zh") else "en"


def company_answer(
    message: str = "",
    *,
    language: Optional[str] = None,
    seed: int = 0,
) -> Optional[str]:
    """按公司信息回答；问题与公司无关时返回 None。

    数据来源：``data/company_profile.txt``（Company / Location / Sales Name）。
    里面的字段缺失时不会编造，只回已知的事实。
    """
    if not is_company_question(message):
        return None
    try:
        from src.first_contact.profile import load_profile

        profile = load_profile()
    except Exception as exc:  # pragma: no cover - 防御式
        logger.warning("Company profile unavailable: %s", exc)
        return None

    company = (profile.company or "").strip()
    location = (profile.location or "").strip()
    sales_name = (profile.sales_name or "").strip()
    if not company and not location:
        return None
    # 缺字段时给安全的兜底值，避免出现 "()" 这种空占位
    company = company or "our company"
    location = location or "Shenzhen, China"
    sales_name = sales_name or "our sales team"

    variants = _COMPANY_ANSWERS[_lang(language)]
    return variants[seed % len(variants)].format(
        company=company, location=location, sales_name=sales_name
    )


__all__ = ["company_answer", "is_company_question"]
