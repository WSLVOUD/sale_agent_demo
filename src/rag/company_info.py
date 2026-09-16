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
        # 代理商问题必须说清三件事：只有中国一个点、自有工厂、因此成本更低
        "We are {company} in {location} — that is our only site, and we do not have distributors or "
        "branch offices elsewhere. We run our own factory there, so there is no middleman between us "
        "and you and the cost stays lower. Overseas customers work with our Shenzhen team directly.",
        "There is no local agent or dealer for us in your market — {company} is based in {location} and "
        "that is our only location. Because we manufacture in our own factory, we keep the cost down "
        "and pass that on to you; you deal with our Shenzhen team directly.",
        "Good question — we do not work through local distributors. Everything is handled from "
        "{company} in {location}, our only site, where we also own the factory. That removes the "
        "middleman markup, so the price stays competitive, and you buy straight from us.",
        "Right now {location} is our only office and factory ({company}); we have no branch or "
        "distributor abroad. Making the screens ourselves in our own factory keeps the overhead low, "
        "which means a better price for you — and our Shenzhen team supports overseas projects directly.",
    ),
    "zh": (
        # 代理商问题必须说清三件事：只有中国一个点、自有工厂、因此成本更低
        "我们只有中国这一个点 —— {company} 在{location}，没有当地代理商、办事处或分公司。"
        "我们自己有工厂，中间没有环节，所以成本能压下来；海外客户都是深圳团队直接对接。",
        "我理解你的问题：我们在当地没有代理商/经销商，公司只有{location}这一处（{company}），"
        "工厂也是我们自己的。自产直供省掉了中间商，开销更低，价格自然更有优势，海外项目由深圳团队直接跟进。",
        "坦率说，我们目前只在{location}有公司和工厂（{company}），没有其它地区的分支机构或代理。"
        "好处是我们从自己的工厂直接供货，中间没有加价环节，成本更低；有需要随时找我们深圳团队。",
        "我们是 {company}，公司和工厂都在{location}，这是唯一的一个点，海外没有设代理或办事处。"
        "因为是自己工厂生产、直接对接客户，省掉了中间环节，所以整体开销更低、报价也更有竞争力。",
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
