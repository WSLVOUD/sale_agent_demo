"""Deterministic fallback responses that avoid unsupported product claims."""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence, Tuple

from .language_utils import _lang



# ── "没有匹配的产品" → 改成"能不能放宽某个参数"（客户口径）─────────────────
# 禁止对客户说"找不到 / 没有匹配的产品"，一律改成邀请客户放宽条件。
_NO_PRODUCT_RE = re.compile(
    r"no matching products?|no suitable (?:outdoor |indoor )?model|"
    r"(?:i )?(?:could ?n[o']t|cannot|can't|am unable to) find (?:a|any|the) "
    r"(?:model|product|match)|couldn't find a model|"
    r"no (?:models?|products?) (?:found|available|match)|"
    r"没有匹配的?(?:产品|型号)|没有合适的?(?:产品|型号)|"
    r"找不到(?:合适的?|匹配的?|适合的?)?(?:产品|型号|屏|大屏|方案)|"
    r"(?:没有|未)找到(?:合适的?|匹配的?)?(?:产品|型号)|"
    r"无匹配(?:产品|型号)|无法推荐|推荐不出来",
    re.IGNORECASE,
)


_RELAXATION_ANSWERS = {
    "en": (
        "Let's take a slightly different angle — if one of the requirements can be relaxed "
        "(for example the pixel pitch, the screen size, or the viewing distance), I can match "
        "a model for you right away.",
        "Happy to get you the closest fit — would you be open to adjusting one requirement, "
        "say the pixel pitch or the screen size? Then I can put the right options in front of you.",
        "One quick option: if any of the requirements is flexible — pitch, size, or installation — "
        "I can match a suitable model immediately.",
        "If you can give a little on one of the conditions (pitch, brightness or screen size), "
        "I'll find you the best matching model straight away.",
    ),
    "zh": (
        "我们换个角度：如果某个条件可以放宽一点（比如点间距、屏体尺寸或观看距离），"
        "我马上就能帮您匹配到合适的型号。",
        "方便的话，看看哪个条件能松一点（例如点间距、亮度或尺寸），我好帮您找到最合适的型号。",
        "只要有一个条件可以灵活一点（点间距 / 尺寸 / 安装方式都行），我就能立刻帮您匹配合适的型号。",
    ),
}


# LCD / IFP 的"放宽条件"话术（客户口径 2026-09-30）：
# 上面那套是 **LED 口径**（点间距 / 观看距离 / 箱体），LCD 会话里出现就会变成
# "if one of the requirements can be relaxed (for example the pixel pitch…)"
# —— 客户一眼就知道系统串链路了。LCD 只谈它自己的条件：尺寸 / 拼缝 / 分辨率 / 安装。
_RELAXATION_ANSWERS_LCD = {
    "en": (
        "If one of the LCD requirements can be relaxed (for example the panel size, the bezel "
        "width, or the resolution), I can match a model for you right away.",
        "Happy to get you the closest fit, would you be open to adjusting one requirement, "
        "say the panel size or the bezel width? Then I can put the right options in front of you.",
        "One quick option: if any of the LCD requirements is flexible, size, bezel or "
        "resolution, I can match a suitable model immediately.",
    ),
    "zh": (
        "我们换个角度：如果某个条件可以放宽一点（比如面板尺寸、拼缝宽度或分辨率），"
        "我马上就能帮您匹配到合适的型号。",
        "方便的话，看看哪个条件能松一点（例如尺寸、拼缝或分辨率），我好帮您找到最合适的型号。",
    ),
}


# 判定"这一轮属于 LCD / IFP 链路"时用的产品族写法
_LCD_FAMILIES = ("lcd", "ifp", "interactive flat panel")



def has_no_product_phrase(text: str) -> bool:
    """回复里是否出现了"找不到 / 没有匹配产品"这类话术。"""
    return bool(_NO_PRODUCT_RE.search(str(text or "")))



def relaxation_answer(
    language: Optional[str] = None,
    seed: int = 0,
    *,
    product_family: str = "",
) -> str:
    """"能不能放宽某个参数"的应答（多种说法轮换）。

    ``product_family``：``"lcd"`` / ``"ifp"`` 时用 LCD 口径的说法
    （尺寸 / 拼缝 / 分辨率），不传或 ``"led"`` 时保持原来的 LED 口径
    （点间距 / 观看距离）。LCD 会话里绝不能出现 LED 口径的放宽条件。
    """
    family = str(product_family or "").strip().lower()
    table = _RELAXATION_ANSWERS_LCD if family in _LCD_FAMILIES else _RELAXATION_ANSWERS
    variants = table.get(_lang(language)) or table["en"]
    return variants[seed % len(variants)]



def product_fallback_answer(
    products: Any,
    *,
    language: Optional[str] = None,
) -> str:
    """【已废弃，客户口径 2026-10】正文被清空时**不再**报"最接近的型号"。

    以前这里会输出 "Based on your requirements, the closest match is {model}.
    Shall I prepare the quotation?"。实测它在闲聊轮反复出现、把真正的回答盖掉
    （客户原话："总是遮挡了该回答的话"），而且客户只说了句 "yes" 也会被报一个型号，
    等于凭空给一个匹配结果。两个调用点（api 空回复分支、solution runner）已改为
    只邀请客户补充条件，本函数保留签名仅为兼容，永远返回空串，**不再产生任何型号文案**。
    """
    return ""



# 跑题/闲聊的确定性兜底说法（接住 + 委婉拉回产品，**不含任何型号与事实**）
_OFF_TOPIC_STEER: Dict[str, tuple] = {
    "en": (
        "Happy to chat — and whenever you're ready, tell me the screen size and where it "
        "is going, and I'll put the right model together for you.",
        "Good to hear from you. Whenever you want to pick the screen back up, the size and "
        "the installation setting are all I need to line up the right model.",
        "Noted — and no rush on my side. When you're ready to continue, just tell me the "
        "screen size and where it will be installed and I'll take it from there.",
    ),
    "zh": (
        "随时聊 —— 您方便时告诉我屏幕尺寸和装在哪里，我就把合适的型号配给您。",
        "收到。想继续选屏的时候，把尺寸和安装场景告诉我就行，我这边接着帮您配型号。",
        "没问题，不着急。等您想继续了，告诉我屏幕尺寸和安装位置，我马上帮您选。",
    ),
}



def is_relaxation_answer(text: str) -> bool:
    """这段回复是不是"放宽条件"的确定性兜底文案（精确匹配全部变体）。

    用途**不是**判断客户说了什么（那必须靠语义理解），而是拦住**我们自己**生成的
    固定模板 —— 它由本模块的文案表产生，所以在这里精确比对最可靠，也不会误伤
    正常回复。客户口径 2026-10：推荐完之后这句话反复出现，必须能可靠识别并拦掉。
    """
    key = " ".join(str(text or "").split()).strip().lower()
    if not key:
        return False
    for table in (_RELAXATION_ANSWERS, _RELAXATION_ANSWERS_LCD):
        for variants in table.values():
            for variant in variants:
                if " ".join(str(variant).split()).strip().lower() == key:
                    return True
    return False



def off_topic_steer_answer(language: Optional[str] = None, seed: int = 0) -> str:
    """跑题/闲聊的确定性兜底：接住 + 委婉拉回产品，**不带任何型号与事实**。

    客户口径（2026-10）：客户问 "do u like watching TV"，正文却变成
    "Based on your requirements, the closest match is TW21-3216-P2.5." ——
    把型号包装成"他需求的答案"就是凭空捏造。宁可只给一句过渡，也不许报型号。
    """
    variants = _OFF_TOPIC_STEER.get(_lang(language)) or _OFF_TOPIC_STEER["en"]
    return variants[seed % len(variants)]



# 客户"同意推进"时的确定性兜底（多句轮换，**不承诺价格 / 交期**）
_QUOTE_CONFIRMATION: Dict[str, tuple] = {
    "en": (
        "Perfect — I'll get the quotation put together for your screen and send it over "
        "shortly.",
        "Great, thanks for confirming. I'll start on the quotation now and come back to "
        "you with it as soon as it's ready.",
        "Noted, and thank you. I'm putting the quotation together for that screen — I'll "
        "send it over in a moment.",
        "Sounds good. Let me pull the quotation together for you now and follow up with it "
        "shortly.",
    ),
    "zh": (
        "好的，我这就把这块屏的报价单整理出来，稍后发您。",
        "收到，谢谢确认。我现在就去准备报价单，弄好马上发您。",
        "没问题，报价单我这就去做，稍等片刻给您。",
        "好的，我这就把报价整理好，随后发您。",
    ),
}



def quote_confirmation_answer(language: Optional[str] = None, seed: int = 0) -> str:
    """客户"同意推进报价"时的确定性兜底（多句轮换）。

    客户口径（2026-10）：推荐完之后客户回来一句 "yes"，应该得到"我去准备报价单，请稍等"，
    **不是** relaxation_answer 的"能不能放宽某个条件" —— 那句话和客户的确认完全不搭
    （客户实测反馈）。这里也刻意不只写一句，避免每次都是同一句死板话术。
    绝不承诺价格 / 折扣 / 交期日期。
    """
    variants = (
        _QUOTE_CONFIRMATION.get(_lang(language)) or _QUOTE_CONFIRMATION["en"]
    )
    return variants[seed % len(variants)]



# ── Phase 15：Best-effort 推荐时"缺了什么 + 会影响什么"的自然说法 ────────────
# 客户不知道某项需求 → 仍然按已有信息推荐，但必须**自然**说明缺的是什么、
# 可能影响什么；绝不能说"信息不足无法推荐"。
_UNKNOWN_IMPACT: Dict[str, Dict[str, Any]] = {
    "viewing_distance": {
        "en": ("the exact viewing distance",
               "the final pixel pitch may need a small adjustment once you know it"),
        "zh": ("具体的观看距离", "拿到后最终点间距可能还需要微调"),
    },
    "installation": {
        "en": ("the installation type",
               "I've assumed a fixed installation for now and can switch you to a rental "
               "series if that changes"),
        "zh": ("安装方式", "目前按固定安装来选，如果改成租赁我可以换租用系列"),
    },
    "size": {
        "en": ("the exact screen size",
               "the cabinet count and final screen dimensions can be worked out as soon as "
               "we have the width and height"),
        "zh": ("具体的屏体尺寸", "拿到宽高后就能算出箱体数量和最终屏体尺寸"),
    },
    "width": {
        "en": ("the target screen width",
               "the cabinet layout can be finalised once we have the width"),
        "zh": ("目标屏幕宽度", "有宽度后就能确定箱体排布"),
    },
    "height": {
        "en": ("the target screen height",
               "the cabinet layout can be finalised once we have the height"),
        "zh": ("目标屏幕高度", "有高度后就能确定箱体排布"),
    },
    "environment": {
        "en": ("the indoor/outdoor setup",
               "an outdoor install would need a brighter, weatherproofed model"),
        "zh": ("室内还是室外", "如果改成室外需要更亮、防护等级更高的型号"),
    },
    "purpose": {
        "en": ("the exact application",
               "the feature set can be tuned once we know how the screen will be used"),
        "zh": ("具体使用场景", "明确场景后功能配置还能再优化"),
    },
    "brightness": {
        "en": ("the required brightness level",
               "a different brightness option can be quoted if the site needs more"),
        "zh": ("亮度要求", "如果现场需要更高亮度可以再换型号"),
    },
    "pixel_pitch": {
        "en": ("the preferred pixel pitch",
               "we can move to a finer or coarser pitch whenever you decide"),
        "zh": ("偏好的点间距", "确定后可以在更细或更粗的点间距之间切换"),
    },
}


_UNKNOWN_ITEM_FALLBACK = {
    "en": ("one of the details", "I can fine-tune the recommendation once we have it"),
    "zh": ("其中一项细节", "拿到后可以再把推荐调得更准"),
}


_DEGRADED_TEMPLATES = {
    "en": (
        "One thing to flag: {items} {verb} not confirmed yet, so this is the best match "
        "for what you've told me — {impacts}.",
        "Just so it's clear: I've based this on your confirmed requirements, as {items} "
        "{verb} still open. {impacts_cap}.",
    ),
    "zh": (
        "有一点先说明：{items}还没确认，所以这是基于您已提供信息的最佳匹配 —— {impacts}。",
        "补充一句：目前是按您已确认的信息来选的，{items}还在待定，{impacts}。",
    ),
}



def missing_impact(slot: str, language: Optional[str] = None) -> Tuple[str, str]:
    """某个未确认槽位 → （"缺的是什么", "可能影响什么"）。"""
    lang = _lang(language)
    table = _UNKNOWN_IMPACT.get(slot)
    if not table:
        table = {"en": _UNKNOWN_ITEM_FALLBACK["en"], "zh": _UNKNOWN_ITEM_FALLBACK["zh"]}
    return tuple(table.get(lang) or table["en"])  # type: ignore[return-value]



def degraded_note(
    slots: Sequence[str],
    language: Optional[str] = None,
    seed: int = 0,
) -> str:
    """Phase 15：存在 unknown 字段时，给推荐话术补一句"缺什么 + 影响什么"。"""
    wanted = [str(slot) for slot in (slots or []) if str(slot).strip()]
    if not wanted:
        return ""
    lang = _lang(language)
    items = [missing_impact(slot, lang)[0] for slot in wanted]
    impacts = [missing_impact(slot, lang)[1] for slot in wanted]
    if lang == "zh":
        item_text = "、".join(items)
        impact_text = "；".join(impacts)
        verb = "还没确认"
    else:
        item_text = " and ".join(items)
        impact_text = " Also, ".join(impacts)
        verb = "is" if len(items) == 1 else "are"
    templates = _DEGRADED_TEMPLATES.get(lang) or _DEGRADED_TEMPLATES["en"]
    template = templates[seed % len(templates)]
    return template.format(
        items=item_text,
        verb=verb,
        impacts=impact_text,
        impacts_cap=impact_text[:1].upper() + impact_text[1:] if impact_text else "",
    ).strip()

__all__ = ['_DEGRADED_TEMPLATES', '_LCD_FAMILIES', '_NO_PRODUCT_RE', '_OFF_TOPIC_STEER', '_QUOTE_CONFIRMATION', '_RELAXATION_ANSWERS', '_RELAXATION_ANSWERS_LCD', '_UNKNOWN_IMPACT', '_UNKNOWN_ITEM_FALLBACK', 'degraded_note', 'has_no_product_phrase', 'is_relaxation_answer', 'missing_impact', 'off_topic_steer_answer', 'product_fallback_answer', 'quote_confirmation_answer', 'relaxation_answer']
