"""Recommendation node - generates product recommendations."""
from typing import Dict, Any, List, Optional
import logging
import re

from ..state import SolutionState
from ....core.llm import get_llm
from ....rag.rerank import is_display_candidate, get_product_category, has_environment_conflict, is_ifp_product
from ....utils.ifp_intent import user_messages_text

logger = logging.getLogger(__name__)


RECOMMEND_PROMPT = """Customer needs: {requirement}

Product data:
{product}

{additional_context}

Generate one natural, conversational recommendation line that includes:
1. Complete product model name
2. 1-2 core selling points / reasons to recommend
3. If necessary, 1-2 key specs (e.g. pixel pitch, dimensions)

Requirements:
- Conversational, like recommending to a friend — do not recite the manual
- Greeting is only needed on the first recommendation; subsequent ones start directly with the product
- Strictly no more than 60 characters, shorter is better
- No markdown, no lists, no bold
- Never reveal internal info: do not say "I have the records", "the product data shows", "not in the data", "from the database", "I queried", etc.
- If a parameter is missing, do not explain the data gap or say "no specific info for this distance"; only recommend products you can confirm match, or briefly ask for one missing parameter
- Do not mention budget, price ranges, or cost
- 【Full model name rule】Always output the complete model name — never just the suffix
- Do not repeat or restate what the customer already stated (e.g. if they've already said outdoor, stage, or indoor, don't repeat those words)
- Only mention positive reasons — never say "not suitable", "not recommended", "doesn't match", or any negative phrasing
- Always answer based on the product data — do not make up content not in the data
- 【Hard environment rule】Do not repeat the customer's stated environment (outdoor, indoor, stage, etc.) in your reply. Output the model and core selling points directly. Never substitute indoor products for outdoor needs or vice versa.
- 【No-product rule】Never tell the customer that nothing matches, that you can't find a product, or that the database has no matching item. If no product fits exactly, invite them to relax one requirement instead (e.g. "if the pixel pitch or screen size can be a little flexible, I can match a model for you") — never state that no product exists.

Output the recommendation directly. Use as many sentences as the content needs —
never cut a sentence short to stay within a sentence count. Always give the
complete model name and every fact the customer needs to act on; still do not
pad, repeat, or add anything that is not in the product data:"""


FOLLOW_UP_PROMPT = """Based on the recommended products, generate 1 short follow-up sentence.

Already recommended: {recommendations}

Requirements:
1. 1 sentence, conversational, like chatting with a friend
2. No emoji, no bold, no lists
3. No more than 15 words
4. Never ask about budget, price, or cost
5. Generate only from the content of already-recommended products
6. Never make up price ranges, cost tiers, or value comparisons not in the data.

Output directly:"""


def _get_product_text(product) -> str:
    """Extract text from a product dict or Document object."""
    if hasattr(product, "page_content"):
        return product.page_content
    if isinstance(product, dict):
        return product.get("text", "")
    return str(product)


def _get_product_metadata(product) -> Dict[str, Any]:
    """Extract metadata from a product dict or Document object."""
    if hasattr(product, "metadata"):
        return product.metadata or {}
    if isinstance(product, dict):
        return product.get("metadata", {}) or {}
    return {}


def _normalize_text(text: str) -> str:
    """Normalize common punctuation variants for regex matching."""
    return (text
        .replace("\uff1a", ":")
        .replace("\u3000", " ")
        .replace("\u2011", "-")
        .replace("\u2010", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-"))


def product_identifier(product, user_asked_model: str = "") -> str:
    """Extract the real product model name from a retrieved chunk."""
    if hasattr(product, "page_content"):
        text = _normalize_text(product.page_content)
        metadata = product.metadata or {}
    else:
        metadata = _get_product_metadata(product)
        text = _normalize_text(_get_product_text(product))

    # Check if user asked about a specific model
    if user_asked_model:
        asked_normalized = _normalize_text(user_asked_model)
        if asked_normalized in text or asked_normalized.upper() in text.upper():
            return user_asked_model

    # Try various patterns
    for pattern in [
        r"(?mi)^Model:\s*(.+?)$",
        r"(?mi)^Product internal models:\s*(.+?)$",
        r"(?mi)^Product:\s*(.+?)$",
    ]:
        for match in re.finditer(pattern, text, re.MULTILINE):
            candidate = match.group(1).strip().split(";")[0].split("|")[0].split(",")[0].strip()
            candidate = re.sub(r"\s+Series\s*$", "", candidate, flags=re.IGNORECASE)
            if candidate:
                return candidate

    # LED file pattern: TW21-COB-P0.6
    m = re.search(r"(?m)^(TW\d+[\w.-]*)\b", text)
    if m:
        return m.group(1).strip()

    # IFP pattern: TxxOmni-Xn
    m = re.search(r"(?m)^(T\d{2}Omni[\w\u2011\u2010\u2012\u2013\u2014\-\.]+\d[\w\u2011\u2010\u2012\u2013\u2014\-\.]*)\b", text)
    if m:
        model = m.group(1).strip()
        for hyphen in ("\u2011", "\u2010", "\u2012", "\u2013", "\u2014", "\u2015", "\ufe63", "\uff0d"):
            model = model.replace(hyphen, "-")
        return model

    # Last resort: uppercase token with digits and hyphen
    m = re.search(r"(?:^|\s)([A-Z][A-Z0-9/\-'.\u2013\u2014]*\d[A-Z0-9/\-'.\u2013\u2014]*)\b", text)
    if m:
        return m.group(1).strip()

    return metadata.get("id") or ""


def _normalize_model(model: str) -> str:
    """Normalize a model identifier for deduplication."""
    if not model:
        return ""
    raw = model.split(",")[0].strip()
    for hyphen in ("\u2011", "\u2010", "\u2012", "\u2013", "\u2014", "\u2015", "\ufe63", "\uff0d"):
        raw = raw.replace(hyphen, "-")
    return raw.upper()


def named_products(products: list, limit: int = 3, user_asked_model: str = "") -> list:
    """Keep the highest-ranked unique product models, up to ``limit`` entries."""
    named = []
    seen_keys = set()
    for product in products:
        model = product_identifier(product, user_asked_model=user_asked_model)
        key = _normalize_model(model)
        if not model:
            metadata = _get_product_metadata(product)
            text = _get_product_text(product)
            fallback_key = _normalize_model(metadata.get("id") or text[:80])
            if fallback_key in seen_keys:
                continue
            seen_keys.add(fallback_key)
            named.append(product)
        else:
            if key in seen_keys:
                continue
            seen_keys.add(key)
            named.append(product)
        if len(named) == limit:
            break
    return named


def _generate_follow_up(llm, recommendations_text: str, requirement_text: str) -> str:
    """Generate a natural conversational follow-up after the recommendation list."""
    try:
        prompt = FOLLOW_UP_PROMPT.format(
            requirement=requirement_text or "未明确",
            recommendations=recommendations_text,
        )
        response = llm.invoke(prompt)
        text = (response.content if hasattr(response, "content") else str(response)).strip()
        text = text.strip("\"'`").strip()
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        if len(text) > 60:
            text = text[:60].rstrip("，,;；") + "。"
        return text if len(text) <= 200 else ""
    except Exception as error:
        logger.warning("Follow-up generation failed: %s", error)
        return ""


_SERVICE = None

def _get_service():
    """进程内复用 RecommendationService（统一入口，强制先过 Gate）。"""
    global _SERVICE
    if _SERVICE is None:
        from ....rag.recommendation_service import RecommendationService

        _SERVICE = RecommendationService()
    return _SERVICE


def _evidence_products(recommendations, raw_products):
    """把选中的型号与 RAG 检索证据对应起来（RAG 只提供事实，不参与选型）。"""
    by_model = {}
    for item in raw_products or []:
        meta = _get_product_metadata(item)
        name = meta.get("model") or meta.get("product_id") or ""
        if name and name not in by_model:
            by_model[name] = item

    products = []
    for rec in recommendations or []:
        model = rec.get("model")
        evidence = by_model.get(model)
        if evidence is not None:
            products.append(evidence)
        else:
            products.append({
                "id": model,
                "text": ", ".join([
                    f"Model {model}",
                    f"{rec.get('series_id')} series",
                    f"pixel pitch {rec.get('pixel_pitch_mm')}mm",
                    f"brightness {rec.get('brightness_nit')}nit",
                    f"cabinet {rec.get('cabinet_size_mm')}",
                    f"{rec.get('modules_per_cabinet')} modules per cabinet",
                ]),
                "metadata": {
                    "model": model,
                    "product_id": model,
                    "series_id": rec.get("series_id"),
                    "display_type": "LED",
                    "indoor": rec.get("indoor"),
                    "outdoor": rec.get("outdoor"),
                    "is_rental": rec.get("installation") == "rental",
                    "pixel_pitch_mm": rec.get("pixel_pitch_mm"),
                    "brightness_nit": rec.get("brightness_nit"),
                },
            })
    return products


def _express_recommendation(
    recommendations,
    profile,
    calculation,
    additional_requirements,
    customer_text,
    need_size_question=False,
    language: str = "en",
    degraded_slots=(),
    follow_up: bool = False,
    previous_models=(),
    calculation_variants=None,
    multi_screen_brief: str = "",
):
    """用一次 LLM 调用把确定性结论表达成销售话术；失败时退化为模板。"""
    # 客户口径（2026-09-18）：**推荐时只报一个型号**，其他型号一律不提。
    # 只有客户主动问"还有别的推荐吗"（follow_up）时，才换成另一个型号给他。
    top_index = 0
    if follow_up and len(recommendations) > 1:
        # 换一个**还没给他看过**的型号（按引擎排名依次给：第 1 好的 → 第 2 好的 → …）
        seen = {str(model) for model in (previous_models or [])}
        top_index = next(
            (
                index
                for index, rec in enumerate(recommendations)
                if str(rec.get("model")) not in seen
            ),
            1,
        )
        if top_index == 0:
            # 候选都被推荐过了 → 至少给排行第二的，而不是把首选再说一遍
            top_index = 1
    top = recommendations[min(top_index, len(recommendations) - 1)]
    # 只把这个型号的实测参数交给 LLM —— 给它三行，它就容易把别的型号也念出来
    alternative_lines: list = []

    spec_lines = [
        f"- {top['model']}: pixel pitch {top['pixel_pitch_mm']}mm, "
        f"brightness {top['brightness_nit']}nit, "
        f"cabinet {top['cabinet_size_mm']}, "
        f"{top['modules_per_cabinet']} modules per cabinet"
        # 客户口径：客户没问质保就不主动提（质保由 FAQ 按 1 年 + 可付费延长 回答）
    ]
    # 备选款：说清"它比首选款多什么"，让客户按自己的偏好选（不提价格）
    reasons = "; ".join(top.get("reasons") or [])
    extra = ""
    if additional_requirements:
        extra = f"\nCustomer's additional requirements: {', '.join(additional_requirements)}"
    latest = ""
    if customer_text:
        latest = f"\nCustomer's latest message: {str(customer_text)[:200]}"
    # 多屏项目：这一段只写这块屏（不要重复招呼 / 不要写另一块屏 / 不要套同一个模板）
    brief = ""
    if multi_screen_brief:
        brief = (
            "\nMulti-screen brief (this paragraph covers ONE screen only — follow it "
            "exactly):\n" + str(multi_screen_brief).strip() + "\n"
        )
    calc_text = ""
    layouts = calculation_variants or (
        {"landscape": calculation} if calculation else {}
    )
    layout_lines = []
    for key in ("landscape", "portrait"):
        calc = layouts.get(key)
        if not calc:
            continue
        label = (
            "Vertical tiling (cabinets rotated 90 degrees)" if key == "portrait"
            else "Horizontal tiling"
        )
        layout_lines.append(
            f"- {label}: {calc['columns']}x{calc['rows']} = {calc['cabinet_count']} cabinets, "
            f"actual size {calc['actual_width_m']}m x {calc['actual_height_m']}m "
            f"({calc['area_sqm']} sqm), {calc['total_modules']} modules"
        )
    if layout_lines:
        calc_text = (
            "\nCalculated configurations (authoritative, do not recompute) — "
            "the customer must see BOTH options:\n" + "\n".join(layout_lines)
        )
    next_step_rule = (
        # 客户只是在问"还有没有别的推荐"：别再把尺寸当成单独一问，
        # 改成邀请客户补充需求（把尺寸/亮度/交期/安装方式当例子提一下），
        # 之后按"已有需求 + 新需求"再锁定一款（客户口径）。
        "4. Finish by inviting the customer to share any other requirements so you can lock in "
        "the right model for them — mention examples such as the target screen size, brightness, "
        "delivery or installation type. Do not ask for the size as a separate question here.\n"
        if follow_up
        else (
            "4. Finish by asking for the target screen width and height so you can work out "
            "the cabinet and module configuration.\n"
            if need_size_question and not calculation
            else (
                # 尺寸已经有了、箱体也算了 → 不许再问尺寸 / 提"尺寸还没定"
                "4. The screen size is already confirmed and the cabinet/module layout has been "
                "calculated — never ask for the size or dimensions again, and never say the size is "
                "still open. Finish with one short next-step question (for example preparing the "
                "quotation).\n"
            )
        )
    )

    # 客户口径：不再在推荐话术里写"某项还没确认 / 可能有偏差"这类说明（只给结论）。
    degraded_rule = ""

    # v2.0 Phase 14：回复语言由策略决定（默认英语，auto 时跟随客户语言）
    from ....rag.query_understanding import response_language_rule

    language_rule = response_language_rule(language)

    prompt = (
        "You are a sales engineer for LED display products. Write the recommendation reply.\n\n"
        "Selected model (already decided by the engineering system — do not change it):\n"
        f"{top['model']}\n\n"
        "Verified product data:\n" + "\n".join(spec_lines) + "\n\n"
        f"Recommendation reasons: {reasons}\n"
        + f"{calc_text}{extra}{latest}{brief}\n\n"
        "Rules:\n"
        "1. Mention the selected model with its full model code — it does not have to be the very first words.\n"
        # 客户口径（2026-09-28）：不要在推荐话术里复述客户的需求。
        # 实测："For a permanent installation at roughly 3m by 5m with a viewing
        # distance around 5m, the model I recommend is TW11-3216-P3.0. …" ——
        # 客户已经说过的话再说一遍，话术就变长了。直接推荐即可。
        "1b. Do NOT restate the customer's requirements. Never open with things like "
        "\"For your indoor/church/permanent install at 3m x 5m with a 5m viewing distance…\" "
        "or \"Since your wall is 3m x 5m…\" — they already told us. Start with the model "
        "(or one short reason about the model) and keep every other sentence about THIS model.\n"
        "2. Add 1-2 concrete selling points using ONLY the verified data above.\n"
        "2b. If the customer's latest message mentions other requirements (quality, lead time, installation, "
        "brightness, size, delivery…), acknowledge them in ONE short clause and tie the choice to them, "
        "so the reply answers what they just said instead of repeating the same text.\n"
        "3. If calculated configurations are given, present BOTH tiling options (horizontal and "
        "vertical) — for each one give the cabinet count (columns x rows = total) and the actual "
        "screen size. Never drop one of the two layouts. Do NOT add a closing comment about how the two "
        "layouts compare or how to choose between them (no \"Both options … so the choice comes down to …\") — "
        "the figures speak for themselves.\n"
        "3b. Mention ONLY the selected model above. Never name, hint at or compare any other model "
        "code, and never write \"If you want <something>, <OTHER MODEL>\". One model per reply.\n"
        "3c. NEVER mention price, price tier, cost, discount, budget or value for money — pricing is handled "
        "separately by the sales team.\n"
        + next_step_rule
        + degraded_rule +
        # 客户口径（2026-09-28）：推荐时**不要**字数上限 —— 型号 + 实测参数 +
        # 两种排布（箱体数 / 实际尺寸 / 模组数）必须完整给到客户，多屏时每一块
        # 屏都要完整。只要求"不注水"（不重复、不加数据里没有的内容）。
        f"5. Plain text only, no markdown, no bullets. Be concise: the complete recommendation "
        "(full model code, key specs, and every calculated tiling option) must be there, "
        "but do not pad, do not repeat, and never restate the customer's requirements or add "
        "commentary about the layouts.\n"
        "6. Vary your wording and sentence structure between replies — avoid any fixed template.\n"
        f"7. {language_rule}\n"
        "8. Never invent specs, never mention internal data sources.\n\n"
        "Reply:"
    )
    try:
        response = get_llm(temperature=0.3).invoke(prompt)
        text = (response.content if hasattr(response, "content") else str(response)).strip()
        text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "").strip()
        text = text.replace("**", "").replace("__", "")
        if text:
            return text
    except Exception as error:  # pragma: no cover - 网络/额度问题时的降级
        logger.warning("Recommendation expression LLM failed, using template: %s", error)

    import random as _random

    opener = _random.choice((
        "{model} looks like the best fit for what you described.",
        "Based on your requirements, I'd go with {model}.",
        "For this setup I'd suggest {model}.",
        "{model} matches your needs best.",
    )).format(model=top["model"])
    fallback = opener
    if top.get("reasons"):
        fallback += " " + "; ".join(top["reasons"][:2]) + "."
    if calculation:
        for key in ("landscape", "portrait"):
            calc = (calculation_variants or {}).get(key)
            if not calc:
                continue
            label = (
                "If the cabinets go in vertically (rotated 90 degrees)"
                if key == "portrait" else
                "With the cabinets in their standard horizontal position"
            )
            fallback += (
                f" {label}: {calc['columns']}x{calc['rows']} = {calc['cabinet_count']} cabinets "
                f"({calc['total_modules']} modules), actual size "
                f"{calc['actual_width_m']}m x {calc['actual_height_m']}m."
            )
    if follow_up:
        fallback += (
            " If you have any other requirements — the target screen size, brightness, delivery or "
            "installation type — tell me and I'll lock in the right model for you."
        )
    elif need_size_question and not calculation:
        fallback += " Could you share the target screen width and height so I can work out the cabinet and module configuration?"
    else:
        fallback += " Would you like me to prepare a quotation for this configuration?"
    return fallback


def _alternative_difference(top: dict, other: dict) -> str:
    """备选款与首选款的差别（实现见 src/rag/alternatives.py，两处共用一份）。"""
    from ....rag.alternatives import alternative_difference

    return alternative_difference(top, other)


# 客户在问"还有没有别的推荐"（这种轮次说备选，而不是重讲首选 + 催尺寸）
_ALTERNATIVES_RE = re.compile(
    r"(?:还|另外|再)?有(?:没有)?(?:其他|其它|别的|别|更多|什么)?(?:的|一些|几款)?"
    r"(?:推荐|型号|选择|方案|款式|屏幕|屏)|"
    r"其他推荐|其它推荐|别的推荐|更多(?:选择|型号|推荐)|换一款|换个型号|"
    # 客户让销售"另外/再 帮我推荐一款" → 换一个型号给他（不是重新采集需求）
    r"(?:另外|再|又|还)(?:帮我|给我|帮忙)?(?:再)?推荐(?:一|几|两)?(?:款|个|种|台)|"
    r"\b(?:recommend|suggest)\s+(?:me\s+)?(?:another|one more|a different)\b|"
    r"\b(?:any other|other options?|other models?|more options?|alternatives?|anything else)\b",
    re.IGNORECASE,
)


_SIZE_ASK_RE = re.compile(
    r"(width|height|size|dimension|dimensions|wide|tall|尺寸|多宽|多高|宽\s*[x×*]?\s*高)",
    re.IGNORECASE,
)


def _ensure_size_question(answer: str, question: str) -> str:
    """缺尺寸时保证回复里一定带上尺寸追问（不依赖 LLM 是否遵守指令）。"""
    text = (answer or "").strip()
    if _SIZE_ASK_RE.search(text):
        return text
    if not text:
        return question
    return f"{text}\n\n{question}"


def _ensure_degraded_note(answer: str, slots, language: str = "en") -> str:
    """Phase 15：客户不知道某字段时，回复里**一定**带一句"缺什么 + 影响什么"。

    不依赖 LLM 是否听话：模型漏说就由这里确定性补上。
    """
    wanted = [str(slot) for slot in (slots or []) if str(slot).strip()]
    if not wanted:
        return answer
    from ....rag.reply_composer import degraded_note, missing_impact

    text = (answer or "").strip()
    # 只要已经提到"缺的那一项"或"可能的影响"，就认为说明到位了
    mentioned = False
    lowered = text.lower()
    for slot in wanted:
        item, impact = missing_impact(slot, language)
        for phrase in (item, impact):
            words = [w for w in re.split(r"[\s,，、]+", str(phrase).lower()) if len(w) > 3]
            if words and all(word in lowered for word in words):
                mentioned = True
                break
        if mentioned:
            break
    if mentioned:
        return text
    note = degraded_note(wanted, language)
    if not text:
        return note
    return f"{text}\n\n{note}"


_LCD_LOCK_FIELDS: tuple = (
    "lcd_category",
    "lcd_size_inch",
    "lcd_resolution",
    "lcd_is_splicing",
    "lcd_splicing_layout",
    "lcd_screen_count",
    "lcd_bezel_mm",
    "lcd_touch_required",
    "lcd_handwriting_required",
    "lcd_ops_required",
    "lcd_camera_required",
    "environment",
    "installation",
)


def _lcd_fingerprint(profile: Any) -> Dict[str, Any]:
    """LCD 选型真正吃的那些字段（客户改了其中任何一个 → 需要重新选型）。"""
    return {name: getattr(profile, name, None) for name in _LCD_LOCK_FIELDS}


def _locked_lcd_candidate(
    state: SolutionState, profile: Any, raw_products: List[Any]
):
    """上一轮已经锁定、并且需求指纹没变的屏幕 → 直接复用（不重新选型）。

    客户口径（2026-09-30）："根据对应品类推进需求询问，然后锁定屏幕" —— 屏幕一旦
    锁定，后续轮次（补充说明、问价、确认）都应该还是同一个型号，不能又换一个。
    """
    try:
        from ....memory.store import memory
    except Exception:  # pragma: no cover - 防御式
        return None
    session_id = str(state.get("session_id") or "")
    if not session_id:
        return None
    locked = memory.get_lcd_lock(session_id)
    if not locked:
        return None
    if locked.get("fingerprint") != _lcd_fingerprint(profile):
        return None
    model = str(locked.get("model") or "").strip()
    if not model:
        return None
    for item in raw_products or []:
        metadata = dict(
            getattr(item, "metadata", None)
            or (item.get("metadata") if isinstance(item, dict) else {})
            or {}
        )
        name = str(metadata.get("model") or metadata.get("product_id") or "").strip()
        if name != model:
            continue
        if str(metadata.get("display_type") or "").upper() not in ("LCD", "IFP"):
            continue
        logger.info("LCD recommend: reusing locked screen %s", model)
        return item, metadata, list(locked.get("reasons") or [])
    return None


def _store_lcd_lock(state: SolutionState, profile: Any, model: str, reasons: List[str]) -> None:
    """锁定屏幕：型号 + 事实指纹一起存，指纹变了才允许换型号。"""
    try:
        from ....memory.store import memory
    except Exception:  # pragma: no cover - 防御式
        return
    session_id = str(state.get("session_id") or "")
    if not session_id or not model:
        return
    memory.set_lcd_lock(
        session_id,
        {
            "model": model,
            "fingerprint": _lcd_fingerprint(profile),
            "reasons": list(reasons or []),
            "category": str(getattr(profile, "lcd_category", "") or ""),
        },
    )


def _recommend_lcd(
    state: SolutionState,
    profile: Any,
    raw_products: List[Any],
    *,
    follow_up: bool = False,
    previous_models: Optional[List[str]] = None,
) -> SolutionState:
    """LCD / IFP 的推荐：从检索到的候选里挑最合适的一个，用事实把结论说出来。

    与 LED 的区别：不碰点间距 / 箱体，不做观看距离推导；只按
    "拼接需求 → 尺寸 → 分辨率 → 拼缝" 从产品数据里选（计划 §十二）。

    ``follow_up``：客户在问"还有其他推荐吗" —— 这时**必须换一个没给过的型号**，
    不能复用锁定款、也不能再讲一遍首选（客户口径 2026-10 实测：
    客户连问三次"还有其他推荐的吗"，AI 三次都推同一款 Omni T65-K4/K4C）。
    """
    from ....dialogue.lcd_decision import is_ifp_requirement
    from ....rag.lcd_recommendation import layout_text, select_lcd_candidate

    already_seen = [str(name) for name in (previous_models or []) if str(name).strip()]

    # ① 已经锁定过、且需求没变 → 沿用同一个型号（锁定屏幕）
    #    但客户明确要"别的推荐"时**不能**沿用锁定款，否则就是重复推荐。
    picked = None if follow_up else _locked_lcd_candidate(state, profile, raw_products)
    # ② 否则重新选型；follow_up 时排除已经给客户看过的型号
    if picked is None:
        picked = select_lcd_candidate(
            profile,
            list(raw_products or []),
            exclude=already_seen if follow_up else (),
        )
    if picked is None:
        # 没有可用候选 → 按客户口径不说"找不到"，邀请放宽一个条件
        # 注意：一定是 **LCD 口径**的放宽条件（尺寸 / 拼缝 / 分辨率），
        # 不能出现 LED 的点间距 / 观看距离（客户口径 2026-09-30）。
        from ....rag.reply_composer import relaxation_answer

        logger.warning("LCD recommend: no usable candidate among %d docs", len(raw_products or []))
        return {
            "recommendation": relaxation_answer(
                product_family="ifp" if is_ifp_requirement(profile) else "lcd"
            ),
            "products": [],
            "next_action": "reflect",
        }

    product, metadata, reasons = picked
    model = str(metadata.get("model") or metadata.get("product_id") or "").strip()
    # 锁定屏幕（客户口径 2026-09-30）：型号 + 事实指纹入库，后续轮次复用同一个型号
    _store_lcd_lock(state, profile, model, reasons)
    is_ifp = is_ifp_requirement(profile)
    layout = layout_text(profile)
    if is_ifp:
        # 硬护栏（客户口径 2026-10）：IFP 是**单块会议平板**，不能拼接 ——
        # 任何情况下都不允许出现拼接排布 / 箱体数（"你的 3x3 要用 9 台会议平板"）。
        layout = ""
    facts: List[str] = []
    # 客户口径（2026-10）：只有**拼接墙**才用 "panels"（多块面板）的说法；
    # 单屏需求说 "Panels" 会让客户以为要拼（实测："65\" panels …" 后面还跟了拼缝）。
    splicing = _splicing_requirement(profile) and not is_ifp
    size = str(metadata.get("display_size_inch") or "").strip()
    if size:
        facts.append(f"{size} panels" if splicing else f"{size}-inch screen")
    resolution = str(metadata.get("resolution") or "").strip()
    if resolution:
        facts.append(f"{resolution} resolution")
    bezel_value = _bezel_mm(metadata)
    bezel = str(metadata.get("bazel_mm") or "").strip()
    if bezel:
        facts.append(f"{bezel} bezel")
    brightness = metadata.get("brightness_nit")
    if brightness:
        facts.append(f"{int(brightness)}nit brightness")

    # 拼缝事实（客户口径 2026-10）：**确定性地**告诉模型"达没达到客户要求"，
    # 以及"这条缝是可见的"。实测两个硬伤：
    #   · 客户要 0.88mm，选了 3.5mm 的 H6530LN-B，话术却说 "meeting your seam requirement"；
    #   · 把 LCD 拼接说成 "seamless / no visible seams"（产品数据明写 seam is visible）。
    seam_facts = _seam_facts(profile, metadata, bezel_value)

    # 语境：最近一段对话。客户口径（2026-10）：话术要结合语境、每次别用同一套模板；
    # 之前 LCD 推荐提示词**完全没有对话上下文**，模型只能套固定句式。
    conversation = ""
    try:
        from ....memory.history_window import dialogue_window_text

        session_id = str(state.get("session_id") or "")
        if session_id:
            conversation = dialogue_window_text(session_id) or ""
    except Exception:  # pragma: no cover - 防御式
        conversation = ""
    if not conversation:
        conversation = user_messages_text(state.get("messages", []))

    text = _express_lcd_recommendation(
        model=model,
        facts=facts,
        reasons=reasons,
        layout=layout,
        profile=profile,
        state=state,
        follow_up=follow_up,
        seam_facts=seam_facts,
        # 只有"客户要拼接"且面板缝确实可见时，才允许跟客户谈拼缝
        seam_is_visible=_seam_is_visible(metadata) and splicing,
        seam_met=_seam_requirement_met(profile, bezel_value),
        product_family_override=_lcd_product_family(profile, metadata),
        bezel_mm=bezel_value,
        conversation=conversation,
        splicing_requirement=splicing,
    )
    logger.info("LCD recommend: model=%s layout=%s", model, layout or "-")
    return {
        **state,
        "products": [product],
        "recommendation": text,
        "recommendation_result": {
            "status": "RECOMMENDED",
            "selected_model": model,
            "lcd_category": getattr(profile, "lcd_category", ""),
        },
        "next_action": "end",
    }


_BEZEL_NUMBER_RE = re.compile(r"(\d+(?:[.,]\d+)?)")
# LCD 拼接**一定有可见拼缝** —— 产品数据也是这么写的。
# 实测（客户 2026-10）：话术把 3x3 拼接墙说成 "seamless / no visible seams"，
# 和产品资料直接矛盾。这里是确定性的措辞护栏（不是关键词路由）。
_SEAMLESS_CLAIM_RE = re.compile(
    r"\bseamless\b|\bno visible seams?\b|\bwithout (?:any )?seams?\b|"
    r"\bframeless\b|\bbezel[-\s]?less\b|\binvisible bezels?\b|"
    r"无缝|无边框|没有拼缝|无拼缝|拼缝不可见",
    re.IGNORECASE,
)

# 声称"满足拼缝要求"的说法（只在**真的满足**时才允许出现）
_SEAM_MET_CLAIM_RE = re.compile(
    r"\bmeet(?:s|ing)? (?:your |the )?(?:seam|bezel)[^.!?]*|"
    r"\b(?:satisf\w+|match\w+) (?:your |the )?(?:seam|bezel)[^.!?]*|"
    r"\bwithin your[^.!?]*seam[^.!?]*|"
    r"满足[^。！？]*(?:拼缝|缝隙)|达到[^。！？]*(?:拼缝|缝隙)",
    re.IGNORECASE,
)


def _bezel_mm(metadata: Dict[str, Any]) -> Optional[float]:
    """产品数据里的拼缝/边框宽度（mm）。认 bazel_mm（数据里的旧拼写）与 bezel_mm。"""
    raw = metadata.get("bazel_mm") or metadata.get("bezel_mm")
    if raw in (None, "", [], {}):
        return None
    match = _BEZEL_NUMBER_RE.search(str(raw))
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", "."))
    except ValueError:  # pragma: no cover - 防御式
        return None


def _seam_is_visible(metadata: Dict[str, Any]) -> bool:
    """这条缝对客户是不是可见的（LCD 拼接一律可见）。"""
    note = str(metadata.get("splicing_note") or "")
    if note and re.search(r"not seamless|seam is visible|visible", note, re.IGNORECASE):
        return True
    # LCD 产品默认按"缝可见"处理；IFP 不拼接。
    return str(metadata.get("display_type") or "").upper() in ("LCD", "IFP")


def _splicing_requirement(profile: Any) -> bool:
    """这一轮客户要的是不是**拼接墙**（客户口径 2026-10：只有拼接才谈拼缝 / 排布）。"""
    return getattr(profile, "lcd_is_splicing", None) is True


def _seam_facts(profile: Any, metadata: Dict[str, Any], bezel_mm: Optional[float]) -> str:
    """把"实际拼缝 vs 客户要求"写成**确定性**事实交给模型（不许它自己推断）。

    客户口径（2026-10）：
      · 客户要 0.88mm、选了 3.5mm 的型号，话术却说 "meeting your seam requirement"
        —— 编造事实，这里直接写清达没达到；
      · **客户没说拼接就不要提拼缝**。单屏需求下把"tiled panel / 拼缝可见"塞给模型，
        它就会跟客户聊一块他根本没打算拼的屏（实测原话：
        "…350nit brightness. The seam between panels stays visible."）。
    """
    parts: List[str] = []
    splicing = _splicing_requirement(profile)
    if bezel_mm is not None:
        parts.append(f"selected panel bezel = {bezel_mm:g}mm")
    want = getattr(profile, "lcd_bezel_mm", None)
    if want not in (None, "", [], {}):
        try:
            want_value = float(want)
        except (TypeError, ValueError):
            want_value = None
        if want_value is not None:
            parts.append(f"customer asked for <= {want_value:g}mm")
            if bezel_mm is None:
                parts.append("we cannot confirm the selected panel meets that")
            elif bezel_mm <= want_value + 0.05:
                parts.append("it DOES meet the customer's seam requirement")
            else:
                parts.append(
                    "it does NOT meet the customer's seam requirement — do not claim it does"
                )
    # 只有"客户要拼接"时，拼缝可见性才是这一轮该讲的事实
    if splicing:
        if _seam_is_visible(metadata):
            parts.append("this is a tiled LCD panel: the seam between panels is visible")
        note = str(metadata.get("splicing_note") or "").strip()
        if note:
            parts.append(f"product data says: {note}")
    else:
        parts.append(
            "the customer did NOT ask for a tiled wall — do not mention seams, tiling "
            "or multi-panel layouts"
        )
    return "; ".join(parts) or "not specified in the product data"


def _lcd_product_family(profile: Any, metadata: Dict[str, Any]) -> str:
    """型号属于哪一类 —— 话术必须说准（客户口径 2026-10）。

    实测：客户要拼接墙，备选推了 P65（P 系列**单屏商用显示器**，产品数据里
    ``is_splicing=False``），话术却把它说成 "video wall … seamless visual surface"。
    P 系列虽然 ``splicing_supported=True``（能拼），但拼缝/边框是**明显可见**的，
    不能叫视频墙面板。
    """
    try:
        from ....dialogue.lcd_decision import is_ifp_requirement

        if is_ifp_requirement(profile):
            return "interactive flat panel (IFP, touch + whiteboard)"
    except Exception:  # pragma: no cover - 防御式
        pass
    if bool(metadata.get("is_splicing")):
        return "LCD video wall panel (designed for splicing into a wall)"
    if metadata.get("splicing_supported"):
        return (
            "commercial LCD monitor (a single display; it can be tiled, but the frames "
            "and seams between panels stay clearly visible — it is not a video wall panel)"
        )
    return "commercial LCD monitor (a single display, not for tiling)"


def _requested_bezel(profile: Any) -> Optional[float]:
    """客户要求的拼缝上限（mm）；没提返回 None。"""
    want = getattr(profile, "lcd_bezel_mm", None)
    if want in (None, "", [], {}):
        return None
    try:
        return float(want)
    except (TypeError, ValueError):
        return None


def _seam_truth_sentence(
    *,
    seam_is_visible: bool,
    seam_met: bool,
    bezel_mm: Optional[float],
    want_bezel_mm: Optional[float],
    splicing_requirement: bool = True,
) -> str:
    """确定性模板里那句"拼缝实话"（不含任何编造/夸大）。

    客户口径（2026-10）：客户要 0.88mm、面板是 3.5mm 时，话术必须说实话，
    不能声称"满足要求"；LCD 拼接也不能说"无缝"。
    另外：**客户没要拼接就一个字都别提拼缝**（实测原话：
    "…350nit brightness. The seam between panels stays visible."——客户只买一块单屏）。
    """
    if not splicing_requirement:
        return ""
    parts: List[str] = []
    if not seam_met and bezel_mm is not None and want_bezel_mm is not None:
        parts.append(
            f"Its bezel is {bezel_mm:g}mm, so it does not reach the {want_bezel_mm:g}mm "
            "seam you asked for — a tighter seam needs a different panel."
        )
    if seam_is_visible:
        parts.append("The seam between panels stays visible.")
    return (" " + " ".join(parts)) if parts else ""


def _seam_requirement_met(profile: Any, bezel_mm: Optional[float]) -> bool:
    """选中的面板拼缝**是否真的达到**客户要求（确定性判断，客户口径 2026-10）。

    客户没提拼缝 → True（没有可比的要求）。客户提了、但我们拿不到面板拼缝
    → False（说不清就不能声称满足）。
    """
    want = getattr(profile, "lcd_bezel_mm", None)
    if want in (None, "", [], {}):
        return True
    try:
        want_value = float(want)
    except (TypeError, ValueError):
        return True
    if bezel_mm is None:
        return False
    return bezel_mm <= want_value + 0.05


def _guard_lcd_seam_claims(
    text: str,
    *,
    seam_is_visible: bool,
    seam_met: bool,
    bezel_mm: Optional[float] = None,
    want_bezel_mm: Optional[float] = None,
) -> str:
    """确定性护栏：把与产品数据矛盾的**说法**改成实话（不是删句子）。

    客户口径（2026-10）：可以让模型润色，但绝不允许编造事实。所以这里采取
    "就地改写"而不是"整句删除"——删句子会把型号和参数一起删掉，回复就空了。

      · LCD 拼接一定有可见拼缝 → "seamless / frameless / 无缝" 一律改写成"拼缝可见"；
      · 面板拼缝**没达到**客户要求 → 把"满足您的拼缝要求"改写成实际拼缝数字 +
        说明更窄的做不到。
    """
    cleaned = str(text or "")
    if not cleaned:
        return cleaned
    # 只做"从句级"改写（语法安全）。"无缝"这类词一律留给调用方退模板，
    # 因为词级替换会写出 "gives you a with a visible seam visual surface" 这种病句。
    if not seam_met and _SEAM_MET_CLAIM_RE.search(cleaned):
        want = f"{want_bezel_mm:g}mm" if want_bezel_mm is not None else "the requested"
        # 句子通常已经写了实际拼缝，这里只补"达不到"的实话，避免重复念数字
        cleaned = _SEAM_MET_CLAIM_RE.sub(
            f"but it does not reach the {want} seam you asked for",
            cleaned,
        )
    return cleaned.strip()


LCD_RECOMMEND_PROMPT = """You are a sales engineer for commercial LCD displays. Write the
recommendation reply for the customer.

Product family of the selected model: {product_family}
Selected model (already decided by the system — do not change it): {model}
Verified product facts: {facts}
Video wall layout (already decided, do not recompute): {layout}
Why it fits: {reasons}
What the customer told us they will use it for: {use_case}
Seam / bezel facts (authoritative — do not reinterpret): {seam_facts}
Recent conversation (for context and tone only — never copy sentences from it):
{conversation}
{follow_up_line}
Rules:
1. Mention the selected model with its full model code.
2. Use only the verified facts above — never invent specs, prices or lead times.
3. If a layout is given, state the panel count for that layout.
3b. Call the product by the correct family above: an interactive flat panel must never be
   described as a plain LCD monitor, and a commercial monitor must not be called interactive.
3c. Describe the product in terms of what the CUSTOMER said they will use it for
   ("{use_case}"). Use the customer's own kind of place/scene, not our internal wording.
   Never call it "advertising", "monitoring", "digital signage" or any other category name
   unless the customer used that word themselves — an exhibition booth is an exhibition
   booth, not an advertising wall. If no use case is given above, do not name one.
3d. Read the seam/bezel facts literally. If they say the seam DOES NOT meet what the customer
   asked for, you must NOT claim it meets/satisfies/matches their seam requirement — state the
   actual bezel figure and that it is the closest available, and that going tighter needs a
   different panel. Never promise a seam figure the facts do not confirm.
3e. These are LCD panels: a tiled wall ALWAYS shows the seam between panels. Never call it
   "seamless", "frameless", "no visible seams", "bezel-less" or "invisible bezel".
4. Do not ask any requirement questions (the requirement is complete).
5. Plain text, no markdown, no bullet lists, no more than 4 sentences.
6. Do not say "based on my records" or reveal internal data sources.
7. Write it in your own words, for this conversation: react to what the customer actually
   said (see the recent conversation) instead of reciting a fixed template. Do not reuse the
   opening, the sentence pattern or the closing you used in your earlier replies. Polishing
   the wording is welcome; changing any fact is not.

Reply:"""


def _customer_use_case(profile: Any) -> str:
    """客户自己说的使用场景（**不是**内部品类 token）。

    客户口径（2026-10）：LCD 推荐话术以前把 ``profile.lcd_category``
    （monitoring / advertising / normal / conference_education）当"客户品类"
    写进提示词，模型就照抄成 "For your advertising video wall…"。
    客户全程说的是展会（exhibition），却被告知是"广告视频墙"。

    档案里 ``purpose`` 就是"客户说的使用场景"（见 models/requirement.py），
    这里把它转成客户口径的说法（唯一词表在 reply_composer._PURPOSE_LABELS），
    **绝不**回退到内部品类名 —— 内部品类只用来决定走哪条需求链，
    不该出现在客户文案里。
    """
    purpose = getattr(profile, "purpose", None)
    if purpose in (None, "", [], {}):
        return ""
    try:
        from ....rag.reply_composer import purpose_phrase

        return str(purpose_phrase(purpose, "en") or "").strip()
    except Exception:  # pragma: no cover - 防御式
        return str(purpose).strip()


def _express_lcd_recommendation(
    *,
    model: str,
    facts: List[str],
    reasons: List[str],
    layout: str,
    profile: Any,
    state: SolutionState,
    follow_up: bool = False,
    seam_facts: str = "not specified in the product data",
    seam_is_visible: bool = True,
    seam_met: bool = True,
    product_family_override: str = "",
    bezel_mm: Optional[float] = None,
    conversation: str = "",
    splicing_requirement: bool = True,
) -> str:
    """把 LCD 结论表达成客户话术（LLM 一次；失败退化为模板）。"""
    fallback_facts = ", ".join(facts) if facts else "commercial-grade panels"
    layout_line = f" For your {layout}, that is the panel count you need." if layout else ""
    # 选中的是交互平板还是普通商用显示器 —— 话术必须说对（客户口径 2026-09-30：
     # 要手写白板的会议室，不能推了普通 LCD 还叫它 LCD）
    try:
        from ....dialogue.lcd_decision import is_ifp_requirement

        product_family = (
            "interactive flat panel (IFP, touch + whiteboard)"
            if is_ifp_requirement(profile)
            else "commercial LCD display"
        )
    except Exception:  # pragma: no cover - 防御式
        product_family = "commercial LCD display"
    if product_family_override:
        product_family = product_family_override
    fallback = (
        f"{model} is the closest fit for your requirement: {fallback_facts}."
        + layout_line
        + _seam_truth_sentence(
            seam_is_visible=seam_is_visible,
            seam_met=seam_met,
            bezel_mm=bezel_mm,
            want_bezel_mm=_requested_bezel(profile),
            splicing_requirement=splicing_requirement,
        )
        + " Shall I prepare the quotation?"
    )
    use_case = _customer_use_case(profile)
    # 客户在问"还有其他推荐吗" → 这是**备选款**，不要说成"我重新给您选了一款"，
    # 也不要再把上一款拿出来对比（客户口径 2026-10）。
    follow_up_line = (
        "The customer has already seen our first suggestion and is asking for another "
        "option — present this one as an alternative, and do not repeat the previous model.\n"
        if follow_up
        else ""
    )
    try:
        prompt = LCD_RECOMMEND_PROMPT.format(
            product_family=product_family,
            model=model,
            facts=fallback_facts,
            layout=layout or "not applicable (single displays)",
            reasons="; ".join(reasons) or "matches the confirmed requirement",
            # 给客户**自己的**场景，不给内部品类 token（见 _customer_use_case）
            use_case=use_case or "not stated by the customer",
            follow_up_line=follow_up_line,
            # 拼缝事实由 Python 确定性给出，模型只能照实说
            seam_facts=seam_facts,
            # 语境：最近对话（客户口径 2026-10：话术要结合语境，不要每次同一套模板）
            conversation=conversation or "(no earlier turns available)",
        )
        response = get_llm(temperature=0.3).invoke(prompt)
        text = (response.content if hasattr(response, "content") else str(response)).strip()
        text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "").strip()
        text = text.replace("**", "").replace("__", "")
        # 事实护栏：把"满足拼缝要求"这种与产品数据矛盾的**从句**就地改成实话。
        # "无缝"这类词不做词级替换（会破坏语法），检测到就直接退确定性模板。
        want_bezel_value = _requested_bezel(profile)
        text = _guard_lcd_seam_claims(
            text,
            seam_is_visible=seam_is_visible,
            seam_met=seam_met,
            bezel_mm=bezel_mm,
            want_bezel_mm=want_bezel_value,
        )
        # 护栏之后仍然带着与事实矛盾的说法（无缝 / 谎称满足）→ 用确定性模板：
        # 模板只陈述产品数据里的事实，不含任何"无缝/满足"式断言。
        if (
            text
            and model in text
            and (not seam_is_visible or not _SEAMLESS_CLAIM_RE.search(text))
            and (seam_met or not _SEAM_MET_CLAIM_RE.search(text))
        ):
            return text
        if text:
            logger.info(
                "LCD recommendation wording rejected by the seam fact guard → 用确定性模板"
            )
    except Exception as error:  # pragma: no cover - 网络/额度问题
        logger.warning("LCD recommendation expression failed: %s", error)
    return fallback


def recommend_node(state: SolutionState) -> SolutionState:
    """Phase 10：确定性选型 → RAG 证据 → 工程计算 → 一次 LLM 表达。

    改造前：把检索到的每个候选都交给 LLM 各写一段推荐（N 次 LLM 调用），
    由 LLM 决定推荐哪个产品 —— 结果不稳定、延迟高、可能编造参数。

    改造后：``RecommendationEngine`` 用真实产品数据做确定性选型，
    工程参数由 ``screen_calculator`` 计算，LLM 只把结论表达成销售话术
    （全程 1 次 LLM 调用，失败时退化为模板）。

    LCD / IFP 走 `_recommend_lcd`（LED 引擎按点间距/箱体选型，对 LCD 不适用）。
    """
    requirement = state.get("requirement", {}) or {}
    raw_products = state.get("products", []) or []
    customer_text = (
        user_messages_text(state.get("messages", [])) or str(state.get("current_message", ""))
    )

    # ── 1. 需求档案（Phase 6）───────────────────────────────────────────
    from ....models.requirement import RequirementProfile

    profile = state.get("requirement_profile")
    if not isinstance(profile, RequirementProfile):
        profile = RequirementProfile.from_legacy(requirement)
        slots = state.get("understood_slots") or {}
        if slots:
            profile = profile.merge(RequirementProfile.from_slots(slots))

    # ── 客户这轮是不是在问"还有别的推荐吗"（换一个型号）────────────────────
    # 只有"客户之前已经拿到过推荐"时，'另外推荐一款'才等于'换一个型号'；
    # 用**本轮消息**判断（不能用整段历史，否则后面每一轮都会一直换型号）。
    # 注意：必须放在 LCD 分支**之前** —— 以前这段在 LCD 分流之后，LCD 永远走不到，
    # 于是"还有其他推荐吗"每次都把同一个型号再讲一遍（客户口径 2026-10 实测）。
    current_message = str(state.get("current_message") or "")
    follow_up = bool(_ALTERNATIVES_RE.search(current_message)) and bool(
        state.get("already_recommended")
    )
    previous_models = list(state.get("previous_recommended_models") or [])

    # ── LCD / IFP：走自己的选型路径（《LCD_IFP 整改计划》Phase 6/§十二）──────
    # LED 的 RecommendationEngine 按点间距/箱体选型，对 LCD 完全不适用
    # （实测 2026-09-30：LCD 需求齐全时这里仍然回 "Is it a permanent install, or
    # rental/events?" —— 那是 LED 的 Gate 在问，客户永远等不到推荐）。
    if str(getattr(profile, "display_type", "") or "").upper() in ("LCD", "IFP"):
        return _recommend_lcd(
            state, profile, raw_products,
            follow_up=follow_up,
            previous_models=previous_models,
        )

    # ── 2. 统一推荐入口：RecommendationService（Gate → 确定性选型）────────
    selection = _get_service().recommend(profile=profile, top_k=3)
    recommendations = selection.get("recommendations") or []
    logger.info(
        "Recommend(engine): candidates=%d top=%s",
        selection.get("candidate_count", 0),
        [r.get("model") for r in recommendations],
    )

    if selection.get("recommendation_status") == "NEED_CLARIFICATION":
        # 第二道保险命中：即使上游漏判，这里也绝不进入推荐，改为追问
        question = selection.get("next_question") or "Could you provide a bit more detail about your requirements?"
        logger.warning(
            "Recommend(service): NEED_CLARIFICATION missing=%s",
            selection.get("missing_fields"),
        )
        return {
            "recommendation": question,
            "products": [],
            "recommendation_result": selection,
            "next_action": "clarify",
        }

    if not recommendations:
        constraints = selection.get("hard_constraints") or {}
        hard_fields = [key for key, value in constraints.items() if key != "sources" and value]
        has_hard = bool(hard_fields)
        logger.warning("Recommend: no matching model (hard_constraints=%s)", constraints)
        if has_hard:
            # 【客户口径】不说"目录里没有匹配的产品"，而是邀请客户放宽某个条件，
            # 并点出最可能卡住的那几项（环境/安装方式/亮度/点间距）。
            from ....rag.reply_composer import relaxation_answer
            from ....utils.product_family import product_family_of

            # 口径跟着链路走：LCD / IFP 会话不能冒出"点间距 / 观看距离"（LED 口径）。
            message = relaxation_answer(product_family=product_family_of(profile))
        else:
            message = (
                "Could you tell me the scenario and whether it is indoors or outdoors? "
                "That will let me match the right model for you."
            )
        return {
            "recommendation": message,
            "products": [],
            "recommendation_result": selection,
            "next_action": "reflect",
        }

    # ── 3. 工程计算（v2.0 Phase 8/9：Calculation Ready Gate）──────────────
    from ....rag.readiness import check_calculation_ready

    calc_decision = check_calculation_ready(profile)
    logger.info(
        "CalculationGate: ready=%s missing=%s reason=%s",
        calc_decision.ready, calc_decision.missing, calc_decision.reason,
    )

    calculation = None
    calculation_variants = None
    if calc_decision.ready:
        try:
            from ....tools.screen_calculator import (
                calculate_screen_variants,
                format_screen_spec,
            )

            # v2.1：客户授权 AI 决定尺寸时，用观看距离推导出的**参考尺寸**算，
            # 而不是把参考尺寸写进客户的确认事实（计划第 14 / 15 节）。
            derived = list(getattr(calc_decision, "derived_size_m", None) or [])
            if len(derived) == 2 and derived[0] and derived[1]:
                target_width_mm = float(derived[0]) * 1000
                target_height_mm = float(derived[1]) * 1000
                logger.info(
                    "[Delegated] 尺寸由观看距离推导：%sm x %sm（仅供参考，不写入客户事实）",
                    derived[0], derived[1],
                )
            else:
                target_width_mm = profile.target_width_mm
                target_height_mm = profile.target_height_mm

            # 客户口径（2026-09-21）：客户报 "x × y" 时不区分哪边是宽 ——
            # 可行性层已经挑出"哪个方向能拼到客户要的分辨率"，这里按它换边计算，
            # 保证回复里的箱体数 / 实际尺寸与可行性判断一致。
            if (
                str(selection.get("size_orientation") or "") == "swapped"
                and target_width_mm and target_height_mm
            ):
                target_width_mm, target_height_mm = target_height_mm, target_width_mm
                logger.info(
                    "[Orientation] 按客户报的尺寸换边计算：%.2fm 作宽、%.2fm 作高",
                    target_width_mm / 1000, target_height_mm / 1000,
                )

            # 客户口径：箱体可以横拼也可以竖拼 → 两种排布都给客户
            calculation_variants = calculate_screen_variants(
                recommendations[0]["model"],
                target_width_mm=target_width_mm,
                target_height_mm=target_height_mm,
            )
            calculation = calculation_variants["landscape"]
            calculation["summary"] = format_screen_spec(calculation)
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("Screen calculation failed: %s", exc)
    else:
        logger.info("CalculationGate 未就绪 → 只推荐产品，本轮不做箱体/模组计算")

    # ── 4. RAG 证据（v2.0 Phase 7：确定性证据排序 + 冲突检测）──────────────
    # 顺序跟随引擎选型结果；环境冲突的证据直接丢弃，绝不交给 LLM"解释"
    from ....rag.rerank import rank_evidence

    ranked_evidence, dropped_evidence = rank_evidence(
        raw_products, selection, profile, limit=3
    )
    if dropped_evidence:
        logger.info("Evidence dropped by rerank: %s", dropped_evidence)
    products = _evidence_products(recommendations, ranked_evidence or raw_products)

    # ── 5. 一次 LLM 表达 ────────────────────────────────────────────────
    degraded_slots = list(selection.get("unknown_requirements") or [])
    # 客户这轮问的是"还有没有别的推荐" → 用"备选 + 邀请补充需求"的格式，
    # 不重讲首选、不催尺寸、不提价格（客户口径）。
    # follow_up / current_message 已在函数开头算好（LCD 分支也要用）。
    # v2.1：尺寸已经延后/客户不说时，不要在本轮再问尺寸（问了就是重复）
    ask_size = (not calc_decision.ready) and bool(calc_decision.next_question)
    answer = _express_recommendation(
        recommendations=recommendations,
        profile=profile,
        calculation=calculation,
        additional_requirements=state.get("additional_requirements", []) or [],
        customer_text=customer_text,
        need_size_question=ask_size,
        language=state.get("understood_language") or "en",
        degraded_slots=degraded_slots,
        follow_up=follow_up,
        previous_models=state.get("previous_recommended_models") or (),
        calculation_variants=calculation_variants,
        multi_screen_brief=state.get("multi_screen_brief", "") or "",
    )

    # 缺尺寸时必须追问（确定性兜底：模型若没问，就补一句尺寸追问，
    # 保证"没尺寸一定问、有尺寸才算"，不依赖 LLM 是否听话）
    if not follow_up and not calc_decision.ready and calc_decision.next_question:
        answer = _ensure_size_question(answer, calc_decision.next_question)

    # 客户口径：推荐话术里不再追加"某项还没确认 / 可能有偏差"的说明，
    # 只给结论（缺的信息在系统内部照旧记录，用于打分与排查）。

    return {
        "products": products,
        "recommendation": answer,
        "recommendation_result": selection,
        "screen_calculation": calculation,
        "screen_calculation_variants": calculation_variants,
        "calculation_gate": calc_decision.to_dict(),
        "evidence_dropped": dropped_evidence,
        "next_action": "reflect",
    }
