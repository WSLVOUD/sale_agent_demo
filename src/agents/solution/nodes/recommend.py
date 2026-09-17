"""Recommendation node - generates product recommendations."""
from typing import Dict, Any, List
import logging
import re

from ..state import SolutionState
from ....core.llm import get_llm
from ....rag.rerank import is_display_candidate, get_product_category, has_environment_conflict, is_ifp_product
from ....utils.ifp_intent import has_ifp_intent, user_messages_text

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

Output the recommendation directly (2-3 sentences max):"""


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
):
    """用一次 LLM 调用把确定性结论表达成销售话术；失败时退化为模板。"""
    # 客户口径（2026-09-18）：**推荐时只报一个型号**，其他型号一律不提。
    # 只有客户主动问"还有别的推荐吗"（follow_up）时，才换成另一个型号给他。
    top = recommendations[1] if (follow_up and len(recommendations) > 1) else recommendations[0]
    # 只把这个型号的实测参数交给 LLM —— 给它三行，它就容易把别的型号也念出来
    alternative_lines: list = []

    spec_lines = [
        f"- {top['model']}: pixel pitch {top['pixel_pitch_mm']}mm, "
        f"brightness {top['brightness_nit']}nit, "
        f"cabinet {top['cabinet_size_mm']}, "
        f"{top['modules_per_cabinet']} modules per cabinet, "
        f"{top['warranty_years']} year warranty"
    ]
    # 备选款：说清"它比首选款多什么"，让客户按自己的偏好选（不提价格）
    reasons = "; ".join(top.get("reasons") or [])
    extra = ""
    if additional_requirements:
        extra = f"\nCustomer's additional requirements: {', '.join(additional_requirements)}"
    latest = ""
    if customer_text:
        latest = f"\nCustomer's latest message: {str(customer_text)[:200]}"
    calc_text = ""
    if calculation:
        calc_text = (
            f"\nCalculated configuration (authoritative, do not recompute): "
            f"{calculation['columns']}x{calculation['rows']} = {calculation['cabinet_count']} cabinets, "
            f"actual size {calculation['actual_width_m']}m x {calculation['actual_height_m']}m "
            f"({calculation['area_sqm']} sqm), {calculation['total_modules']} modules."
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
        + f"{calc_text}{extra}{latest}\n\n"
        "Rules:\n"
        "1. Mention the selected model with its full model code — it does not have to be the very first words.\n"
        "2. Add 1-2 concrete selling points using ONLY the verified data above.\n"
        "2b. If the customer's latest message mentions other requirements (quality, lead time, installation, "
        "brightness, size, delivery…), acknowledge them in ONE short clause and tie the choice to them, "
        "so the reply answers what they just said instead of repeating the same text.\n"
        "3. If a calculated configuration is given, include the cabinet count and actual size.\n"
        "3b. Mention ONLY the selected model above. Never name, hint at or compare any other model "
        "code, and never write \"If you want <something>, <OTHER MODEL>\". One model per reply.\n"
        "3c. NEVER mention price, price tier, cost, discount, budget or value for money — pricing is handled "
        "separately by the sales team.\n"
        + next_step_rule
        + degraded_rule +
        f"5. Plain text only, no markdown, no bullets, max 90 words.\n"
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
        fallback += (
            f" For your target size we need {calculation['columns']}x{calculation['rows']} "
            f"= {calculation['cabinet_count']} cabinets "
            f"({calculation['total_modules']} modules), actual size "
            f"{calculation['actual_width_m']}m x {calculation['actual_height_m']}m."
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


def recommend_node(state: SolutionState) -> SolutionState:
    """Phase 10：确定性选型 → RAG 证据 → 工程计算 → 一次 LLM 表达。

    改造前：把检索到的每个候选都交给 LLM 各写一段推荐（N 次 LLM 调用），
    由 LLM 决定推荐哪个产品 —— 结果不稳定、延迟高、可能编造参数。

    改造后：``RecommendationEngine`` 用真实产品数据做确定性选型，
    工程参数由 ``screen_calculator`` 计算，LLM 只把结论表达成销售话术
    （全程 1 次 LLM 调用，失败时退化为模板）。
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

            message = relaxation_answer()
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
    if calc_decision.ready:
        try:
            from ....tools.screen_calculator import calculate_screen, format_screen_spec

            calculation = calculate_screen(
                recommendations[0]["model"],
                target_width_mm=profile.target_width_mm,
                target_height_mm=profile.target_height_mm,
            )
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
    follow_up = bool(_ALTERNATIVES_RE.search(str(customer_text or "")))
    answer = _express_recommendation(
        recommendations=recommendations,
        profile=profile,
        calculation=calculation,
        additional_requirements=state.get("additional_requirements", []) or [],
        customer_text=customer_text,
        need_size_question=not calc_decision.ready,
        language=state.get("understood_language") or "en",
        degraded_slots=degraded_slots,
        follow_up=follow_up,
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
        "calculation_gate": calc_decision.to_dict(),
        "evidence_dropped": dropped_evidence,
        "next_action": "reflect",
    }
