"""
Fast Path 处理器。

处理简单 Query，直接走结构化过滤 + 模板回复，不经过完整 Agent 流程。
适用于：参数查询、简单推荐、固定销售异议等。
"""
from __future__ import annotations

import logging
from typing import Any

from src.models.product import Product
from src.rag.json_loader import load_structured_products, ProductFilter

logger = logging.getLogger(__name__)

# 全局产品缓存（避免重复加载）
_product_cache: list[Product] | None = None


def _get_products(data_dir: str) -> list[Product]:
    global _product_cache
    if _product_cache is None:
        _product_cache = load_structured_products(data_dir)
    return _product_cache


def _find_products(
    constraints: dict,
    data_dir: str,
    top_k: int = 5,
) -> list[Product]:
    """根据约束条件查找匹配产品。"""
    from src.config import config as global_config

    products = _get_products(data_dir or global_config.DATA_DIR)
    filter_engine = ProductFilter(products)

    # 映射约束
    kwargs: dict = {}
    if constraints.get("pixel_pitch"):
        pitch = float(constraints["pixel_pitch"])
        tolerance = constraints.get("pixel_pitch_tolerance", 0.5)
        kwargs["pixel_pitch_min"] = pitch - tolerance
        kwargs["pixel_pitch_max"] = pitch + tolerance

    if constraints.get("brightness_min"):
        kwargs["brightness_min"] = int(constraints["brightness_min"])

    if constraints.get("outdoor") is True:
        kwargs["outdoor"] = True
    elif constraints.get("indoor") is True:
        kwargs["indoor"] = True

    if constraints.get("is_rental") is not None:
        kwargs["is_rental"] = constraints["is_rental"]

    if constraints.get("waterproof"):
        kwargs["waterproof"] = True

    if constraints.get("cob"):
        kwargs["cob"] = True

    if constraints.get("display_type"):
        kwargs["display_type"] = constraints["display_type"]

    if constraints.get("exclude_ifp"):
        kwargs["exclude_ifp"] = True

    if constraints.get("semi_outdoor"):
        kwargs["semi_outdoor"] = True

    if constraints.get("is_splicing"):
        kwargs["is_splicing"] = True

    if constraints.get("hdr"):
        kwargs["hdr"] = True

    return filter_engine.apply(**kwargs, top_k=top_k)


def _build_product_summary(products: list[Product]) -> str:
    """Format product list into natural conversational list."""
    if not products:
        return "No matching products found."

    names = [f"{p.series}" for p in products]
    return "Here are my recommendations: " + ", ".join(names)


# ── 销售模板 ────────────────────────────────────────────────────────────────

_SALES_TEMPLATES: dict[str, str] = {
    "greeting": "Hello! I'm your LED display advisor. How can I help you today?",
    "warranty": "We offer 1-2 years of warranty. Warranty duration varies by product series: {warranty_info}. Feel free to reach out if you have any questions.",
    "objection_price": "I understand your concern about price. Our products stand out for:\n1. Kinglight LEDs: top quality, long lifespan\n2. 2-year warranty: industry-leading\n3. Full technical support: installation, commissioning, and maintenance\n\nCompared to lower-priced options, we focus on long-term value and stability. What matters most to you?",
    "objection_compare": "Our products have these key differentiators:\n1. COB technology: waterproof and impact-resistant, great for demanding environments\n2. HDR support: more vivid picture quality\n3. Kinglight LEDs: accurate colors, longer lifespan\n\nWhich specs matter most to you? Let me help you compare.",
    "small_talk": "Thanks for reaching out! We carry the full range: LED, LCD, and IFP displays. Do you have a specific use case in mind? Let me help you find the right fit.",
}


def _select_template(template_type: str, **kwargs) -> str:
    """选择并填充销售模板。"""
    template = _SALES_TEMPLATES.get(template_type, _SALES_TEMPLATES["small_talk"])

    if template_type == "warranty":
        template = template.replace(
            "{warranty_info}",
            "TW11 series: 1 year, TW21/TW31 series: 2 years"
        )

    return template


# ── Fast Path 主入口 ────────────────────────────────────────────────────────

def fast_path_handle(
    query: str,
    constraints: dict | None,
    template_type: str | None,
    data_dir: str,
) -> dict[str, Any]:
    """
    Fast Path 主处理函数。

    Args:
        query: 原始用户 Query
        constraints: 从 Query 提取的结构化约束
        template_type: 销售模板类型（greeting / warranty / objection_price 等）
        data_dir: 产品数据目录

    Returns:
        dict，包含：
          - answer: 回复文本
          - products: 匹配产品列表
          - route: "fast"
          - template_used: 是否使用了模板
    """
    answer_parts = []
    products: list[Product] = []
    template_used = False

    # 1. 销售模板优先
    if template_type and template_type in _SALES_TEMPLATES:
        answer_parts.append(_select_template(template_type))
        template_used = True

    # 2. 如果有参数约束，查找匹配产品
    if constraints and any([
        constraints.get("pixel_pitch"),
        constraints.get("brightness_min"),
        constraints.get("outdoor"),
        constraints.get("indoor"),
        constraints.get("is_rental"),
        constraints.get("waterproof"),
        constraints.get("display_type"),
        constraints.get("exclude_ifp"),
        constraints.get("purpose"),
        constraints.get("pixel_pitch_min"),
        constraints.get("pixel_pitch_max"),
        constraints.get("semi_outdoor"),
        constraints.get("is_splicing"),
        constraints.get("hdr"),
    ]):
        try:
            matched = _find_products(constraints, data_dir)
            if matched:
                products = matched
                if not template_used:
                    answer_parts.append(f"Found {len(matched)} matching products:\n")
                    answer_parts.append(_build_product_summary(matched[:3]))
        except Exception as exc:
            logger.warning("Fast path product search failed: %s", exc)

    # 3. 如果只有模板没有产品，生成通用回复
    if not answer_parts:
        answer_parts.append(_SALES_TEMPLATES["small_talk"])

    return {
        "answer": "\n".join(answer_parts).strip(),
        "products": [
            {
                "product_id": p.product_id,
                "series": p.series,
                "display_type": p.display_type,
                "brightness_nit": p.brightness_nit,
                "features": p.features,
            }
            for p in products[:3]
        ],
        "route": "fast",
        "template_used": template_used,
        "complexity": "simple",
    }
