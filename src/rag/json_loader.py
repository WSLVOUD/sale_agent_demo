"""
结构化产品数据加载器。

从 JSON 文件加载已结构化的产品数据，支持：
1. Schema 验证（Pydantic）
2. 结构化过滤（亮度、点间距、租赁等）
3. 为向量库生成带 metadata 的 Document
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from pydantic import ValidationError

from src.models.product import (
    Product,
    LEDProduct,
    LCDProduct,
    IFPProduct,
    LEDSubModel,
    IFPSubModel,
)

logger = logging.getLogger(__name__)

PRODUCT_JSON_FILES = ["led_products.json", "lcd_products.json", "ifp_products.json"]


def _parse_led(data: dict) -> LEDProduct:
    """解析 LED 产品数据。"""
    sub_models = []
    for sub in data.get("sub_models", []):
        sub_models.append(LEDSubModel(**sub))
    data = {**data, "sub_models": sub_models}
    return LEDProduct(**data)


def _parse_lcd(data: dict) -> LCDProduct:
    """解析 LCD 产品数据。"""
    return LCDProduct(**data)


def _parse_ifp(data: dict) -> IFPProduct:
    """解析 IFP 产品数据。"""
    sub_models = []
    for sub in data.get("sub_models", []):
        sub_models.append(IFPSubModel(**sub))
    data = {**data, "sub_models": sub_models}
    return IFPProduct(**data)


def load_structured_products(data_dir: str) -> list[Product]:
    """从 JSON 文件加载所有结构化产品。"""
    root = Path(data_dir)
    products: list[Product] = []
    errors: list[str] = []

    for filename in PRODUCT_JSON_FILES:
        path = root / filename
        if not path.exists():
            logger.warning("Product JSON file not found: %s", path)
            continue

        with open(path, "r", encoding="utf-8") as f:
            container = json.load(f)

        display_type = container.get("display_type", "LED")
        for item in container.get("products", []):
            try:
                if display_type == "LED":
                    item["led_data"] = _parse_led({k: v for k, v in item.items() if k not in ("display_type",)})
                elif display_type == "LCD":
                    item["lcd_data"] = _parse_lcd({k: v for k, v in item.items() if k not in ("display_type",)})
                elif display_type == "IFP":
                    item["ifp_data"] = _parse_ifp({k: v for k, v in item.items() if k not in ("display_type",)})

                product = Product.model_validate(item)
                products.append(product)
            except ValidationError as exc:
                errors.append(f"{filename}/{item.get('product_id', '?')}: {exc}")

    if errors:
        logger.error("Schema validation errors:\n%s", "\n".join(errors))

    logger.info("Loaded %d structured products (%d errors)", len(products), len(errors))
    return products


def product_to_documents(product: Product) -> list[Document]:
    """将单个产品转换为向量库 Document（保留全 metadata）。"""
    desc = product.get_description_text()
    metadata = _build_metadata(product)
    metadata["chunk_id"] = str(uuid.uuid4())
    return [Document(page_content=desc, metadata=metadata)]


def _build_metadata(product: Product) -> dict[str, Any]:
    """从 Product 构建向量库 metadata。"""
    m: dict[str, Any] = {
        "product_id": product.product_id,
        "series": product.series,
        "display_type": product.display_type,
        "environment": product.environment,
        "indoor": "indoor" in product.environment,
        "outdoor": "outdoor" in product.environment,
        "is_rental": product.is_rental,
        "brightness_nit": product.brightness_nit,
        "features": product.features,
        "source": "structured_json",
    }

    if product.led_data:
        led = product.led_data
        m.update({
            "brightness_min_cd": led.brightness_nit,
            "brightness_max_cd": led.brightness_nit,
            "warranty_years": led.warranty_years,
            "price_tier": led.price_tier,
            "cob": led.cob,
            "hdr": led.hdr,
            "waterproof": led.waterproof,
            "lamp_brand": led.lamp_brand,
            "refresh_rate_hz": led.refresh_rate_hz,
        })
        pmin = min(s.pixel_pitch_mm for s in led.sub_models) if led.sub_models else None
        pmax = max(s.pixel_pitch_mm for s in led.sub_models) if led.sub_models else None
        if pmin is not None:
            m["pixel_pitch_min_mm"] = pmin
        if pmax is not None:
            m["pixel_pitch_max_mm"] = pmax

    elif product.lcd_data:
        lcd = product.lcd_data
        m.update({
            "brightness_min_cd": lcd.brightness_nit,
            "brightness_max_cd": lcd.brightness_nit,
            "display_size_inch": lcd.display_size_inch,
            "resolution": lcd.resolution,
            "bazel_mm": lcd.bazel_mm,
            "operation_hours": lcd.operation_hours,
            "is_splicing": lcd.is_splicing,
            "semi_outdoor": "semi_outdoor" in product.environment,
        })

    elif product.ifp_data:
        ifp = product.ifp_data
        m.update({
            "system": ifp.system,
            "touch_points": ifp.touch_points,
            "has_nfc": ifp.has_nfc,
            "has_ai_camera": ifp.has_ai_camera,
        })
        # 取最小/最大尺寸
        sizes = []
        for sub in ifp.sub_models:
            import re
            match = re.search(r'(\d+)', sub.size_inch)
            if match:
                sizes.append(int(match.group(1)))
        if sizes:
            m["size_min_inch"] = min(sizes)
            m["size_max_inch"] = max(sizes)

    return m


def load_structured_documents(data_dir: str) -> list[Document]:
    """加载所有结构化产品并转换为 Document 列表。"""
    products = load_structured_products(data_dir)
    docs = []
    for p in products:
        docs.extend(product_to_documents(p))
    logger.info("Converted %d products to %d documents", len(products), len(docs))
    return docs


# ── 结构化过滤 ─────────────────────────────────────────────────────────────
class ProductFilter:
    """
    基于结构化字段的产品过滤器。

    使用方法：
        filter = ProductFilter(products)
        candidates = filter.apply(
            brightness_min=4000,
            pixel_pitch_max=5.0,
            is_rental=True,
            display_type="LED",
        )
    """

    def __init__(self, products: list[Product]):
        self.products = products

    def apply(
        self,
        brightness_min: int | None = None,
        brightness_max: int | None = None,
        pixel_pitch_min: float | None = None,
        pixel_pitch_max: float | None = None,
        is_rental: bool | None = None,
        display_type: str | None = None,
        indoor: bool | None = None,
        outdoor: bool | None = None,
        waterproof: bool | None = None,
        cob: bool | None = None,
        hdr: bool | None = None,
        price_tier: str | None = None,
        exclude_ifp: bool = False,
        semi_outdoor: bool = False,
        is_splicing: bool | None = None,
        top_k: int = 20,
    ) -> list[Product]:
        """返回满足所有条件的 Product 列表（按相关性排序）。"""
        candidates: list[tuple[float, Product]] = []

        for p in self.products:
            # display_type
            if display_type and p.display_type != display_type:
                continue
            # exclude IFP products (e.g., user explicitly doesn't want interactive flat panel)
            if exclude_ifp and p.display_type == "IFP":
                continue

            # environment
            if indoor is True and "indoor" not in p.environment:
                continue
            if outdoor is True and "outdoor" not in p.environment:
                continue

            # brightness
            bn = p.brightness_nit or 0
            if brightness_min is not None and bn < brightness_min:
                continue
            if brightness_max is not None and bn > brightness_max:
                continue

            # pixel pitch (LED only)
            if p.led_data and p.led_data.sub_models:
                pmin = min(s.pixel_pitch_mm for s in p.led_data.sub_models)
                pmax = max(s.pixel_pitch_mm for s in p.led_data.sub_models)
                if pixel_pitch_min is not None and pmax < pixel_pitch_min:
                    continue
                if pixel_pitch_max is not None and pmin > pixel_pitch_max:
                    continue

            # is_rental
            if is_rental is not None and p.is_rental != is_rental:
                continue

            # waterproof / cob / hdr / semi_outdoor
            if p.led_data:
                if waterproof is True and not p.led_data.waterproof:
                    continue
                if cob is True and not p.led_data.cob:
                    continue
                if hdr is True and not p.led_data.hdr:
                    continue
                if price_tier and p.led_data.price_tier != price_tier:
                    continue

            # semi_outdoor
            if semi_outdoor and "semi_outdoor" not in p.environment:
                continue

            # is_splicing (LCD only)
            if is_splicing is not None:
                if p.lcd_data is None:
                    continue
                if p.lcd_data.is_splicing != is_splicing:
                    continue

            # score = 匹配字段数（越大越相关）
            score = 0.0
            if display_type:
                score += 1
            if indoor is not None or outdoor is not None:
                score += 1
            if brightness_min is not None:
                score += 1
            if pixel_pitch_min is not None or pixel_pitch_max is not None:
                score += 1
            if is_rental is not None:
                score += 1
            if semi_outdoor:
                score += 1
            if is_splicing is not None:
                score += 1

            candidates.append((score, p))

        # 按 score 降序，取 top_k
        candidates.sort(key=lambda x: -x[0])
        return [p for _, p in candidates[:top_k]]
