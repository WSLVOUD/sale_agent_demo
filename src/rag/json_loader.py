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
    CanonicalModel,
    LEDGeometry,
    parse_resolution,
    parse_size_mm,
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


# ── Phase 2：Model 级 RAG Document ──────────────────────────────────────────
ENVIRONMENT_METADATA_VERSION = 6  # 与 loader.py 保持一致（v6：新增室外租赁系列）


def _clean_metadata_value(value: Any) -> Any:
    """Chroma metadata 只接受 str / int / float / bool，列表统一转成字符串。"""
    if isinstance(value, (list, tuple, set)):
        return "|".join(str(v) for v in value)
    return value


def canonical_model_to_document(
    model: CanonicalModel,
    chunk_id: str | None = None,
) -> Document:
    """把一个 Model 级记录转成向量库 Document（一个实际销售型号 = 一个 Document）。"""
    metadata: dict[str, Any] = {
        # 身份
        "chunk_id": chunk_id or f"model-{model.model}",
        "level": "model",
        "model": model.model,
        "product_id": model.model,          # 兼容既有按 product_id 取用的代码
        "series_id": model.series_id,
        "series": model.series,
        "display_type": model.display_type,
        # 硬约束
        "environment": _clean_metadata_value(model.environment),
        "indoor": model.indoor,
        "outdoor": model.outdoor,
        "installation": model.installation,
        "is_rental": model.installation == "rental",
        # 光学 / 像素
        "pixel_pitch_min_mm": model.pixel_pitch_mm,
        "pixel_pitch_max_mm": model.pixel_pitch_mm,
        "pixel_pitch_mm": model.pixel_pitch_mm,
        "resolution_per_sqm": model.resolution_per_sqm,
        "module_resolution": model.module_resolution,
        "cabinet_resolution": model.cabinet_resolution or "",
        "scanning": model.scanning or "",
        "refresh_rate_hz": model.refresh_rate_hz,
        # 亮度
        "brightness_nit": model.brightness_nit,
        "brightness_min_cd": model.brightness_min_nit,
        "brightness_max_cd": model.brightness_max_nit,
        # 工程尺寸
        "module_size_mm": model.module_size_text,
        "cabinet_size_mm": model.cabinet_size_text,
        "module_width_mm": model.module_width_mm,
        "module_height_mm": model.module_height_mm,
        "cabinet_width_mm": model.cabinet_width_mm,
        "cabinet_height_mm": model.cabinet_height_mm,
        "modules_per_cabinet": model.modules_per_cabinet,
        # 商业属性
        "warranty_years": model.warranty_years,
        "lamp_brand": model.lamp_brand or "",
        "price_tier": model.price_tier,
        "cob": model.cob,
        "hdr": model.hdr,
        "waterproof": model.waterproof,
        "gob": model.gob,
        "flexible": model.flexible,
        "features": _clean_metadata_value(model.features),
        # 系统字段
        "environment_metadata_version": ENVIRONMENT_METADATA_VERSION,
        "product_category": "display",
        "source": "canonical_json",
    }
    return Document(page_content=model.to_text(), metadata=metadata)


def load_model_documents(data_dir: str) -> list[Document]:
    """Phase 2 语料：每个 Model 一个 Document，Series 作为 metadata。

    客户口径（2026-09-22）：LCD / IFP 也要进向量库 —— 以前这里只展开 LED，
    客户上传了 LCD/IFP 产品资料后它们在库里检索不到（只有 LED 56 条）。
    现在 LED 走 CanonicalModel，LCD/IFP 走下面同一套 Model 级 metadata。
    """
    models = load_canonical_models(data_dir)
    documents = [canonical_model_to_document(m) for m in models]
    led_count = len(documents)
    extra = non_led_model_documents(data_dir)
    if extra:
        documents.extend(extra)
        logger.info(
            "Built %d model-level documents（LED %d + LCD/IFP %d）",
            len(documents), led_count, len(extra),
        )
    else:
        logger.info("Built %d model-level documents", len(documents))
    return documents


# ── LCD / IFP 的 Model 级语料（2026-09-22）─────────────────────────────────
def _display_metadata(
    *,
    model: str,
    series_id: str,
    series: str,
    display_type: str,
    environment: str = "indoor",
    brightness_nit: Optional[int] = None,
    features: Optional[list[str]] = None,
) -> dict[str, Any]:
    """非 LED 产品共用的 metadata（键与 LED 的 Model 级语料保持一致）。

    注意：点间距 / 模组 / 箱体这些 LED 专有字段**不写**（不留编造的默认值）；
    ``modules_per_cabinet`` 是向量库校验要求的键，非 LED 用 0 表示"不适用"。
    """
    indoors = environment in ("indoor", "室内")
    meta: dict[str, Any] = {
        # 身份
        "chunk_id": f"model-{model}",
        "level": "model",
        "model": model,
        "product_id": model,
        "series_id": series_id,
        "series": series,
        "display_type": display_type,
        # 硬约束
        "environment": environment,
        "indoor": indoors,
        "outdoor": not indoors,
        "installation": "fixed",
        "is_rental": False,
        # 亮度
        "brightness_nit": brightness_nit or 0,
        "brightness_min_cd": brightness_nit or 0,
        "brightness_max_cd": brightness_nit or 0,
        # 非 LED：没有模组/箱体概念
        "modules_per_cabinet": 0,
        "gob": False,
        "flexible": False,
        "warranty_years": 1,
        "features": _clean_metadata_value(features or []),
        # 系统字段
        "environment_metadata_version": ENVIRONMENT_METADATA_VERSION,
        "product_category": "display",
        "source": "canonical_json",
    }
    return meta


def _lcd_text(product: "LCDProduct", model: str) -> str:
    """LCD 型号的检索文本（只写资料里有的事实）。"""
    parts = [
        f"{model} | {product.series} | LCD | {'indoor' if 'indoor' in product.environment else 'outdoor'}",
        f"size={product.display_size_inch}",
        f"resolution={product.resolution}",
        f"brightness={product.brightness_nit}nit",
        f"contrast={product.contrast_ratio}",
        f"operation={product.operation_hours}",
        f"lifespan={product.service_life_hours}h",
        "splicing-video-wall" if product.is_splicing else "single-display",
    ]
    if product.bazel_mm:
        parts.append(f"bezel={product.bazel_mm}")
    if product.power_consumption_w:
        parts.append(f"power={product.power_consumption_w}W")
    if product.features:
        parts.append("features: " + ", ".join(product.features))
    return " | ".join(part for part in parts if part)


def _lcd_model_document(product: "Product", lcd: "LCDProduct") -> Document:
    """一个 LCD 型号（尺寸档）= 一条 Model 级文档。"""
    model = product.product_id
    metadata = _display_metadata(
        model=model,
        series_id=product.product_id,
        series=product.series,
        display_type="LCD",
        environment="indoor" if "indoor" in product.environment else "outdoor",
        brightness_nit=lcd.brightness_nit,
        features=list(product.features or []) + list(lcd.features or []),
    )
    metadata.update({
        "display_size_inch": lcd.display_size_inch,
        "resolution": lcd.resolution,
        "bazel_mm": lcd.bazel_mm or "",
        "operation_hours": lcd.operation_hours,
        "service_life_hours": lcd.service_life_hours,
        "is_splicing": lcd.is_splicing,
        "contrast_ratio": lcd.contrast_ratio,
    })
    return Document(page_content=_lcd_text(lcd, model), metadata=metadata)


def _ifp_text(product: "IFPProduct", model: str, size_inch: str) -> str:
    parts = [
        f"{model} | {product.series} | IFP interactive flat panel | indoor",
        f"size={size_inch}",
        f"resolution={product.resolution}",
        f"system={product.system}",
        f"touch={product.touch_points}-point",
        f"memory={product.memory}",
        f"wifi={product.wifi}",
    ]
    if product.features:
        parts.append("features: " + ", ".join(product.features))
    return " | ".join(part for part in parts if part)


def _ifp_model_documents(product: "Product", ifp: "IFPProduct") -> list[Document]:
    """IFP：每个尺寸型号一条 Model 级文档（没有 sub_models 时退回系列一条）。"""
    sub_models = list(ifp.sub_models or [])
    if not sub_models:
        metadata = _display_metadata(
            model=product.product_id,
            series_id=product.product_id,
            series=product.series,
            display_type="IFP",
            environment="indoor" if "indoor" in product.environment else "outdoor",
            features=list(product.features or []),
        )
        metadata.update({
            "system": ifp.system,
            "touch_points": ifp.touch_points,
            "memory": ifp.memory,
            "wifi": ifp.wifi,
        })
        return [
            Document(
                page_content=_ifp_text(ifp, product.product_id, ""),
                metadata=metadata,
            )
        ]

    documents: list[Document] = []
    for sub in sub_models:
        metadata = _display_metadata(
            model=sub.model,
            series_id=product.product_id,
            series=product.series,
            display_type="IFP",
            environment="indoor" if "indoor" in product.environment else "outdoor",
            features=list(product.features or []),
        )
        metadata.update({
            "system": ifp.system,
            "touch_points": ifp.touch_points,
            "memory": ifp.memory,
            "wifi": ifp.wifi,
            "display_size_inch": sub.size_inch,
            "has_camera_mic": bool(sub.has_camera_mic),
            "resolution": ifp.resolution,
        })
        documents.append(
            Document(page_content=_ifp_text(ifp, sub.model, sub.size_inch), metadata=metadata)
        )
    return documents


def non_led_model_documents(data_dir: str) -> list[Document]:
    """LCD / IFP 的 Model 级文档（LED 继续走 CanonicalModel）。"""
    documents: list[Document] = []
    for product in load_structured_products(data_dir):
        if product.lcd_data is not None:
            documents.append(_lcd_model_document(product, product.lcd_data))
        elif product.ifp_data is not None:
            documents.extend(_ifp_model_documents(product, product.ifp_data))
    return documents


# ── Phase 1：Model 级标准化数据 ─────────────────────────────────────────────
def load_canonical_models(data_dir: str) -> list[CanonicalModel]:
    """把 Series 数据展开成 Model 级标准化记录（一个实际销售型号一条）。

    这是 Phase 2（Model 级 RAG）、Phase 8（推荐引擎）与
    Phase 9（箱体模组计算）的共同数据源。
    """
    products = load_structured_products(data_dir)
    models: list[CanonicalModel] = []
    errors: list[str] = []
    for product in products:
        if not product.led_data:
            continue
        try:
            models.extend(product.led_data.canonical_models())
        except Exception as exc:  # pragma: no cover - 防御式
            errors.append(f"{product.product_id}: {exc}")
    if errors:
        logger.error("Canonical model expansion errors:\n%s", "\n".join(errors))
    logger.info("Expanded %d canonical model records", len(models))
    return models


def canonical_model_index(data_dir: str) -> dict[str, CanonicalModel]:
    """``model`` → ``CanonicalModel`` 索引，供推荐/计算模块 O(1) 查询。"""
    return {m.model: m for m in load_canonical_models(data_dir)}


def validate_product_data(data_dir: str) -> dict[str, Any]:
    """执行计划文档 Phase 1「数据检查」清单。

    返回 ``{"ok": bool, "errors": [...], "warnings": [...], "stats": {...}}``。
    errors 表示数据不可用（必须修）；warnings 表示厂商规格本身存在的不一致
    （保留原始值，但在推荐/计算时以物理尺寸为准）。
    """
    errors: list[str] = []
    warnings: list[str] = []

    products = load_structured_products(data_dir)
    if not products:
        return {
            "ok": False,
            "errors": ["未加载到任何产品数据"],
            "warnings": [],
            "stats": {"series": 0, "models": 0},
        }

    seen_series: dict[str, str] = {}
    seen_models: dict[str, str] = {}

    for product in products:
        led = product.led_data
        if not led:
            errors.append(f"{product.product_id}: 非 LED 产品，Phase 1 尚未覆盖")
            continue

        # ── Series 唯一性 ──
        if product.product_id in seen_series:
            errors.append(f"重复 Series: {product.product_id}")
        seen_series[product.product_id] = led.series_id or product.product_id

        # ── Series 级必需字段 ──
        if not led.environment:
            errors.append(f"{product.product_id}: 缺少 environment")
        if led.installation is None:
            errors.append(f"{product.product_id}: 缺少 installation")
        if led.geometry is None:
            errors.append(f"{product.product_id}: 缺少 geometry（无法做箱体/模组计算）")
        if not led.sub_models:
            errors.append(f"{product.product_id}: 没有任何 Model")
            continue

        # ── 亮度区间 ──
        if led.brightness_min_nit and led.brightness_max_nit:
            if led.brightness_min_nit > led.brightness_max_nit:
                errors.append(
                    f"{product.product_id}: brightness_min {led.brightness_min_nit} > "
                    f"brightness_max {led.brightness_max_nit}"
                )

        # ── Model 级检查 ──
        for sub in led.sub_models:
            if sub.model in seen_models:
                errors.append(f"重复 Model: {sub.model}（同时出现在 {seen_models[sub.model]} 与本系列）")
            seen_models[sub.model] = product.product_id

            if not (0.5 <= sub.pixel_pitch_mm <= 20):
                errors.append(f"{sub.model}: 点间距 {sub.pixel_pitch_mm} 超出合理范围")
            if not sub.module_resolution:
                errors.append(f"{sub.model}: 缺少 module_resolution")

            # resolution_per_sqm 与点间距一致性（±5%）
            expected_per_sqm = 1_000_000 / (sub.pixel_pitch_mm ** 2)
            deviation = abs(sub.resolution_per_sqm - expected_per_sqm) / expected_per_sqm
            if deviation > 0.05:
                warnings.append(
                    f"{sub.model}: resolution_per_sqm={sub.resolution_per_sqm} 与点间距 "
                    f"{sub.pixel_pitch_mm}mm 理论值 {expected_per_sqm:.0f} 偏差 {deviation:.1%}"
                )

            # module_resolution 与模组尺寸 ÷ 点间距一致性
            if led.geometry and sub.module_resolution:
                parsed = parse_resolution(sub.module_resolution)
                if parsed:
                    expected_module = (
                        led.geometry.module_width_mm / sub.pixel_pitch_mm,
                        led.geometry.module_height_mm / sub.pixel_pitch_mm,
                    )
                    for axis, actual, expected in zip(
                        ("width", "height"), parsed, expected_module
                    ):
                        if expected <= 0:
                            continue
                        axis_deviation = abs(actual - expected) / expected
                        if axis_deviation > 0.05:
                            warnings.append(
                                f"{sub.model}: module_resolution {axis}={actual} 与 "
                                f"{led.geometry.module_width_mm}x{led.geometry.module_height_mm}mm ÷ "
                                f"{sub.pixel_pitch_mm}mm ≈ {expected:.0f} 偏差 {axis_deviation:.1%}"
                            )
                            break

            # cabinet_resolution 应等于 每箱模组排布 × module_resolution
            if led.geometry and sub.cabinet_resolution and sub.module_resolution:
                cabinet_px = parse_resolution(sub.cabinet_resolution)
                module_px = parse_resolution(sub.module_resolution)
                if cabinet_px and module_px:
                    per_row = round(led.geometry.cabinet_width_mm / led.geometry.module_width_mm)
                    per_col = round(led.geometry.cabinet_height_mm / led.geometry.module_height_mm)
                    expected_cabinet = (module_px[0] * per_row, module_px[1] * per_col)
                    if cabinet_px != expected_cabinet:
                        warnings.append(
                            f"{sub.model}: cabinet_resolution={sub.cabinet_resolution} 与 "
                            f"module_resolution 平铺值 {expected_cabinet[0]}*{expected_cabinet[1]} 不一致"
                        )

            # 型号级亮度必须落在 Series 区间内
            if sub.brightness_nit is not None and led.brightness_min_nit and led.brightness_max_nit:
                if not (led.brightness_min_nit <= sub.brightness_nit <= led.brightness_max_nit):
                    errors.append(
                        f"{sub.model}: 亮度 {sub.brightness_nit} 超出 Series 区间 "
                        f"[{led.brightness_min_nit}, {led.brightness_max_nit}]"
                    )

    # ── Model 级记录完整性（验收标准）──
    canonical = load_canonical_models(data_dir)
    required_attrs = (
        "model", "series_id", "pixel_pitch_mm", "brightness_nit",
        "module_width_mm", "module_height_mm",
        "cabinet_width_mm", "cabinet_height_mm", "modules_per_cabinet",
    )
    incomplete = [
        m.model for m in canonical
        if any(getattr(m, attr, None) in (None, "", 0) for attr in required_attrs)
    ]
    if incomplete:
        errors.append(f"以下 Model 记录不完整: {', '.join(incomplete)}")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "series": len(products),
            "models": len(canonical),
            "series_ids": sorted(seen_series.values()),
            "indoor_series": sum(1 for p in products if p.led_data and "indoor" in p.led_data.environment),
            "outdoor_series": sum(1 for p in products if p.led_data and "outdoor" in p.led_data.environment),
            "rental_series": sum(1 for p in products if p.led_data and p.led_data.is_rental),
            "fixed_series": sum(1 for p in products if p.led_data and not p.led_data.is_rental),
        },
    }


def _main() -> int:
    """``python -m src.rag.json_loader --validate`` 数据检查入口。"""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1 产品数据检查")
    parser.add_argument("--data-dir", default=None, help="产品数据目录（默认取 config.DATA_DIR）")
    parser.add_argument("--validate", action="store_true", help="执行数据检查")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir:
        from src.config import config
        data_dir = config.DATA_DIR

    report = validate_product_data(data_dir)
    stats = report["stats"]
    print(f"产品数据检查: {'通过' if report['ok'] else '失败'}")
    print(f"  Series: {stats['series']}  Model: {stats['models']}")
    print(
        f"  室内 {stats['indoor_series']} / 户外 {stats['outdoor_series']} / "
        f"固装 {stats['fixed_series']} / 租赁 {stats['rental_series']}"
    )
    if report["errors"]:
        print("\n[ERRORS]")
        for item in report["errors"]:
            print(f"  - {item}")
    if report["warnings"]:
        print("\n[WARNINGS]（厂商规格自身不一致，保留原值）")
        for item in report["warnings"]:
            print(f"  - {item}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())


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
