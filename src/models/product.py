"""
产品数据 Pydantic 模型。

Schema 版本: v2.0（Phase 1 产品数据标准化）
用于验证和结构化产品数据，确保产品参数的唯一事实来源。

Phase 1 新增：
  - ``LEDGeometry``：模组 / 箱体物理尺寸 + 每箱模组数（工程计算的唯一来源）
  - ``CanonicalModel``：Model 级标准化记录（一个实际销售型号一条），
    由 ``LEDProduct.canonical_models()`` 展开，供 Model 级 RAG 与推荐引擎使用
  - ``installation``：fixed / rental，与 ``is_rental`` 互为校验
"""
from __future__ import annotations

import re
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ── 基础枚举 ────────────────────────────────────────────────────────────────
DisplayType = Literal["LED", "LCD", "IFP"]
Environment = Literal["indoor", "outdoor", "semi_outdoor"]
ProductCategory = Literal["display", "mount", "module", "software"]
Installation = Literal["fixed", "rental"]


def parse_size_mm(value: str | None) -> tuple[float, float] | None:
    """把 ``"640mm*480mm"`` / ``"600*337.5mm"`` 解析为 (width_mm, height_mm)。"""
    if not value:
        return None
    numbers = re.findall(r"(\d+(?:\.\d+)?)", str(value))
    if len(numbers) < 2:
        return None
    return float(numbers[0]), float(numbers[1])


def parse_resolution(value: str | None) -> tuple[int, int] | None:
    """把 ``"256*128"`` 解析为 (width_px, height_px)；``"N/A"`` 返回 None。"""
    if not value:
        return None
    text = str(value).strip()
    if text.upper().startswith("N/A"):
        return None
    numbers = re.findall(r"(\d+)", text)
    if len(numbers) < 2:
        return None
    return int(numbers[0]), int(numbers[1])


# ── LED 子型号 ──────────────────────────────────────────────────────────────
class LEDSubModel(BaseModel):
    """LED 屏具体型号（点间距规格）。"""
    model: str = Field(..., description="完整型号名，如 TW21-3216-P2.0")
    pixel_pitch_mm: float = Field(..., description="点间距（mm）", ge=0.5, le=20)
    resolution_per_sqm: int = Field(..., description="每平米像素数")
    module_resolution: str = Field(..., description="模组分辨率，如 160*80")
    cabinet_resolution: Optional[str] = Field(None, description="箱体分辨率，如 320*240；不适用时为 null")
    scanning: Optional[str] = Field(None, description="扫描方式，如 32s")
    brightness_nit: Optional[int] = Field(
        None, description="型号级亮度（cd/m²）；为空时沿用 Series 亮度", ge=50, le=15000
    )

    @model_validator(mode="after")
    def _validate_model_code(self):
        """型号名里的 P 标记必须与实际点间距量级一致。

        厂商命名是营销标签，允许一定偏差（如 ``P6`` 实际 6.67mm、``P1.95`` 实际 1.95mm）。
        只用 L1 距离与宽松阈值做量级校验，避免把合法的命名误判为错误。
        """
        parsed = re.search(r"[Pp](\d+(?:\.\d+)?)", self.model)
        if parsed:
            code_pitch = float(parsed.group(1))
            tolerance = max(0.2, code_pitch * 0.15)
            if abs(code_pitch - self.pixel_pitch_mm) > tolerance:
                raise ValueError(
                    f"{self.model} 命名点间距 {code_pitch} 与实测 {self.pixel_pitch_mm} 偏差过大"
                )
        return self


# ── 几何数据（模组 / 箱体）──────────────────────────────────────────────────
class LEDGeometry(BaseModel):
    """模组 / 箱体物理尺寸与每箱模组数 —— 工程计算的唯一来源。"""
    module_width_mm: float = Field(..., gt=0)
    module_height_mm: float = Field(..., gt=0)
    cabinet_width_mm: float = Field(..., gt=0)
    cabinet_height_mm: float = Field(..., gt=0)
    modules_per_cabinet: int = Field(..., ge=1)

    @classmethod
    def from_size_strings(cls, module_size_mm: str, cabinet_size_mm: str) -> "LEDGeometry":
        """从 ``"320mm*160mm"`` 这类字符串推导几何数据。"""
        module = parse_size_mm(module_size_mm)
        cabinet = parse_size_mm(cabinet_size_mm)
        if not module or not cabinet:
            raise ValueError("无法从尺寸字符串推导 geometry")
        per_row = round(cabinet[0] / module[0])
        per_column = round(cabinet[1] / module[1])
        return cls(
            module_width_mm=module[0],
            module_height_mm=module[1],
            cabinet_width_mm=cabinet[0],
            cabinet_height_mm=cabinet[1],
            modules_per_cabinet=max(1, per_row * per_column),
        )

    @model_validator(mode="after")
    def _validate_tiling(self):
        """箱体必须能被模组整除（允许 1% 误差）。"""
        for axis, cabinet, module in (
            ("width", self.cabinet_width_mm, self.module_width_mm),
            ("height", self.cabinet_height_mm, self.module_height_mm),
        ):
            ratio = cabinet / module
            if abs(ratio - round(ratio)) > 0.01:
                raise ValueError(
                    f"cabinet {axis} {cabinet} 不能被 module {axis} {module} 整除"
                )
        expected = round(self.cabinet_width_mm / self.module_width_mm) * round(
            self.cabinet_height_mm / self.module_height_mm
        )
        if expected != self.modules_per_cabinet:
            raise ValueError(
                f"modules_per_cabinet={self.modules_per_cabinet} 与几何推导值 {expected} 不一致"
            )
        return self


# ── LED 产品 ────────────────────────────────────────────────────────────────
class LEDProduct(BaseModel):
    """LED 显示产品。"""
    product_id: str = Field(..., description="产品唯一标识，如 TW21-3216")
    series: str = Field(..., description="系列名，如 TW21-3216 series")
    series_id: Optional[str] = Field(None, description="Series ID，如 TW21-3216")
    display_type: Literal["LED"] = "LED"
    series_description: str = Field(..., description="系列描述，用于向量库检索")
    environment: list[Environment] = Field(..., description="适用环境")
    is_rental: bool = Field(..., description="是否支持租赁场景")
    installation: Optional[Installation] = Field(
        None, description="安装方式：fixed / rental（Phase 1 新增，与 is_rental 互为校验）"
    )
    brightness_nit: int = Field(..., description="亮度（cd/m² 或 nit）", ge=100, le=15000)
    brightness_min_nit: Optional[int] = Field(None, description="Series 最低亮度", ge=50, le=15000)
    brightness_max_nit: Optional[int] = Field(None, description="Series 最高亮度", ge=50, le=15000)
    refresh_rate_hz: int = Field(..., description="刷新率（Hz）", ge=1000, le=8000)
    module_size_mm: str = Field(..., description="模组尺寸，如 320mm*160mm")
    cabinet_size_mm: Optional[str] = Field(None, description="箱体尺寸，如 640mm*480mm")
    geometry: Optional[LEDGeometry] = Field(
        None, description="模组/箱体物理尺寸与每箱模组数；缺省时由尺寸字符串推导"
    )
    warranty_years: int = Field(..., description="质保年数", ge=1, le=5)
    lamp_brand: Optional[str] = Field(None, description="灯珠品牌，如 Kinglight")
    features: list[str] = Field(default_factory=list, description="特性标签列表")
    sub_models: list[LEDSubModel] = Field(default_factory=list, description="子型号列表")
    price_tier: Literal["low", "mid", "high"] = Field(..., description="价格档位")
    cob: bool = Field(False, description="是否采用 COB 技术")
    hdr: bool = Field(False, description="是否支持 HDR")
    waterproof: bool = Field(False, description="是否防水")

    @field_validator("environment", mode="before")
    @classmethod
    def _parse_environment(cls, v):
        if isinstance(v, str):
            return [v]
        return v

    @model_validator(mode="after")
    def _normalize_phase1_fields(self):
        """Phase 1 一致性处理：补齐 series_id / installation / brightness 区间 / geometry。"""
        # series_id 缺省时从 product_id 推导
        if not self.series_id:
            self.series_id = self.product_id

        # installation 与 is_rental 必须一致
        expected_installation: Installation = "rental" if self.is_rental else "fixed"
        if self.installation is None:
            self.installation = expected_installation
        elif self.installation != expected_installation:
            raise ValueError(
                f"{self.product_id}: installation={self.installation} 与 is_rental={self.is_rental} 冲突"
            )

        # 亮度区间缺省时用 Series 亮度补齐
        if self.brightness_min_nit is None:
            self.brightness_min_nit = min(
                [self.brightness_nit] + [s.brightness_nit for s in self.sub_models if s.brightness_nit]
            )
        if self.brightness_max_nit is None:
            self.brightness_max_nit = max(
                [self.brightness_nit] + [s.brightness_nit for s in self.sub_models if s.brightness_nit]
            )

        # geometry 缺省时从尺寸字符串推导
        if self.geometry is None and self.cabinet_size_mm:
            self.geometry = LEDGeometry.from_size_strings(
                self.module_size_mm, self.cabinet_size_mm
            )
        return self

    # ── Phase 1：展开成 Model 级标准化记录 ──────────────────────────────────
    def canonical_models(self) -> list["CanonicalModel"]:
        """把 Series 展开成完整的 Model 级记录（一个型号一条）。"""
        if not self.geometry:
            raise ValueError(f"{self.product_id} 缺少 geometry，无法展开 Model 级记录")

        records: list[CanonicalModel] = []
        for sub in self.sub_models:
            brightness = sub.brightness_nit or self.brightness_nit
            records.append(
                CanonicalModel(
                    model=sub.model,
                    series_id=self.series_id or self.product_id,
                    series=self.series,
                    display_type="LED",
                    environment=list(self.environment),
                    installation=self.installation or ("rental" if self.is_rental else "fixed"),
                    indoor="indoor" in self.environment,
                    outdoor="outdoor" in self.environment,
                    series_description=self.series_description,
                    features=list(self.features),
                    pixel_pitch_mm=sub.pixel_pitch_mm,
                    resolution_per_sqm=sub.resolution_per_sqm,
                    module_resolution=sub.module_resolution,
                    cabinet_resolution=sub.cabinet_resolution,
                    scanning=sub.scanning,
                    brightness_nit=brightness,
                    brightness_min_nit=self.brightness_min_nit or brightness,
                    brightness_max_nit=self.brightness_max_nit or brightness,
                    refresh_rate_hz=self.refresh_rate_hz,
                    warranty_years=self.warranty_years,
                    lamp_brand=self.lamp_brand,
                    price_tier=self.price_tier,
                    cob=self.cob,
                    hdr=self.hdr,
                    waterproof=self.waterproof,
                    module_width_mm=self.geometry.module_width_mm,
                    module_height_mm=self.geometry.module_height_mm,
                    cabinet_width_mm=self.geometry.cabinet_width_mm,
                    cabinet_height_mm=self.geometry.cabinet_height_mm,
                    modules_per_cabinet=self.geometry.modules_per_cabinet,
                )
            )
        return records


# ── Model 级标准化记录（Phase 1 核心交付）───────────────────────────────────
class CanonicalModel(BaseModel):
    """一个实际销售型号的完整技术档案（RAG 检索与工程计算的唯一事实来源）。"""
    model: str
    series_id: str
    series: str
    display_type: Literal["LED"] = "LED"
    environment: list[Environment]
    installation: Installation
    indoor: bool
    outdoor: bool
    series_description: str = ""
    features: list[str] = Field(default_factory=list)

    # 光学 / 像素
    pixel_pitch_mm: float = Field(..., gt=0)
    resolution_per_sqm: int = Field(..., gt=0)
    module_resolution: str
    cabinet_resolution: Optional[str] = None
    scanning: Optional[str] = None
    brightness_nit: int
    brightness_min_nit: int
    brightness_max_nit: int
    refresh_rate_hz: int = 3840
    warranty_years: int = 1

    # 商业属性
    lamp_brand: Optional[str] = None
    price_tier: Literal["low", "mid", "high"] = "mid"
    cob: bool = False
    hdr: bool = False
    waterproof: bool = False

    # 工程尺寸（模组 / 箱体）
    module_width_mm: float = Field(..., gt=0)
    module_height_mm: float = Field(..., gt=0)
    cabinet_width_mm: float = Field(..., gt=0)
    cabinet_height_mm: float = Field(..., gt=0)
    modules_per_cabinet: int = Field(..., ge=1)

    @property
    def module_size_text(self) -> str:
        return f"{_trim(self.module_width_mm)}mm*{_trim(self.module_height_mm)}mm"

    @property
    def gob(self) -> bool:
        """GOB 灌胶防护（目前由型号命名体现，如 TW11-IR-P1.95(GOB)）。"""
        return "GOB" in self.model.upper()

    @property
    def flexible(self) -> bool:
        """是否为柔性/可弯曲屏（功能硬要求，客户明确提出时必须满足）。"""
        return any("flexible" in str(feature).lower() for feature in self.features)

    @property
    def cabinet_size_text(self) -> str:
        return f"{_trim(self.cabinet_width_mm)}mm*{_trim(self.cabinet_height_mm)}mm"

    def to_text(self) -> str:
        """生成用于向量检索的 Model 级文本。"""
        env = "indoor" if self.indoor and not self.outdoor else "outdoor" if self.outdoor else "indoor"
        install = "rental" if self.installation == "rental" else "fixed installation"
        extras = [name for name, flag in (("COB", self.cob), ("HDR", self.hdr), ("waterproof", self.waterproof)) if flag]
        parts = [
            f"Model {self.model}",
            f"{self.series_id} series",
            f"{env} {install} LED display",
            f"pixel pitch {self.pixel_pitch_mm}mm",
            f"{self.resolution_per_sqm} pixels/sqm",
            f"brightness {self.brightness_nit}nit",
            f"refresh {self.refresh_rate_hz}Hz",
            f"module {self.module_size_text}",
            f"cabinet {self.cabinet_size_text}",
            f"{self.modules_per_cabinet} modules per cabinet",
            f"price tier {self.price_tier}",
        ]
        if self.lamp_brand:
            parts.append(f"lamp {self.lamp_brand}")
        if extras:
            parts.append(" ".join(extras))
        if self.series_description:
            parts.append(self.series_description)
        return ", ".join(parts)


def _trim(value: float) -> str:
    """把 337.5 显示为 ``337.5``，640.0 显示为 ``640``。"""
    return f"{value:g}"


# ── LCD 产品 ───────────────────────────────────────────────────────────────
class LCDProduct(BaseModel):
    """LCD 商用显示产品。"""
    product_id: str = Field(..., description="产品唯一标识，如 HG5530LN")
    series: str = Field(..., description="系列名")
    display_type: Literal["LCD"] = "LCD"
    environment: list[Environment] = Field(default_factory=lambda: ["indoor"], description="适用环境")
    brightness_nit: int = Field(..., description="亮度（cd/m²）", ge=100, le=3000)
    contrast_ratio: str = Field(..., description="对比度，如 1200:1")
    display_size_inch: str = Field(..., description="屏幕尺寸，如 55\"")
    bazel_mm: str = Field(..., description="拼缝（边框）宽度，如 1.8mm")
    resolution: str = Field(..., description="分辨率，如 3840x2160@60Hz")
    panel_brand: Optional[str] = Field(None, description="面板品牌，如 LG、BOE")
    operation_hours: str = Field(..., description="支持运行时长，如 7x24h")
    service_life_hours: int = Field(..., description="使用寿命（小时）", ge=10000)
    power_consumption_w: Optional[int] = Field(None, description="功耗（W）")
    installation: list[str] = Field(default_factory=lambda: ["wall_mount"], description="安装方式")
    features: list[str] = Field(default_factory=list, description="特性标签")
    # LCD 拼接屏特有字段
    is_splicing: bool = Field(False, description="是否为拼接屏（有可见拼缝）")
    maintenance: Optional[str] = Field(None, description="维护方式")
    lifespan_hours: Optional[int] = Field(None, description="光源寿命（小时）")

    @field_validator("environment", mode="before")
    @classmethod
    def _parse_environment(cls, v):
        if isinstance(v, str):
            return [v]
        return v


# ── IFP 产品 ───────────────────────────────────────────────────────────────
class IFPSubModel(BaseModel):
    """IFP 交互平板具体型号。"""
    model: str = Field(..., description="完整型号名")
    size_inch: str = Field(..., description="尺寸，如 65\"")
    applicable_area_sqm: str = Field(..., description="适用面积，如 15-20m²")
    price_usd: float = Field(..., description="单价（美元）")
    has_camera_mic: bool = Field(False, description="是否含摄像头/麦克风")


class IFPProduct(BaseModel):
    """IFP 交互平板产品。"""
    product_id: str = Field(..., description="产品系列标识，如 OmniPAD-G4")
    series: str = Field(..., description="系列名")
    display_type: Literal["IFP"] = "IFP"
    environment: list[Environment] = Field(default_factory=lambda: ["indoor"], description="适用环境")
    system: str = Field(..., description="操作系统，如 Android 14")
    touch_points: int = Field(..., description="触控点数", ge=1)
    resolution: str = Field(..., description="分辨率")
    memory: str = Field(..., description="内存/存储，如 4G/32G")
    wifi: str = Field(..., description="WiFi 规格")
    features: list[str] = Field(default_factory=list, description="特性标签")
    sub_models: list[IFPSubModel] = Field(default_factory=list, description="子型号列表")
    application_scenarios: list[str] = Field(default_factory=list, description="适用场景")
    has_nfc: bool = Field(False, description="是否支持 NFC")
    has_ai_camera: bool = Field(False, description="是否含 AI 跟踪摄像头")
    has_hdmi_out: bool = Field(False, description="是否有 HDMI 输出")
    has_high_color_gamut: bool = Field(False, description="是否高色域屏")

    @field_validator("environment", mode="before")
    @classmethod
    def _parse_environment(cls, v):
        if isinstance(v, str):
            return [v]
        return v


# ── 联合产品模型 ────────────────────────────────────────────────────────────
class Product(BaseModel):
    """统一产品模型，支持 LED/LCD/IFP 三种类型。"""
    product_id: str = Field(..., description="产品唯一标识")
    series: str = Field(..., description="系列名")
    display_type: DisplayType
    environment: list[Environment] = Field(default_factory=lambda: ["indoor"])
    is_rental: bool = Field(False, description="是否支持租赁（IFP/LCD 固定为 False）")
    brightness_nit: Optional[int] = Field(None, description="亮度（cd/m²）")
    features: list[str] = Field(default_factory=list)

    # LED 特定字段
    led_data: Optional[LEDProduct] = Field(None, description="LED 型号详情")
    # LCD 特定字段
    lcd_data: Optional[LCDProduct] = Field(None, description="LCD 型号详情")
    # IFP 特定字段
    ifp_data: Optional[IFPProduct] = Field(None, description="IFP 型号详情")

    @field_validator("environment", mode="before")
    @classmethod
    def _parse_environment(cls, v):
        if isinstance(v, str):
            return [v]
        return v

    def get_description_text(self) -> str:
        """生成供向量库检索用的文本描述。"""
        parts = [f"{self.series}: {', '.join(self.features)}"]
        if self.led_data:
            parts.append(
                f"brightness={self.led_data.brightness_nit}nit, "
                f"refresh={self.led_data.refresh_rate_hz}Hz, "
                f"warranty={self.led_data.warranty_years}年"
            )
            for sub in self.led_data.sub_models:
                parts.append(
                    f"Model {sub.model}, pitch={sub.pixel_pitch_mm}mm, "
                    f"resolution={sub.resolution_per_sqm}px/m²"
                )
        elif self.lcd_data:
            parts.append(
                f"size={self.lcd_data.display_size_inch}, "
                f"resolution={self.lcd_data.resolution}, "
                f"brightness={self.lcd_data.brightness_nit}nit, "
                f"bazel={self.lcd_data.bazel_mm}, "
                f"7x24h={self.lcd_data.operation_hours}"
            )
        elif self.ifp_data:
            for sub in self.ifp_data.sub_models:
                parts.append(
                    f"Model {sub.model}, {sub.size_inch}, "
                    f"touch={self.ifp_data.touch_points}points, "
                    f"system={self.ifp_data.system}"
                )
        return " | ".join(parts)


# ── 验证函数 ────────────────────────────────────────────────────────────────
def validate_product(data: dict) -> Product:
    """验证并解析产品数据，验证失败时抛出异常。"""
    return Product.model_validate(data)
