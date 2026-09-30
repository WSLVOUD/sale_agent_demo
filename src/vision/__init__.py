"""
智谱视觉需求提取模块（计划《智谱视觉需求提取接入实施计划》）。

对外只暴露三件事：
    ZhipuVisionClient    —— 调智谱视觉模型，拿原始文本
    VisionExtractor      —— 图片 → VisionRequirement（结构化 + 标准化 + 缓存）
    apply_vision_to_profile —— VisionRequirement → 合并进 RequirementProfile

视觉模块**不**推荐产品、不选型、不追问、不算箱体。
"""
from src.vision.client import VisionError, ZhipuVisionClient, check_image_payload
from src.vision.extractor import (
    VisionExtractor,
    get_vision_extractor,
    merge_vision_results,
    parse_vision_json,
)
from src.vision.integration import (
    apply_vision_to_profile,
    extract_vision_for_turn,
    resolve_vision_confirmation,
)
from src.vision.schema import (
    VISION_EXPLICIT,
    VISION_INFERRED,
    VisionField,
    VisionRequirement,
)
from src.vision.recognition import (
    CANDIDATE_FIELDS,
    CONFIRMABLE_FIELDS,
    ImageRecognitionResult,
)
# LCD 那一路（新增）：类型只产出 LED / LCD，只取"是否拼接 / 有没有摄像头"。
# LED 那一路（VisionExtractor + prompts.VISION_SYSTEM_PROMPT）原样保留。
from src.vision.lcd_extractor import (
    LCD_VISION_FIELDS,
    LcdVisionExtractor,
    extract_lcd_vision,
    get_lcd_vision_extractor,
)

__all__ = [
    "CANDIDATE_FIELDS",
    "CONFIRMABLE_FIELDS",
    "ImageRecognitionResult",
    "LCD_VISION_FIELDS",
    "LcdVisionExtractor",
    "VISION_EXPLICIT",
    "VISION_INFERRED",
    "VisionError",
    "VisionExtractor",
    "VisionField",
    "VisionRequirement",
    "ZhipuVisionClient",
    "apply_vision_to_profile",
    "check_image_payload",
    "extract_lcd_vision",
    "extract_vision_for_turn",
    "get_vision_extractor",
    "get_lcd_vision_extractor",
    "merge_vision_results",
    "parse_vision_json",
    "resolve_vision_confirmation",
]
