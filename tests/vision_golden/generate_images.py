"""
生成 Vision Golden Dataset 的第一版图片（计划「第二十三阶段：测试体系」）。

说明（重要）：
    这里生成的是**合成图**，用来回归"图片 → 结构化需求 → 合并 Profile"的链路，
    以及模型"看不出来时是否会瞎填"（null precision）。
    精度评测应当用**脱敏后的真实客户图片**替换/补充本目录下的图片 —— 合成图
    不能代表真实照片上的识别准确率。

用法（需要 Pillow）：
    python tests/vision_golden/generate_images.py
"""
from __future__ import annotations

import json
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
MANIFEST = os.path.join(HERE, "manifest.json")


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _base(width=640, height=480, color=(28, 30, 36)):
    image = Image.new("RGB", (width, height), color)
    return image, ImageDraw.Draw(image)


def indoor_conference(label=True):
    image, draw = _base()
    draw.rectangle([0, 380, 640, 480], fill=(70, 60, 55))          # 地板
    draw.rectangle([90, 70, 550, 300], fill=(20, 20, 24))          # 屏体
    for x in range(90, 550, 46):                                   # 模组拼缝
        draw.line([(x, 70), (x, 300)], fill=(60, 60, 66), width=1)
    for y in range(70, 300, 46):
        draw.line([(90, y), (550, y)], fill=(60, 60, 66), width=1)
    draw.rectangle([150, 120, 490, 250], fill=(40, 90, 200))       # 屏上内容
    draw.ellipse([40, 320, 130, 370], fill=(120, 100, 90))         # 会议桌一角
    if label:
        draw.text((110, 320), "INDOOR CONFERENCE ROOM LED DISPLAY",
                  fill=(235, 235, 235), font=_font(20))
    return image


def outdoor_advertising(label=True):
    image, draw = _base(color=(120, 170, 230))
    draw.rectangle([0, 330, 640, 480], fill=(90, 90, 95))          # 地面
    draw.rectangle([70, 60, 570, 320], fill=(15, 15, 18))          # 大屏
    draw.rectangle([120, 110, 520, 270], fill=(220, 120, 40))
    draw.rectangle([70, 320, 570, 340], fill=(60, 60, 62))         # 钢结构
    if label:
        draw.text((150, 360), "OUTDOOR ADVERTISING LED SCREEN",
                  fill=(255, 255, 255), font=_font(22))
    return image


def rental_stage(label=True):
    image, draw = _base(color=(10, 10, 14))
    draw.rectangle([80, 60, 560, 300], fill=(25, 25, 30))
    draw.rectangle([200, 120, 440, 240], fill=(200, 60, 160))
    draw.rectangle([80, 300, 110, 460], fill=(45, 45, 50))         # 桁架立柱
    draw.rectangle([530, 300, 560, 460], fill=(45, 45, 50))
    if label:
        draw.text((150, 330), "RENTAL STAGE LED PANEL WITH TRUSS",
                  fill=(230, 230, 230), font=_font(20))
    return image


def dimension_drawing():
    """带尺寸标注的图纸：用来测"图片里明确写了尺寸"的情况。"""
    image, draw = _base(color=(245, 245, 240))
    draw.rectangle([120, 90, 520, 330], outline=(30, 30, 30), width=3)
    draw.line([(120, 60), (520, 60)], fill=(200, 30, 30), width=2)
    draw.text((250, 30), "5000 mm", fill=(200, 30, 30), font=_font(22))
    draw.line([(560, 90), (560, 330)], fill=(200, 30, 30), width=2)
    draw.text((570, 190), "3000 mm", fill=(200, 30, 30), font=_font(22))
    draw.text((140, 350), "INDOOR LED WALL - FIXED INSTALLATION",
              fill=(40, 40, 40), font=_font(20))
    return image


def no_screen_wall():
    """没有屏幕的普通墙面 → 所有字段都必须是 null。"""
    image, draw = _base(color=(210, 205, 195))
    for y in range(0, 480, 60):
        draw.line([(0, y), (640, y)], fill=(190, 185, 175), width=2)
    draw.rectangle([520, 380, 620, 470], fill=(120, 110, 100))
    return image


def no_screen_landscape():
    image, draw = _base(color=(150, 200, 235))
    draw.rectangle([0, 300, 640, 480], fill=(110, 160, 90))
    draw.ellipse([480, 40, 580, 140], fill=(255, 230, 120))
    return image


CASES = [
    ("indoor_conference_1.png", indoor_conference,
     {"environment": "indoor", "display_type": "LED", "notes": "synthetic"}),
    ("indoor_conference_2.png", lambda: indoor_conference(label=False),
     {"environment": "indoor", "display_type": "LED", "notes": "synthetic, no text label"}),
    ("outdoor_advertising_1.png", outdoor_advertising,
     {"environment": "outdoor", "display_type": "LED", "notes": "synthetic"}),
    ("outdoor_advertising_2.png", lambda: outdoor_advertising(label=False),
     {"environment": "outdoor", "display_type": "LED", "notes": "synthetic, no text label"}),
    ("rental_stage_1.png", rental_stage,
     {"environment": "indoor", "display_type": "LED", "purpose": "stage", "notes": "synthetic"}),
    ("dimension_drawing.png", dimension_drawing,
     {"environment": "indoor", "display_type": "LED", "installation": "fixed",
      "target_width_m": 5.0, "target_height_m": 3.0, "notes": "synthetic drawing"}),
    ("no_screen_wall.png", no_screen_wall,
     {"environment": None, "display_type": None, "notes": "no screen — expect nulls"}),
    ("no_screen_landscape.png", no_screen_landscape,
     {"environment": None, "display_type": None, "notes": "no screen — expect nulls"}),
]


def main() -> None:
    os.makedirs(IMAGE_DIR, exist_ok=True)
    manifest = []
    for name, factory, expected in CASES:
        path = os.path.join(IMAGE_DIR, name)
        factory().save(path)
        manifest.append({"image": name, "expected": expected})
        print("wrote", path)
    with open(MANIFEST, "w", encoding="utf-8") as handle:
        json.dump({"version": 1, "cases": manifest}, handle, ensure_ascii=False, indent=2)
    print("wrote", MANIFEST)


if __name__ == "__main__":
    main()
