"""
首次接待素材配置。

所有素材路径统一在此管理，更换素材只需修改配置。
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

from src.config import config

# ── 素材配置 ──────────────────────────────────────────────────────────────

@dataclass
class FirstContactAsset:
    """单个素材的元信息。"""
    name: str              # 显示名称（如"案例视频 1"）
    filename: str          # 文件名（如"case_video_01.mp4"）
    asset_type: str        # 类型：video / pdf
    description: str       # 对客户展示时的简短描述


# 首次接待固定发送的素材列表（按顺序）
# 注意：产品手册（PDF）暂时不在首次接待自动发送，如有需要可取消下方注释
ASSETS: list[FirstContactAsset] = [
    FirstContactAsset(
        name="Case Video 1",
        filename="case_video_01.mp4",
        asset_type="video",
        description="Check out our latest project case — factory tour & installation highlights",
    ),
    FirstContactAsset(
        name="Case Video 2",
        filename="case_video_02.mp4",
        asset_type="video",
        description="Here is another real application demo — outdoor advertising in action",
    ),
    # ⚠️ 产品手册暂时不在首次接待自动发送，如有需要可添加：
    # Product Catalog 1: product_catalog_01.pdf
    # Product Catalog 2: product_catalog_02.pdf
]


# ── 路径工具 ──────────────────────────────────────────────────────────────

def get_first_contact_dir() -> Path:
    """首次接待素材目录。"""
    return Path(config.DATA_DIR) / "first_contact"


def get_asset_path(asset: FirstContactAsset) -> Path:
    """获取素材的完整磁盘路径。"""
    return get_first_contact_dir() / asset.filename


def get_asset_url(asset: FirstContactAsset) -> str:
    """
    获取素材的访问 URL（供前端 / API 返回）。

    生产环境可替换为 CDN / OSS 等外部 URL。
    当前默认使用 /static/first_contact/ 前缀。
    """
    return f"/static/first_contact/{asset.filename}"


def list_missing_assets() -> list[FirstContactAsset]:
    """返回缺失的素材列表（文件不存在）。"""
    missing = []
    for asset in ASSETS:
        if not get_asset_path(asset).exists():
            missing.append(asset)
    return missing


def get_available_assets() -> list[FirstContactAsset]:
    """返回存在的素材列表。"""
    return [a for a in ASSETS if get_asset_path(a).exists()]
