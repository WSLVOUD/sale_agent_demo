"""
First Contact 模块：首次客户固定接待工作流。

职责：
- 客户第一次发送消息时，自动执行固定接待流程
- 自我介绍 + 案例视频 + 产品手册 PDF
- 素材发送失败不阻塞后续 AI 对话
"""
from .handler import FirstContactHandler, first_contact_handler
from .profile import CompanyProfile, load_profile
from .assets import ASSETS, FirstContactAsset, get_asset_path, get_asset_url

__all__ = [
    "FirstContactHandler",
    "first_contact_handler",
    "CompanyProfile",
    "load_profile",
    "ASSETS",
    "FirstContactAsset",
    "get_asset_path",
    "get_asset_url",
]
