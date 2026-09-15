"""
加载并解析 company_profile.txt，生成结构化的 CompanyProfile。

修改 data/company_profile.txt 即可更新接待信息，无需改代码。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import re
import logging

from src.config import config

logger = logging.getLogger(__name__)


@dataclass
class CompanyProfile:
    """
    公司与销售人员信息。

    所有字段从 data/company_profile.txt 解析而来。
    """
    sales_name: str = ""
    position: str = ""
    company: str = ""
    location: str = ""
    company_introduction: str = ""
    main_products: list[str] = field(default_factory=list)
    main_markets: list[str] = field(default_factory=list)
    company_advantages: list[str] = field(default_factory=list)


def _resolve_profile_path() -> Path:
    """解析 company_profile.txt 路径。"""
    p = Path(config.DATA_DIR) / "company_profile.txt"
    if p.exists():
        return p
    # 后备：项目根目录
    root = Path(config.PROJECT_ROOT) / "data" / "company_profile.txt"
    if root.exists():
        return root
    raise FileNotFoundError(
        f"company_profile.txt not found. "
        f"Checked: {p}, {root}"
    )


def _parse_profile_txt(text: str) -> CompanyProfile:
    """解析 company_profile.txt 的纯文本格式。"""
    profile = CompanyProfile()

    def extract(key: str) -> str:
        """提取 "Key:\\nvalue" 块的值。"""
        pattern = re.escape(key) + r":\s*\n([\s\S]*?)(?=\n[A-Za-z]|\\n[A-Za-z][A-Za-z ]+:|$)"
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            return m.group(1).strip()
        return ""

    def extract_lines(key: str) -> list[str]:
        """提取多行列表。"""
        block = extract(key)
        return [ln.strip() for ln in block.split("\n") if ln.strip()]

    profile.sales_name = extract("Sales Name")
    profile.position = extract("Position")
    profile.company = extract("Company")
    profile.location = extract("Location")
    # 兼容两种写法：Company Introduction / Company description
    profile.company_introduction = extract("Company Introduction") or extract("Company description")
    profile.main_products = extract_lines("Main Products")
    profile.main_markets = extract_lines("Main Markets")
    profile.company_advantages = extract_lines("Company Advantages")
    return profile


# 解析后的单例（模块级缓存，进程启动时加载一次）
_profile_cache: CompanyProfile | None = None


def load_profile() -> CompanyProfile:
    """加载并缓存 company_profile.txt。"""
    global _profile_cache
    if _profile_cache is None:
        path = _resolve_profile_path()
        text = path.read_text(encoding="utf-8")
        _profile_cache = _parse_profile_txt(text)
        logger.info(f"Loaded company profile: {_profile_cache.sales_name} @ {_profile_cache.company}")
    return _profile_cache


def reload_profile() -> CompanyProfile:
    """强制重新加载（测试或热更新用）。"""
    global _profile_cache
    _profile_cache = None
    return load_profile()
