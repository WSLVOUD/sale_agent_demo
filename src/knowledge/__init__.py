"""
知识库分离模块
Phase 3: Product/Sales 知识分离

将知识库拆分为：
- Product Knowledge: 型号、参数、产品能力、限制、适用场景
- Sales Knowledge: 销售话术、行业方案、异议处理、Closing、沟通策略

检索流程：
User Query
    ↓
Query Understanding
    ↓
┌───────────────┬────────────────┐
↓               ↓
Product Search  Sales Knowledge
↓               ↓
产品事实        销售策略
└────────┬──────┘
         ↓
        LLM
         ↓
    Final Response
"""
from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.config import config

logger = logging.getLogger(__name__)


class KnowledgeSource(str, Enum):
    """知识来源枚举"""
    PRODUCT = "product"      # 产品知识
    SALES = "sales"         # 销售话术
    GENERAL = "general"     # 通用知识


@dataclass
class KnowledgeItem:
    """知识条目"""
    id: str
    source: KnowledgeSource
    content: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source.value,
            "content": self.content,
            "metadata": self.metadata,
        }


@dataclass
class DualKnowledgeSearchResult:
    """双知识库搜索结果"""
    product_results: List[KnowledgeItem]
    sales_results: List[KnowledgeItem]
    query_intent: str
    needs_product_knowledge: bool
    needs_sales_knowledge: bool


class KnowledgeSeparator:
    """
    知识分离器
    
    将产品知识和销售知识分开存储和检索，
    避免将销售话术内容误认为产品真实参数。
    """
    
    # 产品相关关键词
    PRODUCT_KEYWORDS = {
        "型号", "参数", "亮度", "点间距", "分辨率", "尺寸", "规格",
        "像素", "刷新率", "功耗", "重量", "尺寸", "规格",
        "TW", "LED", "LCD", "IFP", "display", "screen",
        "pitch", "brightness", "resolution",
    }
    
    # 销售相关关键词
    SALES_KEYWORDS = {
        "推荐", "方案", "价格", "报价", "优惠", "折扣",
        "方案", "场景", "应用", "安装", "维护",
        "怎么选", "哪个好", "有什么区别", "比较",
        "质保", "保修", "售后", "服务",
        "异议", "顾虑", "担心", "竞争对手", "别家",
        "closing", "close", "成交", "签合同",
    }
    
    # 产品型号模式
    PRODUCT_MODEL_PATTERNS = [
        r"^TW\d+", r"^LC\d+", r"^IFP\d+",
        r"P\d+\.?\d*", r"^\d+\.?\d*mm",
    ]
    
    def __init__(self):
        self._product_vectorstore = None
        self._sales_vectorstore = None
    
    def classify_query(self, query: str) -> Tuple[str, bool, bool]:
        """
        分类查询类型
        
        Args:
            query: 用户查询
            
        Returns:
            (intent, needs_product, needs_sales)
        """
        query_lower = query.lower()
        
        # 检查是否包含产品型号
        import re
        has_product_model = any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in self.PRODUCT_MODEL_PATTERNS
        )
        
        # 统计关键词命中
        product_hits = sum(1 for kw in self.PRODUCT_KEYWORDS if kw.lower() in query_lower)
        sales_hits = sum(1 for kw in self.SALES_KEYWORDS if kw.lower() in query_lower)
        
        # 判断意图
        if has_product_model or product_hits > sales_hits:
            intent = "product_query"
            needs_product = True
            needs_sales = sales_hits > 0
        elif sales_hits > 0:
            intent = "sales_consultation"
            needs_product = product_hits > 0
            needs_sales = True
        else:
            # 默认需要两者
            intent = "general"
            needs_product = True
            needs_sales = True
        
        return intent, needs_product, needs_sales
    
    def get_retrieval_filter(self, source: KnowledgeSource) -> Optional[Dict]:
        """获取检索过滤器"""
        return {"source": source.value}
    
    def format_product_context(self, products: List[KnowledgeItem]) -> str:
        """格式化产品上下文"""
        if not products:
            return ""
        
        parts = ["【产品信息】"]
        for item in products[:5]:  # 最多 5 个产品
            parts.append(f"\n{item.content}")
        
        return "\n".join(parts)
    
    def format_sales_context(self, sales: List[KnowledgeItem]) -> str:
        """格式化销售上下文"""
        if not sales:
            return ""
        
        parts = ["【销售策略】"]
        for item in sales[:3]:  # 最多 3 条销售策略
            category = item.metadata.get("category", "general")
            parts.append(f"\n[{category}] {item.content}")
        
        return "\n".join(parts)


class ProductKnowledgeLoader:
    """
    产品知识加载器
    
    从产品数据文件加载并结构化存储。
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    def load_products(self) -> List[KnowledgeItem]:
        """加载所有产品知识"""
        products = []
        
        # 加载 LED 产品
        led_file = os.path.join(self.data_dir, "LED display.txt")
        if os.path.exists(led_file):
            products.extend(self._load_product_file(led_file, "LED"))
        
        # 加载 LCD 产品
        lcd_file = os.path.join(self.data_dir, "LCD display.txt")
        if os.path.exists(lcd_file):
            products.extend(self._load_product_file(lcd_file, "LCD"))
        
        # 加载 IFP 产品
        ifp_file = os.path.join(self.data_dir, "IFP display.txt")
        if os.path.exists(ifp_file):
            products.extend(self._load_product_file(ifp_file, "IFP"))
        
        # 加载 JSON 产品数据
        for json_file in ["led_products.json", "lcd_products.json", "ifp_products.json"]:
            json_path = os.path.join(self.data_dir, json_file)
            if os.path.exists(json_path):
                products.extend(self._load_product_json(json_path))
        
        logger.info(f"Loaded {len(products)} product knowledge items")
        return products
    
    def _load_product_file(self, file_path: str, product_type: str) -> List[KnowledgeItem]:
        """加载产品文件"""
        items = []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 按产品分割（假设用 === 分割）
            sections = content.split("===")
            
            for i, section in enumerate(sections):
                section = section.strip()
                if not section:
                    continue
                
                items.append(KnowledgeItem(
                    id=f"{product_type}_{i}",
                    source=KnowledgeSource.PRODUCT,
                    content=section,
                    metadata={
                        "type": product_type,
                        "file": os.path.basename(file_path),
                    }
                ))
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
        
        return items
    
    def _load_product_json(self, json_path: str) -> List[KnowledgeItem]:
        """加载 JSON 产品数据"""
        items = []
        
        try:
            import json
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            products = data if isinstance(data, list) else data.get("products", [])
            
            for product in products:
                model = product.get("model", "unknown")
                content = self._format_product_text(product)
                
                items.append(KnowledgeItem(
                    id=model,
                    source=KnowledgeSource.PRODUCT,
                    content=content,
                    metadata={
                        "type": product.get("type", "LED"),
                        "model": model,
                        "pixel_pitch": product.get("pixel_pitch"),
                        "brightness": product.get("brightness"),
                    }
                ))
        except Exception as e:
            logger.warning(f"Failed to load {json_path}: {e}")
        
        return items
    
    def _format_product_text(self, product: dict) -> str:
        """将产品字典格式化为文本"""
        lines = []
        
        if "model" in product:
            lines.append(f"型号: {product['model']}")
        if "pixel_pitch" in product:
            lines.append(f"点间距: {product['pixel_pitch']}mm")
        if "brightness" in product:
            lines.append(f"亮度: {product['brightness']}")
        if "refresh_rate" in product:
            lines.append(f"刷新率: {product['refresh_rate']}Hz")
        if "size" in product:
            lines.append(f"尺寸: {product['size']}")
        if "environment" in product:
            lines.append(f"适用环境: {product['environment']}")
        if "features" in product:
            lines.append(f"特点: {product['features']}")
        
        return "\n".join(lines)


class SalesKnowledgeLoader:
    """
    销售知识加载器
    
    从销售话术文件加载销售策略知识。
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = os.path.join(data_dir, "sales_scripts")
    
    def load_sales_knowledge(self) -> List[KnowledgeItem]:
        """加载所有销售知识"""
        knowledge = []
        
        # 加载各类销售话术
        files = {
            "greeting.txt": "greeting",
            "industry_scenarios.txt": "industry",
            "objection_handling.txt": "objection",
            "requirement_questions.txt": "requirement",
        }
        
        for filename, category in files.items():
            file_path = os.path.join(self.data_dir, filename)
            if os.path.exists(file_path):
                items = self._load_file(file_path, category)
                knowledge.extend(items)
        
        logger.info(f"Loaded {len(knowledge)} sales knowledge items")
        return knowledge
    
    def _load_file(self, file_path: str, category: str) -> List[KnowledgeItem]:
        """加载销售话术文件"""
        items = []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 按段落分割
            paragraphs = content.split("\n\n")
            
            for i, para in enumerate(paragraphs):
                para = para.strip()
                if not para:
                    continue
                
                items.append(KnowledgeItem(
                    id=f"sales_{category}_{i}",
                    source=KnowledgeSource.SALES,
                    content=para,
                    metadata={"category": category}
                ))
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
        
        return items


# 全局分离器实例
knowledge_separator = KnowledgeSeparator()
