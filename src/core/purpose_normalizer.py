"""
Purpose Normalizer - 场景标准化器

职责：
1. 将自然语言场景表达映射到标准 purpose token
2. 维护多语言场景词表
3. 提供 canonical purpose 输出
4. 记录映射置信度
"""

import logging
from typing import Optional, Dict, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class CanonicalPurpose(str, Enum):
    """标准场景分类（Canonical Purpose Token）"""
    RETAIL = "retail"
    ADVERTISING = "advertising"
    CONFERENCE = "conference"
    CLASSROOM = "classroom"
    STADIUM = "stadium"
    CONCERT = "concert"
    STAGE = "stage"
    WEDDING = "wedding"
    CHURCH = "church"
    MUSEUM = "museum"
    SHOWROOM = "showroom"
    AIRPORT = "airport"
    BANK = "bank"
    HOTEL = "hotel"
    RESTAURANT = "restaurant"
    OFFICE = "office"
    HOSPITAL = "hospital"
    EXHIBITION = "exhibition"
    HALL = "hall"
    RENTAL = "rental"
    CONTROL_ROOM = "control_room"
    OTHER = "other"


# 多语言场景映射表
PURPOSE_MAPPING: Dict[str, CanonicalPurpose] = {
    # 英文
    "retail": CanonicalPurpose.RETAIL,
    "shopping": CanonicalPurpose.RETAIL,
    "shopping mall": CanonicalPurpose.RETAIL,
    "shopping center": CanonicalPurpose.RETAIL,
    "commercial": CanonicalPurpose.RETAIL,
    "commercial center": CanonicalPurpose.RETAIL,
    "commercial complex": CanonicalPurpose.RETAIL,
    "department store": CanonicalPurpose.RETAIL,
    "store": CanonicalPurpose.RETAIL,
    "shop": CanonicalPurpose.RETAIL,
    "supermarket": CanonicalPurpose.RETAIL,

    "advertising": CanonicalPurpose.ADVERTISING,
    "billboard": CanonicalPurpose.ADVERTISING,
    "signage": CanonicalPurpose.ADVERTISING,
    "digital signage": CanonicalPurpose.ADVERTISING,
    "facade": CanonicalPurpose.ADVERTISING,
    "building facade": CanonicalPurpose.ADVERTISING,
    "exterior wall": CanonicalPurpose.ADVERTISING,
    "outdoor advertising": CanonicalPurpose.ADVERTISING,
    "outdoor signage": CanonicalPurpose.ADVERTISING,

    "conference": CanonicalPurpose.CONFERENCE,
    "meeting": CanonicalPurpose.CONFERENCE,
    "meeting room": CanonicalPurpose.CONFERENCE,
    "boardroom": CanonicalPurpose.CONFERENCE,
    "conference room": CanonicalPurpose.CONFERENCE,
    "corporate": CanonicalPurpose.CONFERENCE,
    "business meeting": CanonicalPurpose.CONFERENCE,

    "classroom": CanonicalPurpose.CLASSROOM,
    "school": CanonicalPurpose.CLASSROOM,
    "training": CanonicalPurpose.CLASSROOM,
    "training room": CanonicalPurpose.CLASSROOM,
    "education": CanonicalPurpose.CLASSROOM,
    "educational": CanonicalPurpose.CLASSROOM,

    "stadium": CanonicalPurpose.STADIUM,
    "sports": CanonicalPurpose.STADIUM,
    "sports venue": CanonicalPurpose.STADIUM,
    "sports complex": CanonicalPurpose.STADIUM,
    "arena": CanonicalPurpose.STADIUM,
    "football": CanonicalPurpose.STADIUM,
    "basketball": CanonicalPurpose.STADIUM,
    "sports field": CanonicalPurpose.STADIUM,

    "concert": CanonicalPurpose.CONCERT,
    "concert hall": CanonicalPurpose.CONCERT,
    "live event": CanonicalPurpose.CONCERT,
    "music event": CanonicalPurpose.CONCERT,
    "music concert": CanonicalPurpose.CONCERT,

    "stage": CanonicalPurpose.STAGE,
    "performance": CanonicalPurpose.STAGE,
    "theater": CanonicalPurpose.STAGE,
    "theatre": CanonicalPurpose.STAGE,
    "live performance": CanonicalPurpose.STAGE,

    "wedding": CanonicalPurpose.WEDDING,
    "wedding event": CanonicalPurpose.WEDDING,
    "marriage": CanonicalPurpose.WEDDING,
    "wedding venue": CanonicalPurpose.WEDDING,

    "church": CanonicalPurpose.CHURCH,
    "worship": CanonicalPurpose.CHURCH,
    "religious": CanonicalPurpose.CHURCH,

    "museum": CanonicalPurpose.MUSEUM,
    "gallery": CanonicalPurpose.MUSEUM,
    "exhibition": CanonicalPurpose.EXHIBITION,
    "exhibition hall": CanonicalPurpose.EXHIBITION,
    "showroom": CanonicalPurpose.SHOWROOM,

    "airport": CanonicalPurpose.AIRPORT,
    "terminal": CanonicalPurpose.AIRPORT,
    "transportation": CanonicalPurpose.AIRPORT,
    "station": CanonicalPurpose.AIRPORT,

    "bank": CanonicalPurpose.BANK,
    "financial": CanonicalPurpose.BANK,

    "hotel": CanonicalPurpose.HOTEL,
    "hospitality": CanonicalPurpose.HOTEL,
    "lobby": CanonicalPurpose.HOTEL,

    "restaurant": CanonicalPurpose.RESTAURANT,
    "dining": CanonicalPurpose.RESTAURANT,
    "food service": CanonicalPurpose.RESTAURANT,
    "cafe": CanonicalPurpose.RESTAURANT,

    "office": CanonicalPurpose.OFFICE,
    "workspace": CanonicalPurpose.OFFICE,

    "hospital": CanonicalPurpose.HOSPITAL,
    "medical": CanonicalPurpose.HOSPITAL,
    "healthcare": CanonicalPurpose.HOSPITAL,

    "control room": CanonicalPurpose.CONTROL_ROOM,
    "command center": CanonicalPurpose.CONTROL_ROOM,
    "monitoring": CanonicalPurpose.CONTROL_ROOM,

    "hall": CanonicalPurpose.HALL,
    "event space": CanonicalPurpose.HALL,

    "rental": CanonicalPurpose.RENTAL,
    "temporary": CanonicalPurpose.RENTAL,
    "event": CanonicalPurpose.RENTAL,

    # 中文
    "零售": CanonicalPurpose.RETAIL,
    "商场": CanonicalPurpose.RETAIL,
    "商业": CanonicalPurpose.RETAIL,
    "商业复合体": CanonicalPurpose.RETAIL,
    "购物": CanonicalPurpose.RETAIL,
    "店铺": CanonicalPurpose.RETAIL,
    "超市": CanonicalPurpose.RETAIL,

    "广告": CanonicalPurpose.ADVERTISING,
    "户外广告": CanonicalPurpose.ADVERTISING,
    "幕墙": CanonicalPurpose.ADVERTISING,
    "外墙": CanonicalPurpose.ADVERTISING,
    "楼体": CanonicalPurpose.ADVERTISING,
    "广告牌": CanonicalPurpose.ADVERTISING,
    "标牌": CanonicalPurpose.ADVERTISING,

    "会议": CanonicalPurpose.CONFERENCE,
    "会议室": CanonicalPurpose.CONFERENCE,
    "会议中心": CanonicalPurpose.CONFERENCE,
    "企业": CanonicalPurpose.CONFERENCE,
    "商务": CanonicalPurpose.CONFERENCE,

    "教室": CanonicalPurpose.CLASSROOM,
    "学校": CanonicalPurpose.CLASSROOM,
    "培训": CanonicalPurpose.CLASSROOM,
    "教育": CanonicalPurpose.CLASSROOM,

    "体育": CanonicalPurpose.STADIUM,
    "体育场": CanonicalPurpose.STADIUM,
    "体育馆": CanonicalPurpose.STADIUM,
    "球场": CanonicalPurpose.STADIUM,
    "赛场": CanonicalPurpose.STADIUM,

    "演唱会": CanonicalPurpose.CONCERT,
    "音乐会": CanonicalPurpose.CONCERT,
    "现场表演": CanonicalPurpose.CONCERT,

    "舞台": CanonicalPurpose.STAGE,
    "演出": CanonicalPurpose.STAGE,
    "表演": CanonicalPurpose.STAGE,
    "剧场": CanonicalPurpose.STAGE,

    "婚礼": CanonicalPurpose.WEDDING,
    "婚宴": CanonicalPurpose.WEDDING,
    "婚庆": CanonicalPurpose.WEDDING,
    "结婚": CanonicalPurpose.WEDDING,

    "教堂": CanonicalPurpose.CHURCH,
    "礼拜": CanonicalPurpose.CHURCH,
    "宗教": CanonicalPurpose.CHURCH,

    "博物馆": CanonicalPurpose.MUSEUM,
    "美术馆": CanonicalPurpose.MUSEUM,
    "展厅": CanonicalPurpose.SHOWROOM,
    "展馆": CanonicalPurpose.SHOWROOM,
    "展览": CanonicalPurpose.EXHIBITION,

    "机场": CanonicalPurpose.AIRPORT,
    "车站": CanonicalPurpose.AIRPORT,
    "码头": CanonicalPurpose.AIRPORT,
    "地铁": CanonicalPurpose.AIRPORT,

    "银行": CanonicalPurpose.BANK,
    "金融": CanonicalPurpose.BANK,

    "酒店": CanonicalPurpose.HOTEL,
    "宾馆": CanonicalPurpose.HOTEL,
    "大堂": CanonicalPurpose.HOTEL,

    "餐厅": CanonicalPurpose.RESTAURANT,
    "酒吧": CanonicalPurpose.RESTAURANT,
    "咖啡厅": CanonicalPurpose.RESTAURANT,
    "餐饮": CanonicalPurpose.RESTAURANT,

    "办公室": CanonicalPurpose.OFFICE,
    "办公": CanonicalPurpose.OFFICE,

    "医院": CanonicalPurpose.HOSPITAL,
    "医疗": CanonicalPurpose.HOSPITAL,
    "诊所": CanonicalPurpose.HOSPITAL,

    "指挥中心": CanonicalPurpose.CONTROL_ROOM,
    "监控中心": CanonicalPurpose.CONTROL_ROOM,
    "控制室": CanonicalPurpose.CONTROL_ROOM,
    "中控室": CanonicalPurpose.CONTROL_ROOM,

    "大厅": CanonicalPurpose.HALL,
    "活动空间": CanonicalPurpose.HALL,

    "租赁": CanonicalPurpose.RENTAL,
    "临时": CanonicalPurpose.RENTAL,
    "活动": CanonicalPurpose.RENTAL,
}


class PurposeNormalizer:
    """场景标准化器"""

    @staticmethod
    def normalize(purpose_text: Optional[str]) -> Tuple[Optional[CanonicalPurpose], float]:
        """
        将自然语言场景表达规范化为标准 token。

        Args:
            purpose_text: 客户或系统识别的场景表达

        Returns:
            (canonical_purpose, confidence)
            - canonical_purpose: 标准 token 或 None
            - confidence: 0.0-1.0 置信度
        """
        if not purpose_text:
            return None, 0.0

        text = str(purpose_text).strip().lower()

        # 直接匹配
        if text in PURPOSE_MAPPING:
            return PURPOSE_MAPPING[text], 1.0

        # 模糊匹配（子串）
        for key, canonical in PURPOSE_MAPPING.items():
            if key in text or text in key:
                confidence = 0.8 if len(key) >= len(text) * 0.5 else 0.7
                logger.info(f"Fuzzy matched: '{purpose_text}' → {canonical.value} (confidence: {confidence})")
                return canonical, confidence

        # 无法匹配
        logger.warning(f"Could not normalize purpose: {purpose_text}")
        return None, 0.0

    @staticmethod
    def get_all_canonical() -> list[CanonicalPurpose]:
        """获取所有标准 purpose token"""
        return list(CanonicalPurpose)

    @staticmethod
    def get_english_search_keywords(purpose: CanonicalPurpose) -> str:
        """获取用于 RAG 检索的英文关键词"""
        keywords_map = {
            CanonicalPurpose.RETAIL: "retail shopping mall commercial",
            CanonicalPurpose.ADVERTISING: "advertising billboard signage facade",
            CanonicalPurpose.CONFERENCE: "conference meeting room boardroom",
            CanonicalPurpose.CLASSROOM: "classroom school training education",
            CanonicalPurpose.STADIUM: "stadium sports arena sports venue",
            CanonicalPurpose.CONCERT: "concert music live event",
            CanonicalPurpose.STAGE: "stage performance theater",
            CanonicalPurpose.WEDDING: "wedding event marriage venue",
            CanonicalPurpose.CHURCH: "church worship religious",
            CanonicalPurpose.MUSEUM: "museum gallery exhibition",
            CanonicalPurpose.SHOWROOM: "showroom exhibition display",
            CanonicalPurpose.AIRPORT: "airport terminal station transportation",
            CanonicalPurpose.BANK: "bank financial",
            CanonicalPurpose.HOTEL: "hotel hospitality lobby",
            CanonicalPurpose.RESTAURANT: "restaurant dining cafe",
            CanonicalPurpose.OFFICE: "office workspace",
            CanonicalPurpose.HOSPITAL: "hospital medical healthcare",
            CanonicalPurpose.EXHIBITION: "exhibition trade show expo",
            CanonicalPurpose.CONTROL_ROOM: "control room command center monitoring",
            CanonicalPurpose.HALL: "hall event space",
            CanonicalPurpose.RENTAL: "rental temporary event",
        }
        return keywords_map.get(purpose, purpose.value)
