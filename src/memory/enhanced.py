"""
增强记忆模块：Phase 4 Structured Memory 改进

三层记忆架构：
1. Short-term Memory: 最近 N 轮对话
2. Structured Memory: 客户需求结构化存储
3. Long-term Summary: 历史对话摘要

设计要点：
1. 关键客户需求使用结构化存储（预算、场景、室内/室外、租赁/固装、感兴趣产品等）
2. 长历史对话周期性 Summary
3. Summary 不覆盖结构化事实
4. 优先读取结构化客户信息
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

from src.config import config

logger = logging.getLogger(__name__)

# 配置
MAX_SHORT_TERM_MESSAGES = 20  # 短期记忆保留消息数
MAX_MESSAGES_BEFORE_SUMMARY = 30  # 超过此数量触发摘要
SUMMARY_TRIGGER_THRESHOLD = 10  # 摘要触发阈值（超过后截断并摘要）


@dataclass
class CustomerProfile:
    """
    客户画像（结构化记忆）
    
    存储客户的关键需求信息，优先级高于摘要。
    """
    # 基本信息
    budget: Optional[str] = None  # 预算
    scene: Optional[str] = None  # 应用场景
    
    # 产品偏好
    display_type: Optional[str] = None  # LED / LCD / IFP
    environment: Optional[str] = None  # indoor / outdoor
    is_rental: Optional[bool] = None  # 租赁 / 固装
    pixel_pitch_preference: Optional[str] = None  # 点间距偏好
    brightness_requirement: Optional[str] = None  # 亮度要求
    size_requirement: Optional[str] = None  # 尺寸要求
    
    # 交互状态
    interested_products: List[str] = field(default_factory=list)  # 感兴趣的产品
    rejected_products: List[str] = field(default_factory=list)  # 拒绝的产品
    pending_questions: List[str] = field(default_factory=list)  # 待确认问题
    stated_objections: List[str] = field(default_factory=list)  # 已提出异议
    
    # 元数据
    first_contact: Optional[float] = None  # 首次接触时间戳
    last_update: Optional[float] = None  # 最后更新时间戳
    interaction_count: int = 0  # 交互次数
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "CustomerProfile":
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ConversationSummary:
    """
    对话摘要（长期记忆）
    
    压缩历史对话，保留关键信息。
    不覆盖结构化事实（CustomerProfile）。
    """
    summary_text: str = ""  # 摘要文本
    key_decisions: List[str] = field(default_factory=list)  # 关键决策
    discussed_topics: List[str] = field(default_factory=list)  # 讨论过的话题
    unresolved_issues: List[str] = field(default_factory=list)  # 未解决问题
    created_at: float = field(default_factory=time.time)  # 创建时间
    updated_at: float = field(default_factory=time.time)  # 更新时间
    version: int = 1  # 摘要版本
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ConversationSummary":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class EnhancedMemoryStore:
    """
    增强记忆存储：三层记忆架构
    
    结构：
    {
        "session_id": {
            "short_term": [...],  # 最近消息
            "structured": CustomerProfile,  # 结构化客户信息
            "summary": ConversationSummary,  # 长期摘要
            "messages": [...],  # 完整消息历史（用于生成摘要）
        }
    }
    """
    
    def __init__(self):
        # session_id -> memory data
        self._sessions: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    
    # ── Short-term Memory ──────────────────────────────────────────────────────
    
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """添加短期消息（保持 FIFO）"""
        self._ensure(session_id)
        session = self._sessions[session_id]
        
        session["short_term"].append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })
        
        # 同步到完整消息历史
        session["messages"].append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })
        
        # FIFO 截断短期记忆
        if len(session["short_term"]) > MAX_SHORT_TERM_MESSAGES:
            session["short_term"] = session["short_term"][-MAX_SHORT_TERM_MESSAGES:]
        
        # 更新结构化记忆中最后交互时间
        profile = self._get_structured_profile(session_id)
        profile.last_update = time.time()
        profile.interaction_count += 1
    
    def get_short_term(self, session_id: str, limit: int | None = None) -> List[dict]:
        """获取最近 N 条短期记忆。

        Args:
            session_id: 会话 ID。
            limit: 最多返回 N 条；None 或 0 表示返回全部（直到 MAX_SHORT_TERM_MESSAGES 上限）。
        """
        session = self._sessions.get(session_id)
        if not session:
            return []
        if limit is None or limit == 0:
            return list(session["short_term"])
        return session["short_term"][-limit:]
    
    def get_history(self, session_id: str) -> List[dict]:
        """获取完整对话历史"""
        session = self._sessions.get(session_id)
        if not session:
            return []
        return list(session.get("messages", []))
    
    # ── Structured Memory ────────────────────────────────────────────────────
    
    def _get_structured_profile(self, session_id: str) -> CustomerProfile:
        """获取或创建结构化客户画像"""
        self._ensure(session_id)
        session = self._sessions[session_id]
        
        if "structured" not in session:
            session["structured"] = CustomerProfile(first_contact=time.time())
        
        return session["structured"]
    
    def update_structured_profile(self, session_id: str, **kwargs) -> None:
        """
        更新结构化客户画像
        
        支持的字段：
        - budget: 预算
        - scene: 场景
        - display_type: 屏幕类型
        - environment: 室内/室外
        - is_rental: 租赁/固装
        - pixel_pitch_preference: 点间距偏好
        - interested_products: 感兴趣产品（追加）
        - rejected_products: 拒绝产品（追加）
        - pending_questions: 待确认问题（追加）
        - stated_objections: 已提出异议（追加）
        """
        profile = self._get_structured_profile(session_id)
        
        # 直接设置字段
        direct_fields = {
            "budget", "scene", "display_type", "environment", "is_rental",
            "pixel_pitch_preference", "brightness_requirement", "size_requirement",
        }
        for field_name in direct_fields:
            if field_name in kwargs:
                setattr(profile, field_name, kwargs[field_name])
        
        # 追加列表字段
        list_fields = {
            "interested_products", "rejected_products",
            "pending_questions", "stated_objections",
        }
        for field_name in list_fields:
            if field_name in kwargs:
                current = getattr(profile, field_name, [])
                new_items = kwargs[field_name]
                if isinstance(new_items, list):
                    current.extend(new_items)
                else:
                    current.append(new_items)
                # 去重
                setattr(profile, field_name, list(dict.fromkeys(current)))
        
        profile.last_update = time.time()
        logger.debug(f"Updated structured profile for {session_id}: {kwargs}")
    
    def get_structured_profile(self, session_id: str) -> Optional[CustomerProfile]:
        """获取结构化客户画像"""
        session = self._sessions.get(session_id)
        if not session or "structured" not in session:
            return None
        return session["structured"]
    
    def get_profile_dict(self, session_id: str) -> dict:
        """获取结构化画像（字典格式）"""
        profile = self.get_structured_profile(session_id)
        if profile:
            return profile.to_dict()
        return {}
    
    def add_interested_product(self, session_id: str, product_id: str) -> None:
        """添加感兴趣产品"""
        profile = self._get_structured_profile(session_id)
        if product_id not in profile.interested_products:
            profile.interested_products.append(product_id)
            profile.last_update = time.time()
    
    def add_rejected_product(self, session_id: str, product_id: str, reason: str = "") -> None:
        """添加拒绝产品"""
        profile = self._get_structured_profile(session_id)
        if product_id not in profile.rejected_products:
            profile.rejected_products.append(product_id)
            if reason:
                profile.stated_objections.append(f"{product_id}: {reason}")
            profile.last_update = time.time()
    
    def add_pending_question(self, session_id: str, question: str) -> None:
        """添加待确认问题"""
        profile = self._get_structured_profile(session_id)
        if question not in profile.pending_questions:
            profile.pending_questions.append(question)
            profile.last_update = time.time()
    
    def resolve_pending_question(self, session_id: str, question: str) -> None:
        """解决待确认问题"""
        profile = self._get_structured_profile(session_id)
        if question in profile.pending_questions:
            profile.pending_questions.remove(question)
            profile.last_update = time.time()
    
    # ── Long-term Summary ────────────────────────────────────────────────────
    
    def _get_summary(self, session_id: str) -> ConversationSummary:
        """获取或创建对话摘要"""
        self._ensure(session_id)
        session = self._sessions[session_id]
        
        summary = session.get("summary")
        if summary is None:
            summary = ConversationSummary()
            session["summary"] = summary
        
        return summary
    
    def generate_summary(self, session_id: str, llm_client=None) -> str:
        """
        生成对话摘要
        
        Args:
            session_id: 会话 ID
            llm_client: LLM 客户端（可选，用于生成智能摘要）
        
        Returns:
            摘要文本
        """
        messages = self.get_history(session_id)
        
        if len(messages) < 5:
            return ""
        
        session = self._sessions[session_id]
        summary = self._get_summary(session_id)
        
        # 构建摘要输入
        conversation_text = "\n".join([
            f"{msg.get('role', 'user')}: {msg.get('content', '')[:200]}"
            for msg in messages[-SUMMARY_TRIGGER_THRESHOLD:]
        ])
        
        if llm_client:
            # 使用 LLM 生成智能摘要
            prompt = f"""请总结以下对话的要点，保留关键信息：

{conversation_text}

请用中文总结，包括：
1. 客户的主要需求和场景
2. 讨论过的产品或方案
3. 客户的顾虑或异议
4. 待解决的问题

摘要："""
            try:
                response = llm_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                )
                summary_text = response.choices[0].message.content
            except Exception as e:
                logger.warning(f"LLM summary failed: {e}")
                summary_text = self._simple_summary(messages)
        else:
            summary_text = self._simple_summary(messages)
        
        # 更新摘要
        summary.summary_text = summary_text
        summary.updated_at = time.time()
        summary.version += 1
        
        # 清空短期记忆（但保留结构化数据）
        session["short_term"] = []
        
        logger.info(f"Generated summary for {session_id}, version {summary.version}")
        return summary_text
    
    def _simple_summary(self, messages: List[dict]) -> str:
        """简单摘要（无需 LLM）"""
        if not messages:
            return ""
        
        # 提取关键信息
        topics = []
        products = []
        requirements = []
        
        for msg in messages:
            content = msg.get("content", "").lower()
            if "led" in content:
                topics.append("LED产品")
            if "lcd" in content:
                topics.append("LCD产品")
            if "租赁" in content or "rental" in content:
                requirements.append("租赁需求")
            if "室内" in content or "indoor" in content:
                requirements.append("室内场景")
            if "室外" in content or "outdoor" in content:
                requirements.append("室外场景")
        
        return f"讨论了 {len(topics)} 个产品类别，涉及 {len(requirements)} 个需求类型"
    
    def get_summary(self, session_id: str) -> Optional[ConversationSummary]:
        """获取对话摘要"""
        session = self._sessions.get(session_id)
        if not session or "summary" not in session:
            return None
        return session["summary"]
    
    # ── 统一查询接口 ────────────────────────────────────────────────────────
    
    def get_context(self, session_id: str) -> dict:
        """
        获取完整上下文（供 Agent 使用）
        
        返回包含：
        - structured_profile: 结构化客户信息
        - short_term: 短期记忆
        - summary: 对话摘要
        """
        return {
            "structured_profile": self.get_profile_dict(session_id),
            "short_term": self.get_short_term(session_id),
            "summary": self.get_summary(session_id).to_dict() if self.get_summary(session_id) else None,
        }
    
    def get_prompt_context(self, session_id: str) -> str:
        """
        获取用于 Prompt 的上下文字符串
        
        格式：
        【客户画像】
        - 预算：xxx
        - 场景：xxx
        - 产品偏好：xxx
        ...
        
        【最近对话】
        - user: xxx
        - assistant: xxx
        ...
        """
        parts = []
        
        # 结构化客户画像
        profile = self.get_structured_profile(session_id)
        if profile:
            parts.append("【客户画像】")
            if profile.budget:
                parts.append(f"- 预算：{profile.budget}")
            if profile.scene:
                parts.append(f"- 场景：{profile.scene}")
            if profile.display_type:
                parts.append(f"- 产品类型：{profile.display_type}")
            if profile.environment:
                parts.append(f"- 环境：{profile.environment}")
            if profile.is_rental is not None:
                parts.append(f"- 租赁/固装：{'租赁' if profile.is_rental else '固装'}")
            if profile.interested_products:
                parts.append(f"- 感兴趣产品：{', '.join(profile.interested_products)}")
            if profile.rejected_products:
                parts.append(f"- 拒绝产品：{', '.join(profile.rejected_products)}")
            if profile.pending_questions:
                parts.append(f"- 待确认：{', '.join(profile.pending_questions)}")
        
        # 对话摘要
        summary = self.get_summary(session_id)
        if summary and summary.summary_text:
            parts.append(f"\n【对话摘要】\n{summary.summary_text}")
        
        # 短期记忆
        short_term = self.get_short_term(session_id, limit=6)
        if short_term:
            parts.append("\n【最近对话】")
            for msg in short_term[-6:]:
                role = "用户" if msg.get("role") == "user" else "助手"
                content = msg.get("content", "")[:200]
                parts.append(f"- {role}：{content}")
        
        return "\n".join(parts) if parts else ""
    
    # ── 清理 ────────────────────────────────────────────────────────────────
    
    def clear(self, session_id: str) -> None:
        """清除指定会话"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("Cleared session: %s", session_id)
    
    def clear_all(self) -> None:
        """清除所有会话"""
        self._sessions.clear()
    
    # ── 内部辅助 ────────────────────────────────────────────────────────────
    
    def _ensure(self, session_id: str) -> None:
        """确保会话存在"""
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "short_term": [],
                "messages": [],
                "structured": CustomerProfile(first_contact=time.time()),
                "summary": None,
            }
    
    def get_size(self, session_id: str) -> int:
        """获取消息数量"""
        session = self._sessions.get(session_id)
        if not session:
            return 0
        return len(session.get("messages", []))
    
    def needs_summary(self, session_id: str) -> bool:
        """检查是否需要生成摘要"""
        return self.get_size(session_id) >= MAX_MESSAGES_BEFORE_SUMMARY


# 全局实例
enhanced_memory = EnhancedMemoryStore()


# 兼容旧接口：导出 memory 别名
def get_memory() -> EnhancedMemoryStore:
    """获取增强记忆存储"""
    return enhanced_memory
