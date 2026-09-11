"""
测试增强记忆模块
Phase 4: pytest 测试工程化

测试覆盖：
- EnhancedMemoryStore 基本操作
- 结构化客户画像更新
- 短期记忆管理
- 对话摘要生成
- 上下文查询
"""
import pytest
import time
from src.memory.enhanced import (
    EnhancedMemoryStore,
    CustomerProfile,
    ConversationSummary,
    MAX_SHORT_TERM_MESSAGES,
    MAX_MESSAGES_BEFORE_SUMMARY,
)


class TestCustomerProfile:
    """测试客户画像"""
    
    def test_create_empty_profile(self):
        """创建空画像"""
        profile = CustomerProfile()
        assert profile.budget is None
        assert profile.scene is None
        assert profile.interested_products == []
        assert profile.interaction_count == 0
    
    def test_create_profile_with_data(self):
        """创建带数据的画像"""
        profile = CustomerProfile(
            budget="10-20万",
            scene="会议室",
            display_type="LED",
            environment="indoor",
            is_rental=False,
        )
        assert profile.budget == "10-20万"
        assert profile.display_type == "LED"
        assert profile.is_rental is False
    
    def test_to_dict(self):
        """转换为字典"""
        profile = CustomerProfile(
            budget="5万",
            interested_products=["TW3", "TW2.5"],
        )
        data = profile.to_dict()
        assert data["budget"] == "5万"
        assert "TW3" in data["interested_products"]
    
    def test_from_dict(self):
        """从字典创建"""
        data = {
            "budget": "20万",
            "scene": "展厅",
            "interested_products": ["TW3"],
            "rejected_products": [],
            "pending_questions": [],
            "stated_objections": [],
            "first_contact": None,
            "last_update": None,
            "interaction_count": 0,
        }
        profile = CustomerProfile.from_dict(data)
        assert profile.budget == "20万"
        assert profile.scene == "展厅"


class TestEnhancedMemoryStore:
    """测试增强记忆存储"""
    
    def test_add_message(self):
        """测试添加消息"""
        store = EnhancedMemoryStore()
        session_id = "test-session-1"
        
        store.add_message(session_id, "user", "你好，我想了解一下LED屏幕")
        store.add_message(session_id, "assistant", "您好，请问有什么可以帮您？")
        
        history = store.get_history(session_id)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert "LED屏幕" in history[0]["content"]
    
    def test_short_term_memory_fifo(self):
        """测试短期记忆 FIFO"""
        store = EnhancedMemoryStore()
        session_id = "test-session-fifo"
        
        # 添加超过限制的消息
        for i in range(MAX_SHORT_TERM_MESSAGES + 5):
            store.add_message(session_id, "user", f"消息 {i}")
        
        short_term = store.get_short_term(session_id)
        assert len(short_term) == MAX_SHORT_TERM_MESSAGES
        # 应该是最新的消息
        assert short_term[-1]["content"] == f"消息 {MAX_SHORT_TERM_MESSAGES + 4}"
    
    def test_update_structured_profile(self):
        """测试更新结构化画像"""
        store = EnhancedMemoryStore()
        session_id = "test-session-profile"
        
        store.update_structured_profile(
            session_id,
            budget="15万",
            scene="会议室",
            display_type="LED",
            environment="indoor",
            is_rental=False,
        )
        
        profile = store.get_structured_profile(session_id)
        assert profile is not None
        assert profile.budget == "15万"
        assert profile.scene == "会议室"
        assert profile.display_type == "LED"
        assert profile.environment == "indoor"
    
    def test_add_interested_product(self):
        """测试添加感兴趣产品"""
        store = EnhancedMemoryStore()
        session_id = "test-session-product"
        
        store.add_interested_product(session_id, "TW3")
        store.add_interested_product(session_id, "TW2.5")
        store.add_interested_product(session_id, "TW3")  # 重复
        
        profile = store.get_structured_profile(session_id)
        assert len(profile.interested_products) == 2
        assert "TW3" in profile.interested_products
        assert "TW2.5" in profile.interested_products
    
    def test_add_rejected_product(self):
        """测试添加拒绝产品"""
        store = EnhancedMemoryStore()
        session_id = "test-session-rejected"
        
        store.add_rejected_product(session_id, "TW3", "太贵了")
        
        profile = store.get_structured_profile(session_id)
        assert "TW3" in profile.rejected_products
        assert "TW3: 太贵了" in profile.stated_objections
    
    def test_pending_questions(self):
        """测试待确认问题"""
        store = EnhancedMemoryStore()
        session_id = "test-session-pending"
        
        store.add_pending_question(session_id, "亮度要求是多少？")
        store.add_pending_question(session_id, "预算范围？")
        
        profile = store.get_structured_profile(session_id)
        assert len(profile.pending_questions) == 2
        
        # 解决问题
        store.resolve_pending_question(session_id, "亮度要求是多少？")
        assert len(profile.pending_questions) == 1
    
    def test_get_context(self):
        """测试获取完整上下文"""
        store = EnhancedMemoryStore()
        session_id = "test-session-context"
        
        # 添加消息
        store.add_message(session_id, "user", "我要室内LED屏幕")
        store.add_message(session_id, "assistant", "好的，室内LED有什么要求？")
        
        # 更新画像
        store.update_structured_profile(
            session_id,
            environment="indoor",
            display_type="LED",
        )
        
        context = store.get_context(session_id)
        
        assert "structured_profile" in context
        assert "short_term" in context
        assert context["structured_profile"]["environment"] == "indoor"
    
    def test_get_prompt_context(self):
        """测试获取 Prompt 上下文"""
        store = EnhancedMemoryStore()
        session_id = "test-session-prompt"
        
        store.update_structured_profile(
            session_id,
            budget="10万",
            display_type="LED",
        )
        store.add_message(session_id, "user", "你好")
        
        prompt_context = store.get_prompt_context(session_id)
        
        assert "【客户画像】" in prompt_context
        assert "10万" in prompt_context
        assert "LED" in prompt_context
        assert "【最近对话】" in prompt_context
    
    def test_simple_summary(self):
        """测试简单摘要生成"""
        store = EnhancedMemoryStore()
        session_id = "test-session-summary"
        
        # 添加足够多的消息
        for i in range(15):
            store.add_message(session_id, "user", f"我想了解LED产品{i}")
        
        summary = store.generate_summary(session_id)
        assert summary != ""
        
        # 验证摘要已保存
        saved_summary = store.get_summary(session_id)
        assert saved_summary is not None
        assert saved_summary.summary_text != ""
    
    def test_clear_session(self):
        """测试清除会话"""
        store = EnhancedMemoryStore()
        session_id = "test-session-clear"
        
        store.add_message(session_id, "user", "测试")
        assert store.get_size(session_id) == 1
        
        store.clear(session_id)
        assert store.get_size(session_id) == 0
    
    def test_needs_summary(self):
        """测试是否需要摘要"""
        store = EnhancedMemoryStore()
        session_id = "test-session-needs-summary"
        
        assert store.needs_summary(session_id) is False
        
        # 添加消息直到触发摘要
        for i in range(MAX_MESSAGES_BEFORE_SUMMARY):
            store.add_message(session_id, "user", f"消息 {i}")
        
        assert store.needs_summary(session_id) is True


class TestMemoryIntegration:
    """集成测试"""
    
    def test_full_conversation_flow(self):
        """完整对话流程"""
        store = EnhancedMemoryStore()
        session_id = "test-full-flow"
        
        # 初始对话
        store.add_message(session_id, "user", "您好，我想要一块室内LED屏")
        store.add_message(session_id, "assistant", "好的，请问预算和尺寸是多少？")
        
        # 更新结构化信息
        store.update_structured_profile(
            session_id,
            scene="会议室",
            display_type="LED",
            environment="indoor",
            budget="15万",
        )
        
        # 继续对话
        store.add_message(session_id, "user", "预算15万，尺寸3米左右")
        store.add_message(session_id, "assistant", "推荐 TW3，3.91mm 点间距，适合室内")
        
        # 记录感兴趣和拒绝的产品
        store.add_interested_product(session_id, "TW3")
        store.add_pending_question(session_id, "亮度是否够会议室使用？")
        
        # 验证最终状态
        profile = store.get_structured_profile(session_id)
        assert profile.budget == "15万"
        assert profile.scene == "会议室"
        assert "TW3" in profile.interested_products
        assert "亮度是否够会议室使用？" in profile.pending_questions
        
        # 验证上下文完整性
        context = store.get_context(session_id)
        assert context["structured_profile"]["budget"] == "15万"
        assert len(context["short_term"]) > 0
