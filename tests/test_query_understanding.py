"""
测试 Query Understanding 模块
Phase 4: pytest 测试工程化

测试覆盖：
- 参数提取
- Intent 分类
- 实体识别
"""
import pytest
from src.rag.parameter_inference import ParameterInference


class TestParameterInference:
    """测试参数推断"""
    
    @pytest.fixture
    def param_infer(self):
        """创建参数推断实例"""
        return ParameterInference()
    
    def test_extract_indoor_outdoor(self, param_infer):
        """测试室内/室外提取"""
        result = param_infer.extract_constraints("室内LED显示屏")
        assert result.get("indoor") is True
        assert result.get("outdoor") is False
        
        result = param_infer.extract_constraints("户外大屏")
        assert result.get("outdoor") is True
        assert result.get("indoor") is False
    
    def test_extract_display_type(self, param_infer):
        """测试屏幕类型提取"""
        result = param_infer.extract_constraints("LED屏幕推荐")
        assert result.get("display_type") == "LED"
        
        result = param_infer.extract_constraints("会议室用LCD")
        assert result.get("display_type") == "LCD"
    
    def test_extract_pixel_pitch(self, param_infer):
        """测试点间距提取"""
        result = param_infer.extract_constraints("P2.5的显示屏")
        assert "pixel_pitch" in result or "p2.5" in str(result).lower()
        
        result = param_infer.extract_constraints("点间距3mm")
        assert "pixel_pitch" in result or "3" in str(result)
    
    def test_extract_rental(self, param_infer):
        """测试租赁场景提取"""
        result = param_infer.extract_constraints("租赁用的LED屏")
        assert result.get("is_rental") is True
        
        result = param_infer.extract_constraints("固装显示屏")
        assert result.get("is_rental") is False
    
    def test_complex_query(self, param_infer):
        """测试复杂查询"""
        result = param_infer.extract_constraints(
            "我要租赁室内P3的LED显示屏，用于演唱会，预算20万"
        )
        
        assert result.get("is_rental") is True
        assert result.get("indoor") is True
        assert "pixel_pitch" in result or result.get("display_type") == "LED"
    
    def test_empty_query(self, param_infer):
        """测试空查询"""
        result = param_infer.extract_constraints("")
        assert result == {}


class TestIntentClassification:
    """测试 Intent 分类"""
    
    def test_greeting_intent(self):
        """测试问候意图"""
        from src.utils.ifp_intent import classify_intent
        
        intent = classify_intent("你好")
        assert intent in ["greeting", "general"]
    
    def test_product_recommendation_intent(self):
        """测试产品推荐意图"""
        from src.utils.ifp_intent import classify_intent
        
        intent = classify_intent("推荐一款室内LED屏")
        assert intent == "product_recommendation"
    
    def test_parameter_query_intent(self):
        """测试参数查询意图"""
        from src.utils.ifp_intent import classify_intent
        
        intent = classify_intent("TW3的亮度是多少")
        assert intent == "parameter_query"
    
    def test_objection_intent(self):
        """测试异议处理意图"""
        from src.utils.ifp_intent import classify_intent
        
        intent = classify_intent("你们的价格太贵了")
        assert intent == "objection"


class TestEntityExtraction:
    """测试实体提取"""
    
    def test_product_model_extraction(self):
        """测试产品型号提取"""
        from src.utils.ifp_intent import extract_entities
        
        entities = extract_entities("TW3和TW2.5哪个好")
        assert "TW3" in entities.get("product_models", [])
        assert "TW2.5" in entities.get("product_models", [])
    
    def test_budget_extraction(self):
        """测试预算提取"""
        from src.utils.ifp_intent import extract_entities
        
        entities = extract_entities("预算20万")
        assert "budget" in entities or "20万" in str(entities)
    
    def test_size_extraction(self):
        """测试尺寸提取"""
        from src.utils.ifp_intent import extract_entities
        
        entities = extract_entities("需要3米宽的屏")
        assert "size" in entities or "3米" in str(entities)
