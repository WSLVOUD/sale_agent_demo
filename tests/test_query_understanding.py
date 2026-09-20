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


class TestKeywordFalsePositives:
    """回归：英文关键词必须整词匹配，不能把名字 / 普通单词当成业务信号。

    实测 bug：客户 "My name is Ar Majeed Akbar ... smart class room in our
    university" 被判成 **restaurant**（"Akbar" 里含 "bar"），
    于是系统回了一句"Alright, a restaurant."。
    """

    def test_name_Akbar_is_not_restaurant(self):
        from src.rag.query_understanding import extract_slots

        message = (
            "My name is Ar Majeed Akbar from Pakistan, actually we are going for "
            "smart class room in our university so I need smart screen"
        )
        slots = extract_slots(message)
        assert slots.get("purpose") == "classroom"

    def test_class_room_two_words_is_classroom(self):
        from src.rag.query_understanding import extract_slots

        assert extract_slots("we need a smart class room screen").get("purpose") == "classroom"

    def test_different_is_not_rental(self):
        from src.rag.query_understanding import extract_slots

        assert "installation" not in extract_slots("I want a different product")
        assert extract_slots("we need it for rental events").get("installation") == "rental"

    def test_chinese_adjacent_english_keyword_still_matches(self):
        """中文紧挨英文关键词时也要能匹配（\\b 对中文无效）。"""
        from src.rag.query_understanding import extract_slots

        assert extract_slots("会议室用LCD").get("display_type") == "LCD"
        assert extract_slots("室内LED显示屏").get("display_type") == "LED"


class TestImperialUnits:
    """实测 bug：客户说 "about 100 feet viewing distance"，系统没解析英制单位，
    于是先回一句"已确认 100 英尺视距"、紧接着又追问观看距离（自相矛盾）。
    """

    @pytest.mark.parametrize("message,expected", [
        ("about 100 feet viewing distance", 30.48),
        ("100ft away", 30.48),
        ("the audience is 30 feet away", 9.144),
        ("viewing distance about 20 feet", 6.096),
        ("the viewers sit 15 ft from the screen", 4.572),
    ])
    def test_feet_converted_to_meters(self, message, expected):
        from src.rag.query_understanding import extract_slots

        value = extract_slots(message).get("viewing_distance_m")
        assert value is not None, message
        assert abs(value - expected) < 0.01, (message, value)

    def test_metric_still_works(self):
        from src.rag.query_understanding import extract_slots

        assert extract_slots("about 5 meters").get("viewing_distance_m") == 5.0
        assert extract_slots("大约5米").get("viewing_distance_m") == 5.0

    def test_area_not_mistaken_for_distance(self):
        from src.rag.query_understanding import extract_slots

        assert extract_slots("会议室 100 平方米").get("viewing_distance_m") is None

    def test_gate_stops_asking_distance_when_customer_volunteered_it(self):
        """客户自己先说出的需求必须被采纳，不能继续追问同一项。"""
        from src.models.requirement import RequirementProfile
        from src.rag.query_understanding import extract_slots
        from src.rag.readiness import check_recommendation_ready

        # 客户口径（2026-09-18）：硬性条件 = 尺寸 / P值 / 室内外 / 固装租赁
        message = (
            "indoor conference room, fixed installation, about 100 feet viewing distance, "
            "mainly for video, price matters more, 5m x 3m"
        )
        slots = extract_slots(message)
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        decision = check_recommendation_ready(profile)

        assert decision.ready is True, decision.missing
        assert "viewing_distance" not in decision.missing


class TestBareMeasurement:
    """实测 bug：客户回答 "129,2cm"（欧洲小数写法）时系统什么都没解析到，
    又把它判成 others 绕到自由问答，结果在 Gate 未通过时倒出一堆型号。
    """

    @pytest.mark.parametrize("message,expected_mm", [
        ("129,2cm", 1292.0),
        ("129.2cm", 1292.0),
        ("1292 mm", 1292.0),
        ("51 inch", 1295.4),
    ])
    def test_bare_measurement_becomes_size_hint(self, message, expected_mm):
        from src.rag.query_understanding import extract_slots

        slots = extract_slots(message)
        assert slots.get("screen_size_hint_mm") == pytest.approx(expected_mm), slots
        # 不能替客户猜方向
        assert slots.get("target_width_mm") is None
        assert slots.get("target_height_mm") is None

    def test_measurement_with_axis_is_the_size_itself(self):
        from src.rag.query_understanding import extract_slots

        slots = extract_slots("129,2cm wide")
        assert slots.get("target_width_mm") == pytest.approx(1292.0)
        assert slots.get("screen_size_hint_mm") is None

    @pytest.mark.parametrize("message,axis", [
        ("it is the width", "width"),
        ("width", "width"),
        ("it's the height", "height"),
        ("高", "height"),
        ("diagonal", "diagonal"),
        ("对角线", "diagonal"),
    ])
    def test_axis_answer(self, message, axis):
        from src.rag.query_understanding import extract_slots

        assert extract_slots(message).get("size_axis") == axis

    def test_metric_distance_is_not_a_size_hint(self):
        from src.rag.query_understanding import extract_slots

        assert extract_slots("about 5m").get("screen_size_hint_mm") is None
        assert extract_slots("about 5m").get("viewing_distance_m") == 5.0

    def test_gate_asks_axis_before_anything_else(self):
        from src.models.requirement import RequirementProfile
        from src.rag.query_understanding import extract_slots
        from src.rag.readiness import check_recommendation_ready

        slots = extract_slots("129,2cm")
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        decision = check_recommendation_ready(profile)

        assert decision.ready is False
        assert "size_axis" in decision.missing
        assert "129.2 cm" in (decision.next_question or "")


class TestAxisWordMeasurements:
    """实测 bug：客户答 "129,2cm us the length and 45xm is width"（两个尺寸 + 两个笔误），
    系统一个都没解析到，还把同一个方向问题又问了一遍。
    """

    @pytest.mark.parametrize("message", [
        "129,2cm us the length and 45xm is width",          # 实测日志原句
        "129,2cm is the length and 45cm is the width",
        "129.2cm length, 45cm height",
        "the width is 129.2cm and the height is 45cm",
        "长1.29米，宽0.45米",
        "宽度 1.29m，高度 0.45m",
    ])
    def test_two_dimensions_are_fully_captured(self, message):
        from src.rag.query_understanding import extract_slots

        slots = extract_slots(message)
        width = slots.get("target_width_mm")
        height = slots.get("target_height_mm")
        assert width is not None and height is not None, slots
        assert 1280 <= width <= 1300, slots
        assert 440 <= height <= 460, slots
        # 已经解析成宽高，就不该再留"待确认方向"的线索
        assert slots.get("screen_size_hint_mm") is None

    def test_width_only(self):
        from src.rag.query_understanding import extract_slots

        slots = extract_slots("45cm is the width")
        assert slots.get("target_width_mm") == pytest.approx(450)
        assert slots.get("target_height_mm") is None

    def test_cm_typo_xm(self):
        from src.rag.query_understanding import extract_slots

        assert extract_slots("45xm wide").get("target_width_mm") == pytest.approx(450)

    def test_dimension_is_not_read_as_distance(self):
        """尺寸不能被当成观看距离（"宽度 1.29m" 曾解析出视距 1.29 m）。"""
        from src.rag.query_understanding import extract_slots

        assert "viewing_distance_m" not in extract_slots("宽度 1.29m，高度 0.45m")
        assert "viewing_distance_m" not in extract_slots("129,2cm is the length")

    def test_decimal_distance_is_not_truncated(self):
        """回归：十进制视距不能被从小数点后面重新匹配（1.29 米 → 29 米）。"""
        from src.rag.query_understanding import extract_slots

        assert extract_slots("距离约1.29米").get("viewing_distance_m") == pytest.approx(1.29)
        assert extract_slots("5.5米视距").get("viewing_distance_m") == pytest.approx(5.5)
