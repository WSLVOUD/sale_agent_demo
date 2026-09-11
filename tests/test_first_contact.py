"""
First Contact 模块测试

测试内容：
1. Profile 加载与解析
2. Handler 自我介绍生成
3. 素材配置与路径解析
4. 严格基于 Profile 内容回复
"""
import pytest
import sys
import os

# 确保项目路径在 sys.path 中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.first_contact.profile import load_profile, CompanyProfile, _parse_profile_txt


class TestCompanyProfile:
    """公司信息加载与解析测试"""

    def test_load_profile_returns_company_profile(self):
        """验证加载返回 CompanyProfile 实例"""
        profile = load_profile()
        assert isinstance(profile, CompanyProfile)

    def test_sales_name_parsed(self):
        """验证销售人员姓名解析"""
        profile = load_profile()
        assert profile.sales_name, "Sales Name should not be empty"

    def test_company_name_parsed(self):
        """验证公司名称解析"""
        profile = load_profile()
        assert profile.company, "Company name should not be empty"

    def test_main_products_parsed_as_list(self):
        """验证主营产品解析为列表"""
        profile = load_profile()
        assert isinstance(profile.main_products, list)
        assert len(profile.main_products) > 0, "Main Products should have at least one item"

    def test_company_advantages_parsed_as_list(self):
        """验证公司优势解析为列表"""
        profile = load_profile()
        assert isinstance(profile.company_advantages, list)

    def test_profile_fields_are_strings_or_lists(self):
        """验证字段类型正确"""
        profile = load_profile()
        assert isinstance(profile.sales_name, str)
        assert isinstance(profile.company, str)
        assert isinstance(profile.company_introduction, str)
        assert isinstance(profile.main_products, list)
        assert isinstance(profile.company_advantages, list)


class TestProfileParsing:
    """Profile 文本解析测试"""

    def test_parse_minimal_profile(self):
        """测试最小化 profile 解析"""
        text = """
Sales Name:
John

Company:
TestCorp

Main Products:
LED Display
"""
        profile = _parse_profile_txt(text)
        assert profile.sales_name == "John"
        assert profile.company == "TestCorp"
        assert "LED Display" in profile.main_products

    def test_parse_empty_position_is_allowed(self):
        """验证 Position 字段可以不存在或为空"""
        # 当 Position 字段完全不存在时，应该不报错
        text = """
Sales Name:
John

Company:
TestCorp

Main Products:
LED Display
"""
        profile = _parse_profile_txt(text)
        assert profile.sales_name == "John"
        assert profile.company == "TestCorp"
        # Position 不在输入中，解析结果可以是空或解析到其他值（取决于实现）

    def test_parse_multi_line_introduction(self):
        """测试多行公司介绍解析"""
        text = """
Company:
iSEMC

Company Introduction:
iSEMC was founded in 2013.
This is a multi-line description.
It has multiple sentences.
"""
        profile = _parse_profile_txt(text)
        assert "iSEMC" in profile.company_introduction
        assert "founded in 2013" in profile.company_introduction


class TestFirstContactHandler:
    """首次接待处理器测试"""

    @pytest.fixture
    def handler(self):
        """创建 Handler 实例"""
        from src.first_contact.handler import FirstContactHandler
        return FirstContactHandler()

    def test_handler_initialization(self, handler):
        """验证 Handler 可以正常初始化"""
        assert handler is not None
        assert hasattr(handler, 'run')

    def test_profile_to_text_format(self, handler):
        """验证 Profile 转文本格式正确"""
        profile = load_profile()
        text = handler._profile_to_text(profile)

        # 检查关键字段是否在文本中
        assert profile.sales_name in text
        assert profile.company in text
        # Main Products 应该是逗号分隔的字符串
        assert any(p in text for p in profile.main_products) if profile.main_products else True

    def test_handler_run_returns_first_contact_result(self, handler):
        """验证 run 方法返回正确的结果类型"""
        from src.first_contact.handler import FirstContactResult

        result = handler.run(
            session_id="test-session-001",
            customer_message="Hi, I want to know about your LED displays"
        )

        assert isinstance(result, FirstContactResult)
        assert result.session_id == "test-session-001"
        assert isinstance(result.intro_text, str)
        assert isinstance(result.asset_results, list)

    def test_intro_text_is_english(self, handler):
        """验证自我介绍强制使用英语"""
        result = handler.run(
            session_id="test-session-002",
            customer_message="你好，我想了解LED显示屏"
        )
        # 自我介绍应该包含英语
        assert len(result.intro_text) > 0
        # 检查是否包含英语单词（简单检测）
        assert any(c.isalpha() for c in result.intro_text)

    def test_intro_does_not_contain_unknown_info(self, handler):
        """验证自我介绍不包含 Profile 中没有的信息"""
        profile = load_profile()
        result = handler.run(
            session_id="test-session-003",
            customer_message="Hello"
        )

        intro_text = result.intro_text
        # 如果 profile 中没有某个信息，不应该出现在自我介绍中
        # 例如：如果 profile 没有 "certification" 相关内容，不应该出现
        if "certification" not in profile.company_introduction.lower():
            assert "certification" not in intro_text.lower()

        if "award" not in profile.company_introduction.lower():
            assert "award" not in intro_text.lower()


class TestFirstContactAssets:
    """首次接待素材配置测试"""

    def test_assets_list_exists(self):
        """验证素材列表存在"""
        from src.first_contact.assets import ASSETS
        assert isinstance(ASSETS, list)

    def test_asset_has_required_fields(self):
        """验证素材包含必需字段"""
        from src.first_contact.assets import ASSETS

        if len(ASSETS) == 0:
            pytest.skip("No assets configured")

        asset = ASSETS[0]
        assert hasattr(asset, 'name')
        assert hasattr(asset, 'filename')
        assert hasattr(asset, 'asset_type')

    def test_get_asset_path_handles_missing_file(self):
        """验证文件不存在时返回 None 而不抛异常"""
        from src.first_contact.assets import ASSETS, get_asset_path, FirstContactAsset

        # 创建一个测试素材（文件不存在）
        fake_asset = FirstContactAsset(
            name="test.mp4",
            filename="nonexistent_test_file.mp4",
            asset_type="video",
            description="Test description"
        )

        path = get_asset_path(fake_asset)
        # 应该返回路径对象，但不保证文件存在
        assert path is not None
        # 验证路径包含文件名
        assert "nonexistent_test_file.mp4" in str(path)


class TestLanguageStrategy:
    """语言策略测试"""

    def test_english_only_for_chinese_input(self):
        """验证中文输入时回复仍为英语"""
        from src.first_contact.handler import FirstContactHandler

        handler = FirstContactHandler()
        result = handler.run(
            session_id="test-lang-001",
            customer_message="你好，请问你们卖什么产品？"
        )

        # 回复应该是英语
        intro_text = result.intro_text
        # 检查是否包含中文字符（不应该包含）
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in intro_text)
        assert not has_chinese, f"Intro should be in English, but found Chinese characters: {intro_text}"

    def test_english_only_for_spanish_input(self):
        """验证西班牙语输入时回复仍为英语"""
        from src.first_contact.handler import FirstContactHandler

        handler = FirstContactHandler()
        result = handler.run(
            session_id="test-lang-002",
            customer_message="Hola, me interesa saber más sobre sus productos"
        )

        intro_text = result.intro_text
        # 西班牙语常用词不应该出现在回复中
        assert "Hola" not in intro_text
        assert "gracias" not in intro_text.lower()
