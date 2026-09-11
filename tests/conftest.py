"""
Pytest 配置文件和测试基础设施

运行方式：
    pytest tests/ -v
    pytest tests/ -v --cov=src
    pytest tests/ -v -k "test_name"

测试组织：
    tests/
    ├── conftest.py          # 共享 fixtures
    ├── test_memory.py       # 记忆模块测试
    ├── test_retrieval.py    # 检索模块测试
    ├── test_query_understanding.py  # Query Understanding 测试
    ├── test_orchestrator.py # 编排器测试
    └── test_api.py          # API 测试
"""
import pytest
import sys
import os


# Pytest-asyncio 配置：让所有 async test 自动运行
# asyncio_mode = "auto" 让 @pytest.mark.asyncio 标记的测试自动被收集
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asyncio: mark function as async coroutine"
    )
    # Configure pytest-asyncio mode
    try:
        import pytest_asyncio
        config.option.asyncio_mode = "auto"
    except ImportError:
        pass

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 设置测试环境变量
os.environ.setdefault("LED_API_KEY", "test-api-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "")


@pytest.fixture(scope="session")
def project_root_path():
    """项目根目录"""
    return project_root


@pytest.fixture(scope="session")
def data_dir(project_root_path):
    """数据目录"""
    return os.path.join(project_root_path, "data")


@pytest.fixture(scope="function")
def clean_memory():
    """每个测试后清理记忆"""
    yield
    try:
        from src.memory.store import memory
        memory.clear_all()
    except ImportError:
        pass


@pytest.fixture(scope="function")
def clean_enhanced_memory():
    """每个测试后清理增强记忆"""
    yield
    try:
        from src.memory.enhanced import enhanced_memory
        enhanced_memory.clear_all()
    except ImportError:
        pass
