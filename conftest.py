"""项目级 pytest 基础设施（放在 rootdir，供整棵 tests/ 树共用）。

测试只有一棵树（客户口径 2026-09-22：全量维持 500~600 条）：

    tests/    全量回归，约 590 条、约 1.5 分钟

运行方式：

    pytest -q                              # 全量
    pytest tests/test_vision_pipeline.py -q   # 单跑某个文件
"""
import os
import sys

import pytest


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
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 设置测试环境变量
os.environ.setdefault("LED_API_KEY", "test-api-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "")
# ── 测试里 LLM 一律"快速失败"（客户口径 2026-09-22：全量测试要瘦身提速）──
# 测试环境没有外网：默认重试 2 次会 sleep 1s + 2s，每个用例白等 3 秒。
# 关掉重试并把超时压到 3 秒 —— 失败路径的行为不变（仍然是"降级到结构化兜底"）。
os.environ.setdefault("LLM_MAX_RETRIES", "0")
os.environ.setdefault("LLM_TIMEOUT_SECS", "3")
os.environ.setdefault("LED_RAG_HISTORY_TOTAL_CHARS", "3000")


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


@pytest.fixture(autouse=True)
def _reset_response_llm_breaker():
    """每个用例后复位 ResponseGenerator 的"LLM 熔断"状态。

    否则"离线失败"的用例会把熔断器打开，后面的用例（用假 LLM 的那种）就再也
    调不到 LLM 了 —— 表现为测试顺序一变就红（实测：同一个测试单跑通过、
    跟在别的用例后面失败）。
    """
    yield
    try:
        from src.dialogue.response_generator import reset_llm_breaker

        reset_llm_breaker()
    except Exception:  # pragma: no cover - 防御式
        pass
