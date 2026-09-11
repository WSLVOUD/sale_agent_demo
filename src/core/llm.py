"""
LLM 统一调用模块
提供 LLM 实例的获取，包含超时和重试支持。
"""
from __future__ import annotations

import time
import logging
from functools import wraps
from typing import Callable, TypeVar

from langchain_openai import ChatOpenAI

from src.config import config

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _with_retry(func: Callable[..., T]) -> Callable[..., T]:
    """为 LLM 调用添加重试装饰器（指数退避）。"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        max_retries = getattr(config, "LLM_MAX_RETRIES", 2)
        timeout_secs = getattr(config, "LLM_TIMEOUT_SECS", 30)

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    wait = 2 ** attempt  # 1s, 2s, 4s...
                    logger.warning(
                        "LLM call failed (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1, max_retries + 1, wait, exc,
                    )
                    time.sleep(wait)
                else:
                    logger.error("LLM call exhausted retries: %s", exc)

        # 所有重试均失败，抛出最后一个错误
        raise last_error

    return wrapper


class RetryableChatOpenAI(ChatOpenAI):
    """带重试能力的 ChatOpenAI。超时走基类 timeout，避免写入未声明字段。"""

    def invoke(self, input, config=None, **kwargs):
        return _with_retry(super().invoke)(input, config=config, **kwargs)


def get_llm(
    model_name: str | None = None,
    temperature: float | None = None,
    use_retry: bool = True,
) -> ChatOpenAI:
    """
    获取配置好的 LLM 实例。

    具备：
    - 超时控制（LLM_TIMEOUT_SECS）
    - 自动重试（LLM_MAX_RETRIES，指数退避）
    """
    max_retries = getattr(config, "LLM_MAX_RETRIES", 2)
    timeout_secs = getattr(config, "LLM_TIMEOUT_SECS", 30)
    kwargs = {
        "model": model_name or config.MODEL_NAME,
        "temperature": temperature if temperature is not None else 0.7,
        "openai_api_key": config.DEEPSEEK_API_KEY,
        "openai_api_base": "https://api.deepseek.com",
        "timeout": timeout_secs,
        "max_retries": 0 if use_retry else max_retries,
    }
    if use_retry:
        return RetryableChatOpenAI(**kwargs)
    return ChatOpenAI(**kwargs)
