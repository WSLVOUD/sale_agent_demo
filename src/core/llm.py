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


def _tracked_invoke(instance, base_invoke, input, config=None, **kwargs):
    """v2.5++++（计划 §15）：所有 LLM 调用统一记账（每次调用都算一次）。

    实测问题：一轮里打了多次 DeepSeek，日志却 `llm_calls=0` —— 只有手工 ++ 的
    那一处被统计。这里把统计放到**唯一调用入口**，Sales 图内部的调用也能算上。
    """
    from src.observability.llm_tracker import get_llm_tracker, track_usage

    tracker = get_llm_tracker()
    model_name = str(getattr(instance, "model_name", "") or getattr(instance, "model", "") or "")
    with tracker.track(model=model_name) as record:
        response = base_invoke(input, config=config, **kwargs)
        track_usage(record, response)
        return response


class RetryableChatOpenAI(ChatOpenAI):
    """带重试能力的 ChatOpenAI。超时走基类 timeout，避免写入未声明字段。"""

    def invoke(self, input, config=None, **kwargs):
        return _tracked_invoke(self, _with_retry(super().invoke), input, config=config, **kwargs)


class TrackedChatOpenAI(ChatOpenAI):
    """不带重试，但同样记账。"""

    def invoke(self, input, config=None, **kwargs):
        return _tracked_invoke(self, super().invoke, input, config=config, **kwargs)


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
    return TrackedChatOpenAI(**kwargs)
