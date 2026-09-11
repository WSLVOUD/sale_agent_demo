"""
Langfuse Observability Integration
Phase 4: 可观测性

提供完整的 Trace 追踪：
- 请求 → Intent → Query Understanding → Memory → Structured Filter → 
  Retrieval → Rerank → Generation → Reflection → Response

记录：
- 每次 LLM 调用
- Token 消耗
- 延迟
- TTFT
- Retrieval 耗时
- Rerank 耗时
- Reflection 轮数
- 错误信息
- 最终响应
"""
from __future__ import annotations

import os
import time
import logging
from typing import Any, Optional
from contextlib import contextmanager
from dataclasses import dataclass, field

from src.config import config

logger = logging.getLogger(__name__)

# Langfuse 客户端（延迟初始化）
_langfuse_client = None
_langfuse_initialized = False


@dataclass
class TraceMetrics:
    """Trace 指标收集器"""
    trace_id: str = ""
    request_start: float = 0
    ttft: Optional[float] = None  # Time to First Token
    total_latency: Optional[float] = None
    
    # LLM 调用统计
    llm_call_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    
    # 组件耗时
    intent_time: float = 0
    query_understanding_time: float = 0
    memory_time: float = 0
    structured_filter_time: float = 0
    retrieval_time: float = 0
    rerank_time: float = 0
    generation_time: float = 0
    reflection_time: float = 0
    
    # 业务指标
    reflection_rounds: int = 0
    retrieval_results_count: int = 0
    route: str = ""  # "fast" | "agent" | "fallback"
    complexity: str = ""
    
    # 错误追踪
    error_message: Optional[str] = None
    fallback_triggered: bool = False
    
    # 额外信息
    extra: dict = field(default_factory=dict)


def _init_langfuse():
    """延迟初始化 Langfuse 客户端"""
    global _langfuse_client, _langfuse_initialized
    
    if _langfuse_initialized:
        return _langfuse_client
    
    _langfuse_initialized = True
    
    # 检查是否配置了 Langfuse
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    if not public_key or not secret_key:
        logger.info("Langfuse not configured (missing LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY)")
        return None
    
    try:
        from langfuse import Langfuse
        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        logger.info("Langfuse initialized successfully")
        return _langfuse_client
    except ImportError:
        logger.warning("langfuse package not installed. Run: pip install langfuse")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize Langfuse: {e}")
        return None


def get_langfuse():
    """获取 Langfuse 客户端"""
    return _init_langfuse()


class LEDTracer:
    """
    LED RAG System 追踪器
    
    使用方式：
    
    ```python
    tracer = LEDTracer()
    
    # 开始追踪
    with tracer.trace("chat_session", session_id="xxx") as t:
        t.set_tag("route", "agent")
        
        # 各阶段追踪
        with t.span("query_understanding"):
            result = understand_query(query)
            t.record("entities", result["entities"])
        
        with t.span("retrieval"):
            products = retrieve(query)
            t.record("count", len(products))
        
        # LLM 调用追踪
        with t.llm_call("generation"):
            response = llm.generate(prompt)
            t.record_tokens(input=100, output=200)
    
    # 获取指标
    metrics = t.metrics
    ```
    """
    
    def __init__(self):
        self._client = get_langfuse()
        self._generation = None
        self._current_span = None
        
    @contextmanager
    def trace(self, name: str, session_id: Optional[str] = None, user_id: Optional[str] = None, metadata: Optional[dict] = None):
        """开始一个新的 Trace"""
        metrics = TraceMetrics()
        metrics.request_start = time.time()
        
        generation = None
        if self._client:
            try:
                generation = self._client.trace(
                    name=name,
                    session_id=session_id,
                    user_id=user_id,
                    metadata=metadata or {},
                )
            except Exception as e:
                logger.warning(f"Failed to create Langfuse trace: {e}")
        
        try:
            yield metrics
        finally:
            metrics.total_latency = time.time() - metrics.request_start
            # Ensure positive latency for sub-millisecond measurements (Python clock resolution)
            if metrics.total_latency == 0:
                metrics.total_latency = 1e-6  # ~1 microsecond floor
            
            # 记录最终指标到 Langfuse
            if generation:
                try:
                    generation.update(
                        metadata={
                            "total_latency_ms": metrics.total_latency * 1000,
                            "llm_call_count": metrics.llm_call_count,
                            "total_tokens": metrics.total_tokens,
                            "route": metrics.route,
                            "complexity": metrics.complexity,
                            "reflection_rounds": metrics.reflection_rounds,
                            "error": metrics.error_message,
                            **metrics.extra,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to update Langfuse trace: {e}")
    
    @contextmanager
    def span(self, name: str, metadata: Optional[dict] = None):
        """记录一个 Span（操作阶段）"""
        span = None
        start_time = time.time()
        
        if self._client:
            try:
                from langfuse import observation
                span = observation.observe(name=name)(lambda: None)
                span.__enter__()
            except Exception:
                pass
        
        try:
            yield span
        finally:
            if span:
                try:
                    span.__exit__(None, None, None)
                except Exception:
                    pass
        
        return span
    
    @contextmanager
    def llm_call(self, name: str, model: Optional[str] = None):
        """记录 LLM 调用"""
        start_time = time.time()
        input_tokens = 0
        output_tokens = 0
        
        if self._client:
            try:
                from langfuse import observation
                gen = observation.observe(name=name)(lambda: None)
                gen.__enter__()
            except Exception:
                gen = None
        else:
            gen = None
        
        try:
            yield gen
        finally:
            latency = time.time() - start_time
            
            if gen:
                try:
                    gen.__exit__(None, None, None)
                except Exception:
                    pass
    
    def record_token_usage(self, metrics: TraceMetrics, input_tokens: int, output_tokens: int):
        """记录 Token 使用"""
        metrics.llm_call_count += 1
        metrics.total_input_tokens += input_tokens
        metrics.total_output_tokens += output_tokens
        metrics.total_tokens += input_tokens + output_tokens
    
    def record_ttft(self, metrics: TraceMetrics):
        """记录首 Token 时间"""
        if metrics.ttft is None:
            metrics.ttft = (time.time() - metrics.request_start) * 1000  # ms
    
    def record_error(self, metrics: TraceMetrics, error: str):
        """记录错误"""
        metrics.error_message = error
    
    def record_fallback(self, metrics: TraceMetrics):
        """记录降级触发"""
        metrics.fallback_triggered = True


class MetricsCollector:
    """
    周维度指标收集器
    
    收集并聚合：
    - 请求量
    - 平均延迟
    - Token 成本
    - LLM 调用次数
    - 错误率
    - 人工接管率
    """
    
    def __init__(self):
        self._daily_stats: dict[str, DailyStats] = {}
    
    def record_request(
        self,
        latency_ms: float,
        tokens: int,
        llm_calls: int,
        is_error: bool = False,
        is_fallback: bool = False,
        route: str = "",
    ):
        """记录一次请求"""
        today = time.strftime("%Y-%m-%d")
        
        if today not in self._daily_stats:
            self._daily_stats[today] = DailyStats(date=today)
        
        stats = self._daily_stats[today]
        stats.request_count += 1
        stats.total_latency_ms += latency_ms
        stats.total_tokens += tokens
        stats.total_llm_calls += llm_calls
        
        if is_error:
            stats.error_count += 1
        if is_fallback:
            stats.fallback_count += 1
        if route:
            stats.route_counts[route] = stats.route_counts.get(route, 0) + 1
    
    def get_weekly_report(self) -> dict:
        """生成周报告"""
        # 聚合最近 7 天
        today = time.time()
        week_stats = {
            "request_count": 0,
            "total_latency_ms": 0,
            "total_tokens": 0,
            "total_llm_calls": 0,
            "error_count": 0,
            "fallback_count": 0,
            "route_counts": {},
        }
        
        for date_str, stats in self._daily_stats.items():
            week_stats["request_count"] += stats.request_count
            week_stats["total_latency_ms"] += stats.total_latency_ms
            week_stats["total_tokens"] += stats.total_tokens
            week_stats["total_llm_calls"] += stats.total_llm_calls
            week_stats["error_count"] += stats.error_count
            week_stats["fallback_count"] += stats.fallback_count
            
            for route, count in stats.route_counts.items():
                week_stats["route_counts"][route] = week_stats["route_counts"].get(route, 0) + count
        
        # 计算平均值
        request_count = week_stats["request_count"]
        if request_count > 0:
            week_stats["avg_latency_ms"] = week_stats["total_latency_ms"] / request_count
            week_stats["avg_tokens_per_request"] = week_stats["total_tokens"] / request_count
            week_stats["avg_llm_calls_per_request"] = week_stats["total_llm_calls"] / request_count
            week_stats["error_rate"] = week_stats["error_count"] / request_count
            week_stats["fallback_rate"] = week_stats["fallback_count"] / request_count
        else:
            week_stats["avg_latency_ms"] = 0
            week_stats["avg_tokens_per_request"] = 0
            week_stats["avg_llm_calls_per_request"] = 0
            week_stats["error_rate"] = 0
            week_stats["fallback_rate"] = 0
        
        return week_stats


@dataclass
class DailyStats:
    """每日统计"""
    date: str
    request_count: int = 0
    total_latency_ms: float = 0
    total_tokens: int = 0
    total_llm_calls: int = 0
    error_count: int = 0
    fallback_count: int = 0
    route_counts: dict = field(default_factory=dict)


# 全局追踪器实例
tracer = LEDTracer()
metrics_collector = MetricsCollector()
