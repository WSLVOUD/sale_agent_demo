"""
测试可观测性模块
Phase 4: pytest 测试工程化

测试覆盖：
- LEDTracer 追踪器
- TraceMetrics 指标收集
- MetricsCollector 周维度统计
"""
import pytest
import time
from src.observability import (
    LEDTracer,
    TraceMetrics,
    MetricsCollector,
)


class TestTraceMetrics:
    """测试 Trace 指标"""
    
    def test_create_metrics(self):
        """创建指标"""
        metrics = TraceMetrics()
        assert metrics.llm_call_count == 0
        assert metrics.total_tokens == 0
        assert metrics.ttft is None
    
    def test_record_tokens(self):
        """记录 Token"""
        metrics = TraceMetrics()
        metrics.total_input_tokens = 100
        metrics.total_output_tokens = 200
        metrics.total_tokens = 300
        metrics.llm_call_count = 1
        
        assert metrics.total_tokens == 300
        assert metrics.llm_call_count == 1
    
    def test_record_ttft(self):
        """记录首 Token 时间"""
        metrics = TraceMetrics()
        metrics.request_start = time.time()
        
        time.sleep(0.01)  # 模拟延迟
        
        # 模拟首 Token 到达
        metrics.ttft = (time.time() - metrics.request_start) * 1000
        
        assert metrics.ttft is not None
        assert metrics.ttft > 0


class TestMetricsCollector:
    """测试指标收集器"""
    
    def test_record_request(self):
        """记录请求"""
        collector = MetricsCollector()
        
        collector.record_request(
            latency_ms=100,
            tokens=500,
            llm_calls=2,
            is_error=False,
            is_fallback=False,
            route="agent",
        )
        
        collector.record_request(
            latency_ms=200,
            tokens=1000,
            llm_calls=3,
            is_error=True,
            route="fast",
        )
        
        report = collector.get_weekly_report()
        
        assert report["request_count"] == 2
        assert report["total_tokens"] == 1500
        assert report["total_llm_calls"] == 5
        assert report["error_count"] == 1
    
    def test_weekly_report_averages(self):
        """测试周报告平均值"""
        collector = MetricsCollector()
        
        # 记录多个请求
        for i in range(5):
            collector.record_request(
                latency_ms=100 + i * 10,
                tokens=500,
                llm_calls=2,
            )
        
        report = collector.get_weekly_report()
        
        assert report["request_count"] == 5
        assert report["avg_latency_ms"] == pytest.approx(120, rel=5)
        assert report["avg_tokens_per_request"] == 500
    
    def test_route_counts(self):
        """测试路由统计"""
        collector = MetricsCollector()
        
        collector.record_request(latency_ms=100, tokens=100, llm_calls=1, route="fast")
        collector.record_request(latency_ms=200, tokens=200, llm_calls=2, route="agent")
        collector.record_request(latency_ms=150, tokens=150, llm_calls=1, route="fast")
        
        report = collector.get_weekly_report()
        
        assert report["route_counts"]["fast"] == 2
        assert report["route_counts"]["agent"] == 1
    
    def test_error_rate(self):
        """测试错误率"""
        collector = MetricsCollector()
        
        # 10 个请求，2 个错误
        for i in range(10):
            collector.record_request(
                latency_ms=100,
                tokens=100,
                llm_calls=1,
                is_error=(i < 2),
            )
        
        report = collector.get_weekly_report()
        assert report["error_rate"] == pytest.approx(0.2, rel=0.1)


class TestLEDTracer:
    """测试追踪器"""
    
    def test_trace_context(self):
        """测试追踪上下文"""
        tracer = LEDTracer()
        
        with tracer.trace("test_session", session_id="test123") as metrics:
            metrics.route = "agent"
            metrics.llm_call_count = 2
            metrics.total_tokens = 1000
        
        assert metrics.total_latency is not None
        assert metrics.total_latency > 0
    
    def test_span_context(self):
        """测试 Span 上下文"""
        tracer = LEDTracer()
        
        with tracer.span("query_understanding"):
            pass  # 模拟操作
        
        # span 上下文管理器不应该抛出异常
    
    def test_record_error(self):
        """测试记录错误"""
        tracer = LEDTracer()
        metrics = TraceMetrics()
        
        tracer.record_error(metrics, "LLM timeout")
        
        assert metrics.error_message == "LLM timeout"
    
    def test_record_fallback(self):
        """测试记录降级"""
        tracer = LEDTracer()
        metrics = TraceMetrics()
        
        tracer.record_fallback(metrics)
        
        assert metrics.fallback_triggered is True
