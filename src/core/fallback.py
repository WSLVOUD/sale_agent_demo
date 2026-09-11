"""
LLM 降级与熔断模块
Phase 4: LLM 异常与降级

提供：
- Circuit Breaker（熔断器）
- LLM Fallback
- 人工接管机制
"""
from __future__ import annotations

import time
import threading
from enum import Enum
from typing import Callable, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """熔断器状态"""
    CLOSED = "closed"    # 正常，流量通过
    OPEN = "open"        # 熔断，拒绝请求
    HALF_OPEN = "half_open"  # 半开，允许部分请求试探


@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 5      # 连续失败多少次后熔断
    success_threshold: int = 2       # 半开状态下成功多少次后关闭
    timeout_seconds: float = 60.0    # 熔断持续时间
    half_open_max_calls: int = 3    # 半开状态下允许的试探请求数


class CircuitBreaker:
    """
    熔断器
    
    当 LLM 调用连续失败超过阈值时，熔断一段时间，
    期间直接返回降级响应，避免持续失败。
    
    状态转换：
    
    CLOSED → OPEN: 连续失败达到阈值
    OPEN → HALF_OPEN: 熔断超时
    HALF_OPEN → CLOSED: 成功达到阈值
    HALF_OPEN → OPEN: 试探失败
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()
    
    @property
    def state(self) -> CircuitState:
        """获取当前状态"""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # 检查是否超时
                if time.time() - self._last_failure_time >= self.config.timeout_seconds:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info(f"CircuitBreaker {self.name}: OPEN → HALF_OPEN (timeout)")
            return self._state
    
    def is_available(self) -> bool:
        """检查是否允许请求"""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        
        if state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
        
        return False  # OPEN 状态
    
    def record_success(self):
        """记录成功调用"""
        with self._lock:
            self._failure_count = 0
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._success_count = 0
                    logger.info(f"CircuitBreaker {self.name}: HALF_OPEN → CLOSED")
    
    def record_failure(self):
        """记录失败调用"""
        with self._lock:
            self._failure_count += 1
            self._success_count = 0
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(f"CircuitBreaker {self.name}: HALF_OPEN → OPEN (probe failed)")
            
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        f"CircuitBreaker {self.name}: CLOSED → OPEN "
                        f"(failures={self._failure_count})"
                    )
    
    def reset(self):
        """重置熔断器"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0


class LLMFallbackManager:
    """
    LLM 降级管理器
    
    提供多层降级策略：
    1. Circuit Breaker 保护
    2. Retry with exponential backoff
    3. Fallback responses
    4. Human handover
    """
    
    def __init__(self):
        # 熔断器
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        
        # Fallback responses
        self._fallback_responses: dict[str, str] = {
            "greeting": "Hello! I'm your LED display advisor. How can I help you today?",
            "warranty": "Our products come with 1-2 years of warranty (varies by series). What scenario are you looking to use it for?",
            "product_recommendation": "Thanks for your question! We carry the full range: LED, LCD, and IFP displays. Is there anything specific you'd like to know more about?",
            "default": "Sorry, the system is busy right now. Please try again shortly, or reach out to our support team for assistance.",
        }
        
        # 人工接管状态
        self._human_handover_triggered = False
        self._consecutive_failures = 0
        self._handover_threshold = 3  # 连续失败 N 次后提示人工接管
        
        self._lock = threading.Lock()
    
    def get_circuit_breaker(self, name: str = "default") -> CircuitBreaker:
        """获取熔断器"""
        with self._lock:
            if name not in self._circuit_breakers:
                self._circuit_breakers[name] = CircuitBreaker(name)
            return self._circuit_breakers[name]
    
    def get_fallback(self, query_type: str = "default") -> str:
        """获取降级响应"""
        return self._fallback_responses.get(query_type, self._fallback_responses["default"])
    
    def classify_for_fallback(self, query: str) -> str:
        """根据问题分类选择降级响应"""
        q_lower = query.lower()
        
        # 使用词边界匹配，避免 "which" 匹配 "hi"
        import re
        if re.search(r'\b(hi|hello|hey)\b', q_lower) or any(kw in q_lower for kw in ["你好", "您好"]):
            return "greeting"
        if any(kw in q_lower for kw in ["warranty", "质保", "保修", "几年", "guarantee"]):
            return "warranty"
        if any(kw in q_lower for kw in ["recommend", "推荐", "选", "哪个好", "型号", "model", "better", "best"]):
            return "product_recommendation"
        return "default"
    
    def record_failure(self, circuit_name: str = "default"):
        """记录失败"""
        cb = self.get_circuit_breaker(circuit_name)
        cb.record_failure()
        
        with self._lock:
            self._consecutive_failures += 1
    
    def record_success(self, circuit_name: str = "default"):
        """记录成功"""
        cb = self.get_circuit_breaker(circuit_name)
        cb.record_success()
        
        with self._lock:
            self._consecutive_failures = 0
    
    def should_handover(self) -> bool:
        """检查是否应该人工接管"""
        with self._lock:
            return self._consecutive_failures >= self._handover_threshold
    
    def reset_handover(self):
        """重置人工接管状态"""
        with self._lock:
            self._consecutive_failures = 0
            self._human_handover_triggered = False
    
    def add_fallback_response(self, query_type: str, response: str):
        """添加自定义降级响应"""
        with self._lock:
            self._fallback_responses[query_type] = response


def with_fallback(
    fallback_manager: LLMFallbackManager,
    circuit_name: str = "default",
):
    """
    装饰器：为函数添加降级逻辑
    
    使用方式：
    ```python
    fallback_manager = LLMFallbackManager()
    
    @with_fallback(fallback_manager)
    def call_llm(query):
        return llm.generate(query)
    ```
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            cb = fallback_manager.get_circuit_breaker(circuit_name)
            
            # 检查熔断器
            if not cb.is_available():
                logger.warning(f"CircuitBreaker {circuit_name} is OPEN, returning fallback")
                return fallback_manager.get_fallback("default")
            
            try:
                result = func(*args, **kwargs)
                fallback_manager.record_success(circuit_name)
                return result
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                fallback_manager.record_failure(circuit_name)
                
                # 检查是否应该人工接管
                if fallback_manager.should_handover():
                    return {
                        "type": "human_handover",
                        "message": "System has encountered repeated failures. Handing over to a human agent.",
                    }
                
                # 返回降级响应
                query = args[0] if args else ""
                query_type = fallback_manager.classify_for_fallback(str(query))
                return {
                    "type": "fallback",
                    "message": fallback_manager.get_fallback(query_type),
                }
        
        return wrapper
    return decorator


# 全局实例
fallback_manager = LLMFallbackManager()
