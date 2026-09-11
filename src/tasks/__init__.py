"""
Async Task Manager for background operations.
Phase 4: 异步 rebuild 后台任务

提供：
- 任务创建与追踪
- 进度更新
- 状态查询
- 结果存储
"""
from __future__ import annotations

import asyncio
import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TaskStatus(str, enum.Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    name: str
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0  # 0-100
    message: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
        }


class AsyncTaskManager:
    """
    异步任务管理器
    
    使用方式：
    
    ```python
    task_manager = AsyncTaskManager()
    
    # 启动后台任务
    task_id = await task_manager.create_task(
        name="rebuild_vectorstore",
        coro=rebuild_coroutine(),
        progress_callback=lambda p, m: update_progress(p, m)
    )
    
    # 查询任务状态
    status = task_manager.get_task(task_id)
    
    # 取消任务
    task_manager.cancel_task(task_id)
    ```
    """
    
    def __init__(self, max_concurrent: int = 3):
        self._tasks: dict[str, TaskInfo] = {}
        self._task_handles: dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()
    
    async def create_task(
        self,
        name: str,
        coro: Coroutine,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> str:
        """
        创建并启动异步任务
        
        Args:
            name: 任务名称
            coro: 异步协程
            progress_callback: 进度回调 (progress: int, message: str) -> None
            
        Returns:
            task_id: 任务 ID
        """
        task_id = str(uuid.uuid4())[:8]
        
        async with self._lock:
            self._tasks[task_id] = TaskInfo(
                task_id=task_id,
                name=name,
            )
        
        # 创建包装协程
        async def _run_task():
            async with self._semaphore:  # 限制并发数
                task_info = self._tasks[task_id]
                task_info.status = TaskStatus.RUNNING
                task_info.started_at = time.time()
                task_info.message = "任务开始执行"
                
                try:
                    # 执行任务，传入进度更新回调
                    result = await coro
                    task_info.status = TaskStatus.COMPLETED
                    # 保持用户最后显式设置的进度（不再覆盖为 100%）
                    task_info.message = "任务完成"
                    task_info.result = result
                    task_info.completed_at = time.time()
                    logger.info(f"Task {task_id} ({name}) completed successfully")
                    
                except asyncio.CancelledError:
                    task_info.status = TaskStatus.CANCELLED
                    task_info.message = "任务已取消"
                    task_info.completed_at = time.time()
                    logger.warning(f"Task {task_id} ({name}) was cancelled")
                    raise
                    
                except Exception as e:
                    task_info.status = TaskStatus.FAILED
                    task_info.error = str(e)
                    task_info.message = f"任务失败: {e}"
                    task_info.completed_at = time.time()
                    logger.error(f"Task {task_id} ({name}) failed: {e}")
        
        # 启动任务
        handle = asyncio.create_task(_run_task())
        async with self._lock:
            self._task_handles[task_id] = handle
        
        # 设置完成回调，清理 handle
        def _cleanup(t):
            async def _remove_handle():
                async with self._lock:
                    self._task_handles.pop(task_id, None)
            asyncio.create_task(_remove_handle())
        
        handle.add_done_callback(_cleanup)
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """获取任务状态"""
        return self._tasks.get(task_id)
    
    def list_tasks(self) -> list[dict]:
        """列出所有任务"""
        return [task.to_dict() for task in self._tasks.values()]
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        handle = self._task_handles.get(task_id)
        if handle and not handle.done():
            handle.cancel()
            task_info = self._tasks.get(task_id)
            if task_info:
                task_info.status = TaskStatus.CANCELLED
                task_info.message = "正在取消..."
            return True
        return False
    
    def update_progress(self, task_id: str, progress: int, message: str = ""):
        """更新任务进度（供任务内部调用）"""
        task_info = self._tasks.get(task_id)
        if task_info:
            task_info.progress = max(0, min(100, progress))
            if message:
                task_info.message = message
    
    async def wait_task(self, task_id: str, timeout: float = 300) -> Optional[Any]:
        """
        等待任务完成
        
        Args:
            task_id: 任务 ID
            timeout: 超时时间（秒）
            
        Returns:
            任务结果，超时返回 None
        """
        handle = self._task_handles.get(task_id)
        if not handle:
            return None
        
        try:
            await asyncio.wait_for(handle, timeout=timeout)
            return self._tasks[task_id].result
        except asyncio.TimeoutError:
            logger.warning(f"Task {task_id} wait timeout")
            return None
        except asyncio.CancelledError:
            return None


# 全局任务管理器实例
task_manager = AsyncTaskManager(max_concurrent=3)
