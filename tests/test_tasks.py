"""
测试异步任务管理器
Phase 4: pytest 测试工程化

测试覆盖：
- AsyncTaskManager 任务创建与追踪
- 进度更新
- 任务取消
- 状态查询
"""
import pytest
import asyncio
import time
from src.tasks import AsyncTaskManager, TaskStatus


class TestAsyncTaskManager:
    """测试异步任务管理器"""
    
    @pytest.mark.asyncio
    async def test_create_task(self):
        """测试创建任务"""
        manager = AsyncTaskManager(max_concurrent=3)
        
        async def dummy_task():
            await asyncio.sleep(0.1)
            return "result"
        
        task_id = await manager.create_task(name="test_task", coro=dummy_task())
        
        assert task_id is not None
        assert len(task_id) == 8  # UUID 前 8 位
        
        # 等待任务完成
        await asyncio.sleep(0.2)
        
        task_info = manager.get_task(task_id)
        assert task_info is not None
        assert task_info.status == TaskStatus.COMPLETED
        assert task_info.result == "result"
    
    @pytest.mark.asyncio
    async def test_task_progress_update(self):
        """测试进度更新"""
        manager = AsyncTaskManager()

        captured_task_id = []

        async def task_with_progress():
            # 从 manager 任务列表中找最新任务（被 create_task 注入的）
            for _ in range(200):
                if manager._tasks:
                    captured_task_id.append(list(manager._tasks.keys())[-1])
                    break
                await asyncio.sleep(0.001)
            tid = captured_task_id[0]
            manager.update_progress(tid, 10, "开始")
            await asyncio.sleep(0.01)
            manager.update_progress(tid, 50, "进行中")
            await asyncio.sleep(0.01)
            manager.update_progress(tid, 90, "快完成了")
            return "done"

        # 创建任务
        task_id = await manager.create_task(
            name="progress_test",
            coro=task_with_progress(),
        )

        # 等待任务完成
        await asyncio.sleep(0.1)

        task_info = manager.get_task(task_id)
        assert task_info.progress == 90  # 最后更新的进度
    
    @pytest.mark.asyncio
    async def test_task_failure(self):
        """测试任务失败"""
        manager = AsyncTaskManager()
        
        async def failing_task():
            await asyncio.sleep(0.01)
            raise ValueError("Task failed intentionally")
        
        task_id = await manager.create_task(
            name="failing_task",
            coro=failing_task(),
        )
        
        # 等待任务失败
        await asyncio.sleep(0.1)
        
        task_info = manager.get_task(task_id)
        assert task_info.status == TaskStatus.FAILED
        assert "Task failed" in task_info.error
    
    @pytest.mark.asyncio
    async def test_list_tasks(self):
        """测试列出所有任务"""
        manager = AsyncTaskManager()
        
        async def dummy_task():
            await asyncio.sleep(0.05)
            return "done"
        
        # 创建多个任务
        for i in range(3):
            await manager.create_task(name=f"task_{i}", coro=dummy_task())
        
        tasks = manager.list_tasks()
        assert len(tasks) == 3
    
    @pytest.mark.asyncio
    async def test_cancel_task(self):
        """测试取消任务"""
        manager = AsyncTaskManager()
        
        async def long_task():
            await asyncio.sleep(10)  # 长时间任务
            return "done"
        
        task_id = await manager.create_task(name="long_task", coro=long_task())
        
        # 等待任务启动
        await asyncio.sleep(0.05)
        
        # 取消任务
        success = await manager.cancel_task(task_id)
        assert success is True
        
        # 验证状态
        task_info = manager.get_task(task_id)
        assert task_info.status == TaskStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_max_concurrent(self):
        """测试并发限制"""
        manager = AsyncTaskManager(max_concurrent=2)
        
        start_times = []
        
        async def tracked_task(task_num):
            start_times.append(time.time())
            await asyncio.sleep(0.2)
            return task_num
        
        # 创建 4 个任务（限制为 2 并发）
        task_ids = []
        for i in range(4):
            task_id = await manager.create_task(
                name=f"concurrent_task_{i}",
                coro=tracked_task(i),
            )
            task_ids.append(task_id)
        
        # 等待所有任务完成
        await asyncio.sleep(1)
        
        # 验证所有任务都完成了
        for task_id in task_ids:
            task_info = manager.get_task(task_id)
            assert task_info.status == TaskStatus.COMPLETED


class TestTaskInfo:
    """测试任务信息"""
    
    def test_to_dict(self):
        """测试转换为字典"""
        from src.tasks import TaskInfo
        
        info = TaskInfo(
            task_id="test123",
            name="test_task",
            status=TaskStatus.RUNNING,
            progress=50,
            message="进行中",
        )
        
        data = info.to_dict()
        assert data["task_id"] == "test123"
        assert data["status"] == "running"
        assert data["progress"] == 50
