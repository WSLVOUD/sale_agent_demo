"""
Phase 14（补测）：HTTP 层端到端测试。

用 FastAPI TestClient 真启动应用（会触发 startup_event：构建 Model 级语料、
校验/加载向量库、初始化 Sales + Solution Agent 与 Orchestrator），覆盖：

  - 根路径 / 健康检查 / 检索诊断
  - API Key 鉴权（无 Key 401 / 有 Key 200）
  - /chat 非流式：首次接待固定流程 + 后续轮次的降级兜底
  - /chat SSE 流式：事件格式 start → chunk → done
  - /memory/{session_id} 与 /memory/clear
  - /rebuild 异步任务与状态查询
  - 静态资源（聊天页面）

注意：本环境无外网，LLM 调用会失败并走降级路径 —— 这正是我们要验证的
"LLM 不可用时系统仍可用"。真实 LLM 输出质量仍需在有 Key 的环境另测。
"""
import json
import os
import sys
import time

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# conftest 已设置 LED_API_KEY=test-api-key（见 tests/conftest.py）
API_KEY = os.environ.get("LED_API_KEY", "test-api-key")
AUTH = {"X-API-Key": API_KEY}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from src.api import app

    with TestClient(app) as test_client:
        yield test_client


class TestHealthAndDiagnostics:

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "running"

    def test_health_reports_model_level_store(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["orchestrator_ready"] is True
        assert payload["vectorstore_ready"] is True
        # Phase 2 之后：语料必须是 49 个 Model（9 Series）
        assert payload["record_count"] == 49

    def test_retrieval_diagnostics(self, client):
        payload = client.get("/diagnostics/retrieval").json()
        assert payload["record_count"] == 49
        assert payload["indoor_records"] == 31
        assert payload["outdoor_records"] == 18
        assert payload["led_records"] == 49


class TestAuth:

    def test_chat_requires_api_key(self, client):
        response = client.post("/chat", json={"session_id": "auth-1", "question": "hi"})
        assert response.status_code == 401

    def test_chat_accepts_valid_api_key(self, client):
        response = client.post(
            "/chat",
            json={"session_id": "auth-2", "question": "hi"},
            headers=AUTH,
        )
        assert response.status_code == 200

    def test_memory_and_rebuild_require_api_key(self, client):
        assert client.get("/memory/auth-3").status_code == 401
        assert client.post("/memory/clear", json={"session_id": "auth-3"}).status_code == 401
        assert client.post("/rebuild").status_code == 401


class TestChatFirstContact:
    """首轮消息走固定接待流程（自我介绍 + 素材 + 名片）"""

    def test_first_turn_runs_fixed_flow(self, client):
        session_id = f"http-fc-{int(time.time())}"
        response = client.post(
            "/chat",
            json={"session_id": session_id, "question": "Hi, I need LED displays for my shop"},
            headers=AUTH,
        )
        assert response.status_code == 200
        payload = response.json()

        # 固定流程必须在第一轮完整执行并直接返回
        assert payload["route"] == "first_contact"
        assert payload["first_contact_intro"]
        assert payload["first_contact_messages"], "应返回自我介绍 + 素材消息"
        assert payload["products"] == []

        # 生成的消息要落进记忆（含客户首条消息）
        history = client.get(f"/memory/{session_id}", headers=AUTH).json()
        roles = [msg["role"] for msg in history["messages"]]
        assert "user" in roles, "客户的首条消息必须写入记忆"
        assert roles.count("assistant") >= 2, "自我介绍与素材消息都应写入记忆"

        # v2.0 Phase 优化：客户在首条消息中描述的需求必须被提取并存入记忆
        # 这样第二轮 Sales Agent 不会重复询问已经说过的需求
        requirements = history.get("requirements", {})
        assert requirements, "客户首条消息中的需求必须被提取（即使 LLM 提取可能为空，但不能抛异常）"

        client.post("/memory/clear", json={"session_id": session_id}, headers=AUTH)

    def test_first_contact_extracts_requirements_for_second_turn(self, client):
        """首次联系时客户描述的需求必须被提取，这样第二轮不会重复询问"""
        session_id = f"http-req-{int(time.time())}"

        # 第一轮：客户发送包含需求的消息
        r1 = client.post(
            "/chat",
            json={"session_id": session_id, "question": "I need an LED screen for a meeting room with 15 people"},
            headers=AUTH,
        )
        assert r1.status_code == 200
        assert r1.json()["route"] == "first_contact"

        # 验证需求被提取并保存
        mem = client.get(f"/memory/{session_id}", headers=AUTH).json()
        requirements = mem.get("requirements", {})
        assert requirements, "客户在首条消息中描述的需求必须被提取"
        # 至少应该提取到 display_type=LED 和 purpose 相关的信息
        assert requirements.get("display_type") == "LED" or requirements.get("purpose"), \
            f"应提取到 LED 类型或用途信息，实际: {requirements}"

        # 第二轮：客户继续对话
        r2 = client.post(
            "/chat",
            json={"session_id": session_id, "question": "What's the brightness level?"},
            headers=AUTH,
        )
        assert r2.status_code == 200
        # 第二轮进入 Sales Agent 或 Solution Agent，路由不应是 first_contact
        assert r2.json()["route"] != "first_contact", "第二轮不应再触发 first_contact"

        client.post("/memory/clear", json={"session_id": session_id}, headers=AUTH)


class TestStreaming:

    def test_sse_stream_format(self, client):
        with client.stream(
            "POST",
            "/chat",
            json={"session_id": f"http-sse-{int(time.time())}", "question": "outdoor LED screen"},
            headers={**AUTH, "Accept": "text/event-stream"},
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            events = []
            for line in response.iter_lines():
                if not line:
                    continue
                text = line if isinstance(line, str) else line.decode("utf-8")
                if text.startswith("data: "):
                    events.append(json.loads(text[6:]))

        assert events, "SSE 至少应返回一个事件"
        assert events[0]["type"] == "start"
        assert events[-1]["type"] == "done"
        assert any(event["type"] == "chunk" for event in events), "应有文本增量事件"


class TestMemoryEndpoints:

    def test_memory_roundtrip(self, client):
        session_id = f"http-mem-{int(time.time())}"
        client.post("/chat", json={"session_id": session_id, "question": "Hi"}, headers=AUTH)
        payload = client.get(f"/memory/{session_id}", headers=AUTH).json()
        assert payload["session_id"] == session_id
        assert payload["count"] >= 1

        cleared = client.post("/memory/clear", json={"session_id": session_id}, headers=AUTH)
        assert cleared.status_code == 200
        assert client.get(f"/memory/{session_id}", headers=AUTH).json()["count"] == 0


class TestRebuildTask:

    def test_rebuild_is_async_and_pollable(self, client):
        submitted = client.post("/rebuild", headers=AUTH)
        assert submitted.status_code == 200
        task_id = submitted.json()["task_id"]
        assert task_id

        deadline = time.time() + 90
        status_payload = None
        while time.time() < deadline:
            status_payload = client.get(f"/rebuild/{task_id}", headers=AUTH).json()
            if status_payload["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(3)

        assert status_payload is not None
        assert status_payload["name"] == "rebuild_vectorstore"
        assert status_payload["status"] in ("pending", "running", "completed", "failed", "cancelled")
        assert 0 <= status_payload["progress"] <= 100

        if status_payload["status"] in ("pending", "running"):
            client.post(f"/rebuild/{task_id}/cancel", headers=AUTH)

        # 回归：重建必须是"替换"而不是"追加"，否则语料会翻倍（49 → 98）
        if status_payload["status"] == "completed":
            health = client.get("/health").json()
            assert health["record_count"] == 49, "重建后向量库记录数必须回到 49"

        tasks = client.get("/rebuild", headers=AUTH)
        assert tasks.status_code == 200
        assert isinstance(tasks.json(), list)

    def test_unknown_task_returns_404(self, client):
        assert client.get("/rebuild/nope-not-exist", headers=AUTH).status_code == 404


class TestStaticAssets:

    def test_chat_page_served(self, client):
        response = client.get("/static/index.html")
        assert response.status_code == 200
        assert "html" in response.text.lower()

    def test_first_contact_assets_mounted(self, client):
        # 目录已挂载（具体文件是否存在取决于素材是否放置）
        response = client.get("/static/first_contact/README.md")
        assert response.status_code in (200, 404)
