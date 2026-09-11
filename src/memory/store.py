"""
记忆模块：负责按 session_id 保存对话历史和已抽取的需求。

设计要点：
1. 只暴露一个 `memory` 单例，API、Orchestrator、Sales Agent、Solution Agent 全部共享。
2. 同一会话的存储结构为
       {"messages": [ {role, content}, ... ],
        "requirements": {...},
        "first_contact_sent": bool}
3. 超过 MAX_MEMORY_SIZE 条消息时丢弃最早的消息（FIFO），保留最近 20 条。
4. 提供 dict-like 接口 (get / __setitem__ / __contains__ / clear) 给老代码用，
   同时提供面向消息的 add / get_history 接口给新代码用。
5. first_contact_sent 用明确的业务状态判断首次接待，不依赖消息数量。
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_MEMORY_SIZE = 20  # 每个会话最多保留的消息条数


def _normalize_role(role: str) -> str:
    """将 LangChain 的消息类型 (human/ai/system) 归一化为 user/assistant/system。"""
    mapping = {"human": "user", "ai": "assistant", "system": "system"}
    return mapping.get(role, role)


def _to_message_dict(msg: Any) -> Dict[str, str]:
    """把 dict / LangChain message 对象统一转成 {role, content}。"""
    if isinstance(msg, dict):
        return {
            "role": _normalize_role(msg.get("role", "") or ""),
            "content": msg.get("content", "") or "",
        }
    # LangChain message object (HumanMessage / AIMessage / SystemMessage)
    return {
        "role": _normalize_role(getattr(msg, "type", "unknown")),
        "content": getattr(msg, "content", str(msg)),
    }


class MemoryStore:
    """
    会话级记忆：保存消息列表 + 累计需求。

    同一进程内只能有一个实例（见模块底部的 `memory` 全局变量）。
    所有读取、写入、清理都走这一个对象。
    """

    def __init__(self) -> None:
        # session_id -> {
        #     "messages": [...],
        #     "requirements": {...},
        #     "previous_display_type": str,
        #     "first_contact_sent": bool,       # 首次接待是否已完成
        #     "suppress_sales_greeting": bool,  # 首次接待刚完成后，抑制 Sales Agent 的问候语
        # }
        self._sessions: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    # ------------------------------------------------------------------
    # dict-like 接口（兼容 orchestrator / runner 旧用法）
    # ------------------------------------------------------------------
    def get(self, session_id: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """取会话数据。返回的是内部引用的拷贝，避免外部误改内部结构。"""
        if session_id not in self._sessions:
            return default if default is not None else {}
        data = self._sessions[session_id]
        return {
            "messages": list(data.get("messages", [])),
            "requirements": dict(data.get("requirements", {})),
            "first_contact_sent": bool(data.get("first_contact_sent", False)),
        }

    def __setitem__(self, session_id: str, value: Any) -> None:
        self._set(session_id, value)

    def __getitem__(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self._sessions:
            raise KeyError(session_id)
        return self.get(session_id)

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._sessions

    # ------------------------------------------------------------------
    # 面向消息的接口
    # ------------------------------------------------------------------
    def add(self, session_id: str, role: str, content: str) -> None:
        """追加一条消息，超过 MAX_MEMORY_SIZE 时丢掉最早的。"""
        self._ensure(session_id)
        messages = self._sessions[session_id]["messages"]
        messages.append({"role": _normalize_role(role), "content": content})
        # FIFO 截断
        if len(messages) > MAX_MEMORY_SIZE:
            del messages[: len(messages) - MAX_MEMORY_SIZE]

    def extend(self, session_id: str, new_messages: List[Any]) -> None:
        """批量追加消息，自动归一化角色并截断到 MAX_MEMORY_SIZE。"""
        if not new_messages:
            return
        self._ensure(session_id)
        bucket = self._sessions[session_id]["messages"]
        for msg in new_messages:
            normalized = _to_message_dict(msg)
            if normalized["role"] not in {"user", "assistant", "system"}:
                # 忽略无法识别的消息，避免污染历史
                continue
            bucket.append(normalized)
        if len(bucket) > MAX_MEMORY_SIZE:
            del bucket[: len(bucket) - MAX_MEMORY_SIZE]

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """取会话的全部历史消息，按时间顺序。"""
        if session_id not in self._sessions:
            return []
        return list(self._sessions[session_id].get("messages", []))

    def get_requirements(self, session_id: str) -> Dict[str, Any]:
        """取累计需求。"""
        if session_id not in self._sessions:
            return {}
        return dict(self._sessions[session_id].get("requirements", {}))

    def set_requirements(self, session_id: str, requirements: Dict[str, Any]) -> None:
        """覆盖写入累计需求。"""
        self._ensure(session_id)
        self._sessions[session_id]["requirements"] = dict(requirements or {})

    def get_previous_display_type(self, session_id: str) -> Optional[str]:
        """取上一个已知的屏幕类型（LED/LCD/IFP/BOTH）。"""
        if session_id not in self._sessions:
            return None
        return self._sessions[session_id].get("previous_display_type")

    def set_previous_display_type(self, session_id: str, display_type: Optional[str]) -> None:
        """保存本次检测到的屏幕类型，供下一轮判断是否发生变化。"""
        self._ensure(session_id)
        self._sessions[session_id]["previous_display_type"] = display_type

    def clear_requirements(self, session_id: str) -> None:
        """清空累计需求（屏幕类型切换时调用）。"""
        if session_id in self._sessions:
            self._sessions[session_id]["requirements"] = {}
            logger.info("Cleared accumulated requirements for session: %s", session_id)

    def get_size(self, session_id: str) -> int:
        """当前会话消息条数。"""
        if session_id not in self._sessions:
            return 0
        return len(self._sessions[session_id].get("messages", []))

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------
    def clear(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("Cleared session: %s", session_id)

    def clear_all(self) -> None:
        """清空所有会话（主要用于测试）。"""
        self._sessions.clear()

    # ── First Contact 状态 ────────────────────────────────────────────────

    def is_first_contact_done(self, session_id: str) -> bool:
        """
        判断是否已完成首次接待。

        注意：不要用 messages 数量 == 0 来判断，因为服务重启、
        Webhook 重试、多端连接等情况都可能导致该判断不准确。
        """
        if session_id not in self._sessions:
            return False
        return bool(self._sessions[session_id].get("first_contact_sent", False))

    def mark_first_contact_done(self, session_id: str) -> None:
        """标记首次接待已完成。"""
        self._ensure(session_id)
        self._sessions[session_id]["first_contact_sent"] = True
        # 抑制 Sales Agent 的问候语，因为首次接待已经自我介绍过了
        self._sessions[session_id]["suppress_sales_greeting"] = True
        logger.info("First contact completed for session: %s", session_id)

    def should_suppress_sales_greeting(self, session_id: str) -> bool:
        """
        判断是否应该抑制 Sales Agent 的问候语。
        首次接待刚完成后返回 True，被消费后自动重置为 False。
        """
        if session_id not in self._sessions:
            return False
        return bool(self._sessions[session_id].get("suppress_sales_greeting", False))

    def consume_suppress_sales_greeting(self, session_id: str) -> bool:
        """
        消费抑制标记（读取后清除）。
        返回 True 表示本次应该抑制 Sales Agent 的问候语。
        """
        if session_id not in self._sessions:
            return False
        suppress = bool(self._sessions[session_id].get("suppress_sales_greeting", False))
        if suppress:
            self._sessions[session_id]["suppress_sales_greeting"] = False
            logger.info("Sales greeting suppression consumed for session: %s", session_id)
        return suppress

    # ── 兼容旧接口 ────────────────────────────────────────────────────────

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    # ── 内部辅助 ────────────────────────────────────────────────────────────

    def _ensure(self, session_id: str) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "messages": [],
                "requirements": {},
                "previous_display_type": None,
                "first_contact_sent": False,
                "suppress_sales_greeting": False,
            }

    def _set(self, session_id: str, value: Any) -> None:
        """统一处理旧的 dict-like 赋值，使数据格式归一。"""
        if isinstance(value, dict) and "messages" in value:
            messages = [_to_message_dict(m) for m in value.get("messages", []) or []]
            requirements = dict(value.get("requirements", {}) or {})
        elif isinstance(value, list):
            # 旧格式：纯消息列表
            messages = [_to_message_dict(m) for m in value]
            requirements = self.get_requirements(session_id)
        else:
            logger.warning("Unknown session payload type for %s: %r", session_id, type(value))
            messages = []
            requirements = {}

        if len(messages) > MAX_MEMORY_SIZE:
            messages = messages[-MAX_MEMORY_SIZE:]

        self._sessions[session_id] = {
            "messages": messages,
            "requirements": requirements,
            "previous_display_type": self.get_previous_display_type(session_id),
            "first_contact_sent": self._sessions[session_id].get("first_contact_sent", False)
            if session_id in self._sessions else False,
        }


# 全局唯一实例。所有模块必须引用这个对象，不要再各自 new。
memory = MemoryStore()
