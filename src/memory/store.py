"""
记忆模块：负责按 session_id 保存对话历史和已抽取的需求。

设计要点：
1. 只暴露一个 `memory` 单例，API、Orchestrator、Sales Agent、Solution Agent 全部共享。
2. 同一会话的存储结构为
       {"messages": [ {role, content}, ... ],
        "requirements": {...},
        "first_contact_sent": bool}
3. 超过 MAX_MEMORY_SIZE 条消息时丢弃最早的消息（FIFO），保留最近 50 条。
4. 提供 dict-like 接口 (get / __setitem__ / __contains__ / clear) 给老代码用，
   同时提供面向消息的 add / get_history 接口给新代码用。
5. first_contact_sent 用明确的业务状态判断首次接待，不依赖消息数量。
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_MEMORY_SIZE = 50  # 每个会话最多保留的消息条数（短期记忆上限）

# 进程内最多保留多少个会话。客户刷新页面 / 开新标签页都会新建 session_id，
# 不设上限的话长期运行的服务会一直堆积（内存只增不减，只有重启才释放）。
MAX_SESSIONS = 200


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

    def replace_last_assistant(self, session_id: str, content: str) -> bool:
        """把历史里最后一条 assistant 消息换成**真正发给客户的那条**。

        客户口径（2026-09-22）：`others / product_question` 这一轮，Sales 只写了
        占位符（"Sure."），真正的答复是 Solution Agent 产出的。如果不换掉，
        下一轮喂给 LLM 的"最近 50 条"里，AI 自己说过什么就看不见了
        （客户回 "yes" 时也就不知道上一句在问"要不要出报价"）。

        返回 True 表示真的替换了内容。
        """
        text = str(content or "").strip()
        if not text or session_id not in self._sessions:
            return False
        messages = self._sessions[session_id].get("messages") or []
        for entry in reversed(messages):
            if not isinstance(entry, dict):
                continue
            if str(entry.get("role") or "") != "assistant":
                continue
            if str(entry.get("content") or "").strip() == text:
                return False
            entry["content"] = text
            return True
        return False

    def get_requirements(self, session_id: str) -> Dict[str, Any]:
        """取累计需求（M10：旧字段，是 RequirementProfile 的**只读投影**）。

        需求的主状态是 ``requirement_profile``；这里的 requirements 只在
        "老会话还没有 Profile"时作为兜底输入使用，不再作为第二份真相。
        """
        if session_id not in self._sessions:
            return {}
        return dict(self._sessions[session_id].get("requirements", {}))

    def set_requirements(self, session_id: str, requirements: Dict[str, Any]) -> None:
        """覆盖写入累计需求（M10：写入的是 Profile 的投影，见 legacy_adapter）。"""
        self._ensure(session_id)
        self._sessions[session_id]["requirements"] = dict(requirements or {})

    # ── Phase 6：结构化需求档案 ──────────────────────────────────────────
    def get_requirement_profile(self, session_id: str) -> Optional[Dict[str, Any]]:
        """取结构化需求档案（RequirementProfile 的 dict 形式）。"""
        if session_id not in self._sessions:
            return None
        profile = self._sessions[session_id].get("requirement_profile")
        return dict(profile) if profile else None

    def set_requirement_profile(self, session_id: str, profile: Any) -> None:
        """写入结构化需求档案（接受 RequirementProfile 或 dict）。

        全空档案（每个字段都是 None / []）按"没有档案"处理 —— 需求重置后
        不应该留下一个"看起来存在、其实什么都没有"的档案给下一轮复用。
        """
        if profile is None:
            return
        self._ensure(session_id)
        if hasattr(profile, "model_dump"):
            profile = profile.model_dump()
        data = dict(profile or {})
        meaningful = {
            key: value
            for key, value in data.items()
            if key != "sources" and value not in (None, "", [], {})
        }
        self._sessions[session_id]["requirement_profile"] = data if meaningful else None

    def clear_requirement_profile(self, session_id: str) -> None:
        """清空结构化需求档案（屏幕类型切换时调用）。"""
        if session_id in self._sessions:
            self._sessions[session_id]["requirement_profile"] = None

    # ── 一个项目下的多条屏体需求（客户口径 2026-09-18）───────────────────
    # 客户可能一次要两块屏（教堂里一块室内屏 + 门口一块室外屏，或 LED + LCD/IFP）。
    # 每条记录 = 该块屏的需求档案 + 它自己的推荐结果；`active_item` 指向"当前正在
    # 采集/推荐的那一块"。`requirement_profile` 始终是**当前这一块**的档案，
    # 所有老逻辑（Gate / 推荐 / 计算）都不用改。
    def get_project_items(self, session_id: str) -> List[Dict[str, Any]]:
        if session_id not in self._sessions:
            return []
        items = self._sessions[session_id].get("project_items") or []
        return [dict(item) for item in items]

    def set_project_items(self, session_id: str, items: Optional[List[Dict[str, Any]]]) -> None:
        self._ensure(session_id)
        self._sessions[session_id]["project_items"] = [dict(item) for item in (items or [])]

    def get_active_item_index(self, session_id: str) -> int:
        if session_id not in self._sessions:
            return 0
        try:
            return int(self._sessions[session_id].get("active_item_index") or 0)
        except (TypeError, ValueError):
            return 0

    def set_active_item_index(self, session_id: str, index: int) -> None:
        self._ensure(session_id)
        self._sessions[session_id]["active_item_index"] = max(0, int(index or 0))

    def record_item_recommendation(self, session_id: str, item: Dict[str, Any]) -> None:
        """把"当前这块屏"的推荐结果写进项目条目里（供最后汇总用）。"""
        self._ensure(session_id)
        items = list(self._sessions[session_id].get("project_items") or [])
        index = self.get_active_item_index(session_id)
        while len(items) <= index:
            items.append({})
        merged = dict(items[index] or {})
        merged.update(item)
        items[index] = merged
        self._sessions[session_id]["project_items"] = items

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

    # ── 会话内推荐状态（用于识别"客户要换产品 / 换需求"）──────────────────
    def mark_recommendation_done(self, session_id: str, products: Optional[List[Any]] = None) -> None:
        """记录"本会话已经给过产品推荐"，供下一轮判断客户是否在换产品。

        只保留型号名，避免把整份产品对象留在会话里。
        """
        self._ensure(session_id)
        models: List[str] = []
        for item in products or []:
            if isinstance(item, dict):
                model = item.get("model") or item.get("name") or item.get("series")
            else:
                model = getattr(item, "model", None)
            if model and str(model) not in models:
                models.append(str(model))
        self._sessions[session_id]["recommendation"] = {
            "delivered": True,
            "models": models[:10],
            "at": time.time(),
        }
        logger.info("Marked recommendation delivered for session: %s (%s)", session_id, models[:3])

    def has_recommendation(self, session_id: str) -> bool:
        """本会话是否已经给客户推荐过产品。"""
        if session_id not in self._sessions:
            return False
        record = self._sessions[session_id].get("recommendation") or {}
        return bool(record.get("delivered"))

    def get_recommendation(self, session_id: str) -> Dict[str, Any]:
        """取本会话最近一次推荐记录（没有则返回 {}）。"""
        if session_id not in self._sessions:
            return {}
        return dict(self._sessions[session_id].get("recommendation") or {})

    def clear_recommendation(self, session_id: str) -> None:
        """清除推荐记录（需求重置后，这是一次全新的咨询）。"""
        if session_id in self._sessions:
            self._sessions[session_id]["recommendation"] = None

    def reset_requirement_state(self, session_id: str) -> None:
        """清空一次咨询的全部中间状态：累计需求 + 结构化档案 + 推荐记录。

        客户在拿到推荐之后要换产品 / 换项目 / 改需求时调用，
        让 Sales Agent 回到"从零开始采集需求"的状态。
        """
        if session_id not in self._sessions:
            return
        self._sessions[session_id]["requirements"] = {}
        self._sessions[session_id]["requirement_profile"] = None
        self._sessions[session_id]["recommendation"] = None
        logger.info("Reset requirement state for session: %s", session_id)

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
                "recommendation": None,
            }
            # LRU 淘汰：超出上限时丢掉最久未使用的会话，避免长时间运行内存无限增长
            while len(self._sessions) > MAX_SESSIONS:
                evicted, _ = self._sessions.popitem(last=False)
                logger.info("Evicted least-recently-used session: %s", evicted)
        else:
            # 被访问过 → 移到队尾（OrderedDict 的 LRU 语义）
            self._sessions.move_to_end(session_id)

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
            "recommendation": self._sessions[session_id].get("recommendation")
            if session_id in self._sessions else None,
        }


# 全局唯一实例。所有模块必须引用这个对象，不要再各自 new。
memory = MemoryStore()
