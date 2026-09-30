"""架构护栏：Sales / Solution 图里的 state 键必须**声明过**。

为什么要有这条护栏（2026-09-30 实测踩坑）：

    Sales 用的是 LangGraph `StateGraph(SalesState)`。**只有 SalesState 里声明过的键
    才会在节点之间传递** —— 没声明的键会被静默丢掉。

    实测：`requirement_mining` 明明算出了 LCD 的下一问
    （"Do you need a video wall (spliced screens) or single displays?"），但因为
    `lcd_action` 没写进 SalesState，`script_generator` 拿不到它 → 走旧兜底 →
    回复变成空字符串 → API 用"放宽条件"话术顶上，客户看到既不是问题也不是承接。

    同一类坑以前踩过一次：`display_type_decision` 没声明时，收口层的"类型没确认
    就不许问需求细节"闸门在真实链路上从来没生效过。

所以这里做静态检查：**Sales/Solution 节点与 orchestrator 里读写的 state 键，
必须在对应的 TypedDict 里声明**（只查"看起来是 state 键"的字符串下标访问，
不追求完备，但足以拦住这一类回归）。
"""
import os
import re
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

SRC = os.path.join(project_root, "src")

# 允许的例外：这些键写进的是别的字典（不是 LangGraph state），或只用于临时计算。
_ALLOWED_UNKNOWN = frozenset(
    {
        # 多屏 / 项目条目等业务结构（写在 memory 或 result 里，不是 sales state）
        "items", "profile", "reply", "model", "calculation", "active_item_index",
        "project_items", "lcd_action",  # 已在 schema 里声明（这里再兜一层，避免误报）
    }
)

_KEY_RE = re.compile(r"""state(?:\.get\(|\[)\s*['"]([a-zA-Z_][\w]*)['"]""")


def _declared_fields(module_path: str, class_name: str) -> set:
    with open(module_path, encoding="utf-8") as handle:
        text = handle.read()
    start = text.index(f"class {class_name}(")
    body = text[start:]
    end = body.find("\nclass ")
    if end != -1:
        body = body[:end]
    return set(re.findall(r"^\s{4}([a-zA-Z_][\w]*)\s*:", body, re.MULTILINE))


def _state_keys_used(package_dir: str) -> set:
    used = set()
    for root, _dirs, files in os.walk(package_dir):
        if "__pycache__" in root:
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            with open(os.path.join(root, name), encoding="utf-8", errors="ignore") as handle:
                used.update(_KEY_RE.findall(handle.read()))
    return used


class TestSalesStateDeclaresEveryUsedKey:
    def test_sales_nodes_only_use_declared_state_keys_merged(self):
        """合并自 3 条同类测试（瘦身；断言全部保留）。"""

        # ── test_sales_nodes_only_use_declared_state_keys ──
        from src.agents.sales.state import SalesState

        declared = set(SalesState.__annotations__.keys())
        used = _state_keys_used(os.path.join(SRC, "agents", "sales"))

        missing = sorted(used - declared - _ALLOWED_UNKNOWN)

        assert missing == [], (
            "这些 state 键没有写进 SalesState —— LangGraph 会把它们丢掉"
            f"（放进 src/agents/sales/state.py 的 schema 里）：{missing}"
        )

        # ── test_solution_state_declares_every_used_key ──
        from src.agents.solution.state import SolutionState

        declared = set(SolutionState.__annotations__.keys())
        used = _state_keys_used(os.path.join(SRC, "agents", "solution"))

        missing = sorted(used - declared - _ALLOWED_UNKNOWN)

        assert missing == [], (
            "这些 state 键没有写进 SolutionState —— LangGraph 会把它们丢掉："
            f"{missing}"
        )

        # ── test_lcd_action_is_a_first_class_channel ──
        """LCD 决策结果必须是图上的正式通道（否则逐轮问句会再次被丢掉）。"""
        from src.agents.sales.state import SalesState

        assert "lcd_action" in SalesState.__annotations__, SalesState.__annotations__.keys()
