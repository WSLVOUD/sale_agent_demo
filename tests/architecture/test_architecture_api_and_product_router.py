"""计划 2.0（Phase 12-4 / 12-5）：ProductRouter 边界 + API 不参与业务决策。

Phase 12-4 规定：

    ProductTypeRouter       → 唯一判断 LED / LCD / IFP / UNKNOWN，输出 ProductTypeDecision
    ProductTypeUnderstanding→ 只理解"客户对类型问题的回应"（chose/rejects/…），不产出最终类型
    ProductRouter           → 只根据**已确认**的类型进入 LED_ENTRY / LCD_ENTRY / IFP_ENTRY

禁止：

    ProductRouter 内部重新判断"这个客户到底是 LED 还是 LCD"
    ProductTypeUnderstanding 内部直接"最终选择 LED"

Phase 12-5 规定：API 只负责输入 → TurnExecutor，不参与任何业务决策；
旧的消息合并入口（`_merge_turn_request` / `input.turn_payload`）已删除，
只保留 `API → collect_messages → TurnExecutor`。
"""
import os
import re
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _read(rel_path: str) -> str:
    with open(os.path.join(project_root, rel_path), encoding="utf-8") as handle:
        return handle.read()


_DOCSTRING_RE = re.compile(r'"""(?:.|\n)*?"""|\'\'\'(?:.|\n)*?\'\'\'')


def _code_only(text: str) -> str:
    without_docs = _DOCSTRING_RE.sub("", str(text or ""))
    return "\n".join(line.split("#", 1)[0] for line in without_docs.splitlines())


class TestProductRouterOnlyMapsConfirmedTypes:

    def test_product_router_never_decides_the_type(self):
        code = _code_only(_read("src/dialogue/product_router.py"))
        for token in (
            "route_display_type",   # 决策在 ProductTypeRouter
            "_LED_RE", "_LCD_RE",   # 也不许自己写类型识别规则
            "_IFP_RE",
            "detect_product_domain",
            "understand_product_type_reply",
        ):
            assert token not in code, f"ProductRouter 不得重新判断产品类型（{token}）"

    def test_product_type_understanding_only_returns_signals(self):
        """Understanding 只给信号，不能产出最终类型。"""
        code = _code_only(_read("src/dialogue/product_type_understanding.py"))
        for token in ("DisplayTypeDecision", "route_display_type", "load_decision"):
            assert token not in code, f"Understanding 不得直接决定最终类型（{token}）"
        assert "def understand_product_type_reply" in code

    def test_router_is_the_single_decider(self):
        """全项目只有 route_display_type 一个决策函数。"""
        hits = []
        for root, _dirs, files in os.walk(os.path.join(project_root, "src")):
            if "__pycache__" in root:
                continue
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(root, name)
                if re.search(r"^def route_display_type\b", _read(
                    os.path.relpath(path, project_root).replace("\\", "/")
                ), re.M):
                    hits.append(os.path.relpath(path, project_root).replace("\\", "/"))
        assert hits == ["src/dialogue/product_type_router.py"], hits


class TestApiDoesNotDecideBusiness:

    FORBIDDEN_IN_API = (
        "route_display_type",          # 产品类型
        "check_recommendation_ready",  # 推荐 Gate
        "recommendation_engine",       # 推荐引擎
        "field_action",                # 字段策略
        "plan_next_question",          # 提问决策
        "decide_turn_action",          # 对话决策
    )

    def test_api_only_feeds_the_turn_executor(self):
        code = _code_only(_read("src/api.py"))
        for token in self.FORBIDDEN_IN_API:
            assert token not in code, f"API 不应参与业务决策（{token}）"

    def test_legacy_turn_merge_entry_is_gone(self):
        for rel in ("src/api.py", "src/input/__init__.py"):
            code = _code_only(_read(rel))
            assert "_merge_turn_request" not in code, f"{rel} 仍挂着旧合并入口"
            assert "merge_request_payload" not in code, f"{rel} 仍挂着旧 payload 入口"
        assert not os.path.exists(
            os.path.join(project_root, "src", "input", "turn_payload.py")
        ), "旧 turn_payload 模块应已删除"

    def test_api_still_collects_messages_for_the_executor(self):
        code = _code_only(_read("src/api.py"))
        assert "collect_messages" in code
        assert "_collect_turn_request" in code
