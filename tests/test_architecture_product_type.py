"""计划《项目架构瘦身与代码整理计划》Phase 2：产品类型只有一个判断入口 / 一套词汇。

契约：

  1. 产品类型词汇（LED / LCD / IFP / UNKNOWN）**只允许有一处定义**；
     其它模块必须引用它，不能各写一份字符串常量。
  2. "客户该进哪条链"只有一个决策函数 ``route_display_type``。
  3. ``product_router.route_product_domain`` 只做"产品域 → 入口名"的映射，
     **不能**变成第二个类型判断入口（返回的必须是入口名，不是产品类型）。
  4. 兼容门面（``product_router``）对外的公开符号保持不变（有调用方依赖）。
"""
import os
import re
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

SRC = os.path.join(project_root, "src")


def _read(rel_path: str) -> str:
    with open(os.path.join(project_root, rel_path), encoding="utf-8") as handle:
        return handle.read()


def _scan_src(pattern: str):
    hits = []
    for root, _dirs, files in os.walk(SRC):
        if "__pycache__" in root:
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8", errors="ignore") as handle:
                if re.search(pattern, handle.read(), re.MULTILINE):
                    hits.append(os.path.relpath(path, project_root).replace("\\", "/"))
    return sorted(hits)


class TestSingleProductTypeSource:

    def test_vocabulary_is_defined_in_one_place_only(self):
        defining = _scan_src(r"^LED = \"LED\"")
        assert defining == ["src/dialogue/turn_kind.py"], (
            f"产品类型词汇只能有一处定义，实际：{defining}"
        )

    def test_router_reuses_the_shared_vocabulary(self):
        from src.dialogue import product_type_router as router
        from src.dialogue import turn_kind

        assert router.LED == turn_kind.LED
        assert router.LCD == turn_kind.LCD
        assert router.UNKNOWN == turn_kind.UNKNOWN
        assert router.SUBTYPE_IFP == turn_kind.IFP
        assert router.SUBTYPE_PLAIN_LCD == turn_kind.LCD

    def test_only_one_decision_entry(self):
        entries = _scan_src(r"^def route_display_type\b")
        assert entries == ["src/dialogue/product_type_router.py"], entries


class TestEntryNameMappingIsNotASecondDecisionSource:

    def test_route_product_domain_returns_entry_names_only(self):
        from src.dialogue.product_router import ALL_ENTRIES, route_product_domain

        for domain in ("LED", "LCD", "IFP", "MULTI", "UNKNOWN", "", None, "led"):
            entry = route_product_domain(domain)
            assert entry in ALL_ENTRIES, (domain, entry)
        assert route_product_domain("LCD", comparison=True) in ALL_ENTRIES

    def test_compatibility_facade_keeps_its_public_surface(self):
        """兼容门面对外符号不变（有调用方依赖，Phase 2 不动它）。"""
        from src.dialogue import product_router

        for name in (
            "ALL_ENTRIES",
            "LCD_ENTRY",
            "LED_ENTRY",
            "PRODUCT_SELECTION",
            "PRODUCT_SELECTION_QUESTION",
            "LCDPolicy",
            "LEDPolicy",
            "ProductPolicy",
            "policy_for",
            "route_product_domain",
        ):
            assert hasattr(product_router, name), name


class TestRouterStillBehaves:

    def test_first_layer_never_returns_ifp_as_a_type(self):
        from src.dialogue.product_type_router import LCD, LED, UNKNOWN, route_display_type

        for message in (
            "i need a display",
            "i need an LED display",
            "i need an LCD video wall",
            "we need to write on it in a meeting room",
            "outdoor advertising screen",
        ):
            decision = route_display_type(message)
            assert decision.display_type in (LED, LCD, UNKNOWN), (message, decision.display_type)
