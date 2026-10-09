"""Keep product routing, readiness gates, and LCD next-action decisions singular."""
from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"


def _top_level_definitions(name: str) -> list[str]:
    matches = []
    for path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
            for node in tree.body
        ):
            matches.append(path.relative_to(PROJECT_ROOT).as_posix())
    return sorted(matches)


def test_decision_functions_have_one_canonical_module():
    assert _top_level_definitions("route_display_type") == [
        "src/dialogue/product_type_router.py"
    ]
    assert _top_level_definitions("check_recommendation_ready") == [
        "src/rag/readiness.py"
    ]
    assert _top_level_definitions("check_calculation_ready") == [
        "src/rag/readiness.py"
    ]
    assert _top_level_definitions("decide_lcd_next_action") == [
        "src/dialogue/lcd_decision.py"
    ]
    assert _top_level_definitions("lcd_turn") == ["src/dialogue/lcd_decision.py"]
