"""用例表 → 单条测试的小工具（客户口径 2026-09-22：全量测试瘦身到 500~600 条）。

很多回归用例是"同一段断言 × N 个说法"（例如 10 条中文"重新来"的说法）。
以前每条说法是一个独立用例，条数被撑得很大；现在用 :func:`assert_all_cases`
把它们压在**一条**测试里：

    · 断言一条不少（每个 case 都会真正执行）；
    · 失败时一次性列出所有失败的 case，定位反而更快；
    · 收集到的用例条数大幅下降。
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, List


def assert_all_cases(
    cases: Iterable[Any],
    check: Callable[[Any], None],
    *,
    label: str = "case",
) -> None:
    """对 ``cases`` 里每个 case 跑 ``check``；有任何失败就一次性报出来。"""
    items = list(cases)
    failures: List[str] = []
    for case in items:
        try:
            check(case)
        except AssertionError as exc:
            failures.append(f"  · {label}={case!r} → {exc}")
    assert not failures, f"{len(failures)}/{len(items)} 个用例失败：\n" + "\n".join(failures)


__all__ = ["assert_all_cases"]
