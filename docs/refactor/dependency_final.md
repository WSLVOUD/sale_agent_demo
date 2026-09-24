# 最终依赖扫描（Phase 12-10）

> 扫描方式：解析 `src/**/*.py` 的三类 import（绝对 `src.x.y`、相对 `.x` / `..x.y`、
> 子模块形式 `from .pkg import sub`），src 与 tests（含根 `conftest.py`）
> 分别计数；**两边都是 0 才算无人引用**。
>
> 扫描时间：2026-09-24（Phase 12.1~12.10 改动之后）

## 1. 结果总览

```text
src 模块总数              160 → 156      （.py 文件 161 → 157）
无人引用的模块            2   → 0        ✅
仅测试引用                3   → 2
圆形依赖（import 环）      0
```

## 2. 本轮删除（Phase 12 期间）

| 模块 | 行数 | 删除依据 |
|---|---:|---|
| `src/core/sales_requirement_adapter.py` | 137 | Phase 12（计划 1.0）核对：src 0 引用 + tests 0 引用；只做 `RequirementExtractor` 包装与旧字段投影 |
| `src/core/solution_requirement_adapter.py` | 145 | 同上 |
| `src/input/turn_payload.py` | 110 | Phase 12-5：旧的消息合并入口（`TurnPayload` / `merge_message_parts` / `merge_request_payload`），只有测试与 `_merge_turn_request` 在用，而后者 0 业务调用；断言已迁移到真实入口 `collect_messages` |
| `src/memory/enhanced.py` | 493 | Phase 12-10：`EnhancedMemoryStore` 仅被 `conftest.py` 里**无人使用**的 `clean_enhanced_memory` 夹具引用；业务链已统一用 `memory/store.py` |

本轮同时删除的死代码 / 死入口：

```text
src/api.py::_merge_turn_request        （旧 Turn 合并入口，0 业务调用）
src/agents/solution/runner.py::_extract_requirements   （history 重新抽取用的包装）
src/dialogue/question_order.py::next_in_order          （0 调用方，仅挂包出口）
conftest.py::clean_enhanced_memory                     （未被任何用例使用的夹具）
```

## 3. 当前"仅测试引用"的模块（保留 + 说明）

| 模块 | 引用者 | 说明 |
|---|---|---|
| `src/api.py` | `test_api_http.py` 等 | 其实是**进程入口**（uvicorn 直接跑），不是被 import 的库模块 |
| `src/input/turn_builder.py` | `tests/input/test_turn_engine_concurrency.py` | Turn 构造工具，只被并发用例使用 —— 属计划 §12-7「TEST_ONLY」，暂留（并入测试或改名留待专门一轮，见 `docs/plans/future_optimization.md` A2） |

## 4. 已用护栏钉死的兼容例外（不允许扩散）

| 例外 | 数量 | 护栏 |
|---|---|---|
| `dialogue/product_router.py` → `agents.sales.question_planner`（函数内延迟导入） | 1 | `tests/architecture/test_architecture_boundaries.py` |
| `models/requirement.py` → `rag`（函数内延迟导入） | 3 | 同上 |
| `recommendation_engine.py` 无 profile 时的兼容解析 | 1（标 `LEGACY-COMPAT`） | `tests/architecture/test_architecture_agents_and_recommendation.py` |
| Solution 无档案时的兼容分支写旧 `requirement` | 2（标 `LEGACY-COMPAT`） | 同上 |

## 5. 边界正确性（扫描确认）

```text
RAG            → 不引用 dialogue / agents        ✅ 只做知识，不决定对话
engineering    → 不引用 agents / dialogue        ✅ 计算层独立
vision         → 不引用推荐 / 问题 / 决策        ✅ 只抽需求
memory         → 只依赖 config                   ✅ 会话层轻量
API            → 不引用任何业务决策函数          ✅ 只做输入 → TurnExecutor
```

对应护栏：`tests/architecture/`（60 条）。
