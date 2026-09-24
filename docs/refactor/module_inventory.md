# 模块清单（Phase 0）

> 给 `src/` 下每个模块定一个"角色"，决定它能不能被合并 / 删除。
> 判定方式是**真实引用索引**（不用肉眼猜），脚本见文末。

## 1. 分类口径

| 分类 | 含义 | 处理 |
|---|---|---|
| CORE | 有真实业务职责、被别处引用 | 保留；只做职责收敛 |
| ADAPTER | 只做"旧结构 → RequirementProfile"这类转换 | 保留为兼容层，等调用方迁移完再删 |
| COMPATIBILITY | 对外/对旧调用方的稳定接口（改名会破坏调用） | 保留，标注兼容性质 |
| LEGACY | **没有任何引用者**（src 与 tests 都没有） | 确认后删除（Phase 10） |
| TEST_ONLY | 只有测试引用 | 保留（测试契约），必要时并回 CORE |

## 2. 结论（160 个模块）

### 2.1 LEGACY —— 无人引用（可直接进入 Phase 10 删除清单）

| 模块 | 行数 | 证据 |
|---|---:|---|
| `src/core/sales_requirement_adapter.py` | 137 | src 内 0 引用、tests 0 引用；只在自身内部出现 `get_sales_requirement_adapter()` |
| `src/core/solution_requirement_adapter.py` | 145 | src 内 0 引用、tests 0 引用；只在自身内部出现 `get_solution_requirement_adapter()` |

> 这两个就是计划 §4 里点名的"旧需求适配层"。现在 Sales / Solution 都已经直接用
> `RequirementProfile`，它们没有任何调用方 —— 属于"新模块已接管"的情形。

### 2.2 TEST_ONLY —— 只有测试引用

| 模块 | 引用者 | 说明 |
|---|---|---|
| `src/api.py` | `test_api_http.py` 等 | 其实是**进程入口**（uvicorn 直接跑），不是被 import 的库模块 |
| `src/input/turn_builder.py` | `tests/input/test_turn_engine_concurrency.py` | Turn 引擎的构造工具，只被并发测试用 |
| `src/memory/enhanced.py` | 根目录 `conftest.py` | 增强记忆，只有测试夹具在清它；业务链路已统一用 `memory/store.py` |

### 2.3 COMPATIBILITY —— 兼容/门面（保留）

| 模块 | 现状 | 结论 |
|---|---|---|
| `src/models/legacy_adapter.py` | `profile_to_legacy()` 把 RequirementProfile 投影成旧 `requirements` dict | 保留：orchestrator / sales 链路仍在用（旧字段是只读投影） |
| `src/dialogue/product_router.py` | 只做"产品域 → 入口名"映射 + LED/LCD/IFP 策略占位 | 保留：是入口名与策略接口的稳定门面，Phase 2 只收敛职责，不删 |

### 2.4 CORE —— 其余全部模块

按包看核心职责（详见 `dependency_map.md`）：

```text
agents/     Sales / Solution 两个 LangGraph Agent（节点 + runner + state）
dialogue/   对话决策：产品类型、问题、动作、回复生成与校验
rag/        检索与知识：query understanding / 混合检索 / corpus / 推荐引擎 / readiness
core/       基础设施：LLM、embeddings、需求抽取、语言与兜底
input/      Turn 引擎：去重、聚合、会话锁、幂等、状态快照
engineering/ 工程计算：箱体 / 模组 / 视距 / 点间距 / 可行性
vision/     图片需求识别（智谱）与并入 RequirementProfile
memory/     会话记忆（store）+ 历史窗口
models/     数据模型（RequirementProfile / Product）
observability / tools / utils / tasks / first_contact
```

## 3. 复用"引用索引"的方法（可复查）

判定脚本做的事：

```text
1. 列出 src/**/*.py，映射成模块名（含包 __init__）
2. 扫三类 import：
   · from src.x.y import ...        （绝对导入，含父包）
   · from .x / from ..x import ...  （相对导入，按所在包解析）
   · from .pkg import sub           （子模块形式，例如 from .nodes import retrieval）
3. src 与 tests（含根目录 conftest.py / init_vectorstore.py）分别计数
4. "src 与 tests 都为 0" → LEGACY 候选
```

> 这一步是必须的：最初用"文本 grep"扫时，`src/agents/solution/nodes/retrieval.py`
> 被误判成无人引用 —— 实际它在 `solution/graph.py` 里以
> `from .nodes import retrieval as ret_node` 的形式被挂进 LangGraph。
> **凡是要删的文件，都必须用引用索引 + 测试双重确认。**

## 4. 待确认（暂不删，留到 Phase 10 复核）

| 模块 | 为什么暂不删 |
|---|---|
| `src/input/turn_builder.py` | 测试在用；如果确认只是"测试专用构造器"，应改名为 `_test_` 前缀或并入 `input/` 的公开接口 |
| `src/memory/enhanced.py` | 只有 conftest 在用；如果业务链确实不再需要，需要先把它从 conftest 摘掉再删 |
| `src/dialogue/natural_response.py` / `natural_continuation.py` / `ab_test.py` | 属于 Phase 4"Response 层"范围，先做职责核对（见 dependency_map），不单独删 |
