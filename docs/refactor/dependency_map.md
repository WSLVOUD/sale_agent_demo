# 依赖地图（Phase 0）

## 1. 分层调用链（目标口径）

```text
HTTP /chat
   ↓
src/api.py                       ← FastAPI 入口（鉴权、SSE、回退）
   ↓
src/input/turn_executor.py       ← Turn 引擎：会话锁 / 去重 / 聚合 / 幂等 / 快照
   ↓
src/orchestrator.py              ← 双 Agent 协调 + 收口（唯一客户可见出口）
   ├── First Contact（固定接待，首轮）
   ├── Vision（带图时）
   ↓
Sales Agent  (src/agents/sales/*)      Solution Agent (src/agents/solution/*)
   classify → requirement_mining →         intent → understand → gate →
   router → script_generator               retrieve → recommend → reflect
        │                                        │
        │ 产品类型判断                            │ RAG 检索
        ↓                                        ↓
src/dialogue/product_type_router.py      src/rag/*（query_understanding,
  ├─ product_type_understanding（语境）     sparse/bm25/fusion/rerank,
  └─ product_router（入口名 + 策略占位）      recommendation_engine, readiness）
        │
        └────────────→ RequirementProfile ←────────────┘
                        （唯一需求模型，src/models/requirement.py）
                              ↓
                  engineering（计算）/ memory（会话）
                              ↓
                  Response（生成 → 校验 → Final Guard）
```

## 2. 真实 import 关系（被引用次数 Top）

统计口径：src 内部模块被多少个**别的 src 模块**引用（含父包）。

```text
 49  src.rag                     ← 检索与推荐总入口（被 agents/core 大量引用）
 31  src.models                  ← 数据模型（RequirementProfile / Product）
 26  src.config                  ← 配置
 26  src.models.requirement      ← 唯一需求模型
 24  src.dialogue                ← 对话层包入口（对外只暴露收口能力）
 23  src.core                    ← 基础设施
 22  src.agents                  ← Agent 包
 22  src.rag.query_understanding ← 查询理解（RAG 的入口）
 15  src.rag.readiness           ← Recommendation / Calculation Ready Gate
 13  src.agents.solution / src.core.llm / src.engineering / src.rag.reply_composer
 12  src.agents.sales
 10  src.memory
```

结论：

- **需求模型已经收敛**：`src/models/requirement.py` 是被引用最多的领域模型之一，
  RAG / agents / engineering 都通过它交互（Phase 1 只需补契约断言）。
- **RAG 与对话层的边界**：`src.rag.query_understanding` 被 22 处引用，是 RAG 最重的模块，
  也是计划 §14 要拆职责的对象（见 Phase 7）。
- **对话层**：外部只通过 `src.dialogue`（包 `__init__`）拿能力，`dialogue/` 内部有 34 个模块 ——
  Phase 3/4 主要在这里做职责收敛。

## 3. 关键链路（改代码前必须知道的几条）

| 链路 | 路径 |
|---|---|
| 一轮对话（主） | `api.chat → turn_executor → orchestrator.process_message → sales.runner(graph) → orchestrator._finalize_turn_response` |
| 产品类型 | `sales/nodes/classify → dialogue.product_type_router.route_display_type`（语境信号来自 `dialogue.product_type_understanding`） |
| 推荐 | `orchestrator → solution.runner → nodes.intent/requirement/retrieval/recommend/reflection → rag.readiness + rag.recommendation_engine` |
| 计算 | `sales 需求就绪 → engineering.* / tools.screen_calculator`（`rag.readiness` 的 Calculation Ready Gate 把关） |
| 图片 | `orchestrator（Vision）→ vision.pipeline → vision.integration → RequirementProfile` |
| 会话状态 | `memory.store`（消息 / 需求 / 推荐 / 项目多屏）+ `dialogue.conversation_state`（问过什么、承接计数） |

## 4. 已知的"多入口"风险点（后续阶段逐个收敛）

| 风险 | 现状 | 计划阶段 |
|---|---|---|
| 产品类型判断 | 决策源只有一个（`product_type_router`），但入口名/策略占位在 `product_router`，语境信号在 `product_type_understanding` | Phase 2 |
| 问题决策 | `question_planner` / `question_flow` / `question_order` / `question_registry` 四个模块共同决定"问什么" | Phase 3 |
| 回复组装 | `response_context` / `response_planner` / `response_generator` / `response_coordinator` / `final_guard` / `final_response` 多层 | Phase 4 |
| 需求抽取 | Sales / Solution 各自的需求节点 + `core.requirement_extractor` | Phase 5 |
| 旧适配层 | `core/sales_requirement_adapter.py`、`core/solution_requirement_adapter.py` 已无人引用 | Phase 10 |
