# LED RAG 优化计划 · 最终验收报告

- 日期：2026-09-13
- 范围：计划文档 Phase 0 ~ Phase 14 全部执行完毕
- 数据集：`eval/golden_dataset.json`（72 条用例）
- 语料：`data/led_products.json` → **49 个 Model / 9 个 Series**（Model 级 RAG）
- 测试：`python -m pytest tests/` → **235 passed / 0 skipped / 0 failed**（约 6 分钟，含 HTTP 端到端与向量库重建）

## 一、目标 vs 实际

| 指标 | 目标 | 基线（Phase 0） | 最终 | 结论 |
|---|---|---|---|---|
| Requirement Slot Accuracy | ≥ 0.90 | 0.526 | **0.996** | ✅ |
| Model Recall@5（检索） | ≥ 0.80 | 0.382 | 0.748 | ⚠️ 接近未达 |
| Model Recall@10（检索） | ≥ 0.90 | 0.488 | **0.919** | ✅ |
| Model MRR（检索） | ≥ 0.80 | 0.381 | 0.689 | ⚠️ 未达 |
| Top-1 Recommendation Accuracy | ≥ 0.85 | 0.366 | **0.878** | ✅ |
| Top-3 Recommendation Coverage | ≥ 0.95 | 0.537 | **1.000** | ✅ |
| Hard Constraint Violation | = 0 | 0.444 | **0.000** | ✅ |
| Calculator Accuracy | = 1.00 | 未实现 | **1.000（14/14）** | ✅ |

> 「Top-1 / Top-3」是**最终推荐结果**指标（Recommendation Engine，Phase 8）；
> 「Model Recall@5 / MRR」是**检索候选**指标，检索的目标是把正确型号放进候选集，
> 最终由确定性打分完成选择 —— 当前 Top-5 覆盖 0.927、点间距区间命中 0.976。

## 二、分阶段结果

| Phase | 关键结果 |
|---|---|
| 0 评估基线 | 建立 72 条 Golden Dataset + 三个评估脚本；量化出「Series 能召回、Model 定位不住、点间距过滤误删」三大问题 |
| 1 数据标准化 | 9 Series / 49 Model 全部通过数据检查；修正户外系列 module_resolution 错误；记录 5 条厂商规格自身不一致 |
| 2 Model 级 RAG | 语料从 21 个 Series 文本块 → 49 个 Model 文档；Series Recall@5 0.689 → 0.925，Model Recall@5 0.382 → 0.659（后续筛选后 0.748） |
| 3 硬约束过滤 | 硬约束违规率 0.444 → **0.000**；删除 rerank / recommend 里"候选为空就退回全部产品"的兜底复活逻辑 |
| 4 Query Understanding | 槽位准确率 0.526 → 0.992；观看距离槽位 0.000 → 0.966；标准化英文检索式落地 |
| 5 参数推断重构 | 多套重复规则合并为唯一规则表；LLM 退出工程计算；客户显式参数优先 |
| 6 Requirement Profile | 结构化档案 + 来源追踪 + 合并策略 + 记忆持久化接口 |
| 7 需求采集 | 一次只问一个高价值问题；信息不足不再直接推荐；追问不消耗 LLM |
| 8 Recommendation Engine | Top-1 0.878 / Top-3 1.000 / 违规 0；输出 Model 级 + 打分明细 + 推荐理由 |
| 9 Screen Calculator | 14/14（100%），含整除/不整除/小数箱体/超大尺寸 |
| 10 Solution Agent | 「选型 → 证据 → 计算 → 一次表达」；LLM 不可用时模板降级仍给出完整方案 |
| 11 Reflection | 10 项确定性校验取代 LLM 重新决策；可识别伪造参数与算错的箱体/模组数量 |
| 12 LLM 调用 | 单轮约 9~11 次 → 3~4 次；工程参数与选型完全脱离 LLM 随机性 |
| 13 多语言 | 11 种语言关键词 + 语言检测 + 统一英文检索式；9 条多语言用例槽位全对 |
| 14 自动化测试 | 新增 53 条测试（要求档案 / 推荐引擎 / 计算器 / 校验 / LLM 调用预算 / HTTP 端到端），全量 **235 通过、0 跳过** |

## 三、仍未达成的项与原因

1. **检索 Model Recall@5 = 0.748（目标 0.80）、Model MRR = 0.689（目标 0.80）**
   - 评测口径严格：期望集合只列 1~3 个"最合适"型号，而检索会返回同系列相邻点间距（这些型号同样合规）。
   - Top-5 覆盖 0.927、点间距区间命中率 0.976，说明候选集质量已可用；最终推荐由 Phase 8 收敛到 Top-1 0.878。
   - 若要把检索层指标也推到 0.80+，下一步可做：Query Rewrite 增加场景→点间距的显式提示词、对同系列做点间距分层重排。
2. **`advertising` tag 的 Series Recall@5 = 0.867、`conflict` tag = 0.333**
   - 户外广告用例常同时出现多个品牌系列，期望集合写得较窄；冲突需求（室内外同时出现）本身允许两种解释。
3. **产品库仍只有 LED**：LCD / IFP 用例（如指挥中心拼接屏、会议一体机）按设计返回"无匹配"，需要补数据才能推荐。

## 四、复现方式

```bash
python -m src.rag.json_loader --validate   # 产品数据检查（9 Series / 49 Model）
python -m eval.recommendation_eval         # 需求理解 + 路由 + 推荐引擎
python -m eval.retrieval_eval              # 检索（约 1~2 分钟，需要本地 BGE-M3）
python -m eval.calculator_eval             # 箱体 / 模组计算
python -m pytest tests/ -q                 # 全量回归
```

## 五、复核阶段（交付后自查）额外发现并修复

| 问题 | 影响 | 处理 |
|---|---|---|
| `tests/test_retrieval.py` 把 LangChain `Document` 直接传给 `BM25Search`，fixture 抛错 | 7 条检索测试**一直被静默 skip**，等于没有回归保护 | 改为使用生产语料 + `corpus_as_dicts()`，并新增环境过滤、Model 级语料断言；现在 11 条真实运行 |
| 磁盘默认向量库 `vectorstore/` 是过期的 34 条 Series 级数据 | 直接读默认库会拿到旧语料，检索质量与报告不符 | 用 `init_vectorstore.py` 按 49 条 Model 重建；API 启动自检也能识别并自动重建 |
| `parameter_inference_node` 只从 `messages` 取用户消息 | 无 `messages`（如仅传 `current_message`）时参数推断为空 | 回退到 `current_message`，与 retrieval 节点一致 |
| 计划文档 Phase 12 的"LLM 调用降到 3~4 次"只是推算 | 无证据 | 新增 `tests/test_llm_budget.py` 用计数假 LLM 固化：推荐链路 1 次、Reflection 0 次、参数推断 0 次，并验证 LLM 掉线时模板降级 |
| **`POST /rebuild` 返回的 task_id 查不到状态（永远 404）** | 高：异步重建功能实际不可用 | `create_task()` 支持显式传入 task_id，API 传入并返回同一个 id；已由 `tests/test_api_http.py` 覆盖 |
| **`POST /rebuild` 用追加语义写向量库** | 高：一次重建把语料从 49 条写成 **98 条重复** | 改用 `recreate_vectorstore`（先清空 collection），并在 `create_vectorstore` 上标注"追加语义"警告；测试断言重建后 `record_count == 49` |
| **首轮客户消息没有写入记忆** | 中：后续轮次的历史缺失客户第一句话 | 首次接待前先把 user 消息写入 memory |
| 首次接待轮次返回 `route: None` | 低：前端拿不到路由类型 | 补 `route="first_contact"` / `complexity="fixed_flow"` |

## 六、HTTP 层补测（`tests/test_api_http.py`，14 条）

用 FastAPI `TestClient` 真启动应用（触发 `startup_event`：构建 Model 级语料 → 校验/加载向量库 → 初始化 Sales + Solution Agent 与 Orchestrator → 检索冒烟测试），覆盖：

| 覆盖点 | 结果 |
|---|---|
| `/`、`/health`、`/diagnostics/retrieval` | 200，记录数 49（31 室内 / 18 户外），确认跑的是 Model 级语料 |
| API Key 鉴权 | 无 Key 时 `/chat`、`/memory/*`、`/rebuild` 全部 401；带 Key 200 |
| `/chat` 首轮固定接待流程 | 200，返回 `route=first_contact` + 自我介绍 + 素材消息，并正确写入记忆（含客户首条消息） |
| `/chat` 后续轮次（无 LLM） | 200，走降级兜底返回可用话术，不会 500 |
| `/chat` SSE 流式 | `text/event-stream`，事件序列 `start → chunk → done` 正确 |
| `/memory/{id}`、`/memory/clear` | 读/清正常 |
| `/rebuild` 异步任务 | 提交后 task_id 可查询、进度可轮询、列表接口正常（修复后） |
| 静态资源 | 聊天页面可访问 |

> ⚠️ 仍未覆盖：真实 LLM（DeepSeek）输出质量、前端浏览器交互、多轮真实对话。这些需要可用网络/Key 才能验证。
