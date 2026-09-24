# 基线（Phase 0）

> 用途：架构瘦身**前**的现状快照。之后每完成一个模块迁移，都要与本文件对比；
> 数字对不上就说明动了行为，必须回退或补证据。
>
> 采集时间：2026-09-24 ｜ 提交：`00ff260`（工作区只多出计划文档本身）

## 1. 运行环境

| 项 | 值 |
|---|---|
| 解释器 | CPython **3.11.9**（全局 Python，不是 `venv`） |
| 注意 | `C:\demo-agent\venv` 有运行依赖但**没有 pytest**；跑测试用全局解释器 |
| 关键依赖 | langchain 0.3.7 / langgraph 0.2.56 / chromadb 0.5.5 / fastapi 0.115.0 / uvicorn 0.32.0 / sentence-transformers 3.3.1 / FlagEmbedding 1.4.2 / rank-bm25 0.2.2 / openai 1.55.3 / pytest 9.1.1 |
| 本地模型 | `models/BAAI--bge-m3/snapshots/master`（约 2.2GB，不联网下载） |
| 向量库 | `vectorstore/`（chroma.sqlite3 + sparse_index.json，87 条 Model 级记录，metadata v7） |

## 2. 启动方式与入口

```bash
python -m uvicorn src.api:app --port 8000      # 联调/生产都用这个（不要 --reload）
python init_vectorstore.py                     # 仅在需要重建向量库时
```

- 应用入口：`src/api.py`（FastAPI）
- 启动自检日志必须出现：
  `[Build] ProductTypeRouter=enabled（probe 'i need a display' → UNKNOWN/UNKNOWN）`
  以及 `Application startup complete.`
- 首次启动要加载本地 BGE-M3 + 向量库（实测约 15 秒）

## 3. HTTP 接口（12 个 —— API 行为基线）

| 方法 | 路径 |
|---|---|
| GET | `/`（前端页面）、`/health`、`/diagnostics/retrieval` |
| POST | `/chat`（主入口，支持 SSE 流式）、`/memory/clear` |
| GET | `/memory/{session_id}` |
| POST | `/rebuild`、`GET /rebuild`、`GET /rebuild/{task_id}`、`POST /rebuild/{task_id}/cancel` |
| GET | `/circuit-breaker/status`、`POST /circuit-breaker/reset` |

`/chat` 请求：`{session_id, question, images, messages[], client_message_ids[], turn_id, source}`；
响应：`{session_id, answer, requirement, reflection_score, products, route, complexity, first_contact_intro}`。

## 4. 代码规模（瘦身起点）

```text
src:   160 个模块
tests:  81 个测试文件 / 726 条用例
```

| 包 | 文件 | 行数 |
|---|---:|---:|
| rag | 33 | 12446 |
| dialogue | 34 | 8046 |
| agents | 22 | 6725 |
| core | 12 | 2634 |
| input | 12 | 2392 |
| engineering | 11 | 2103 |
| models | 4 | 1740 |
| vision | 7 | 1610 |
| memory | 4 | 1084 |
| observability | 4 | 917 |
| first_contact | 4 | 663 |
| tools | 5 | 522 |
| utils | 4 | 440 |
| tasks | 1 | 223 |

## 5. 测试基线

```bash
python -m pytest -q
```

| 项 | 基线值 |
|---|---|
| 收集用例 | **726** |
| 结果 | **726 passed / 0 failed** |
| 用时 | 约 2 分 35 秒 |
| 测试树 | 只有一棵 `tests/`（`pytest.ini`: `testpaths = tests`） |
| LLM 口径 | 测试环境无外网：`LLM_MAX_RETRIES=0`、`LLM_TIMEOUT_SECS=3`，失败走结构化兜底 |

## 6. 评测基线（可复现命令）

```bash
python -m eval.calculator_eval              # 计算
python -m eval.recommendation_eval          # 需求理解 / 路由 / 推荐引擎
python -m eval.retrieval_eval --limit 12    # 检索（抽样 12；全量去掉 --limit）
```

| 指标 | 基线值 |
|---|---|
| Calculator Accuracy | **1.0（14/14）** |
| Requirement Slot Accuracy | 0.7032 |
| Hard Constraint Capture | 0.5923 |
| Route Accuracy | 0.7439 |
| 逐槽位命中率 | environment 0.8871 / installation 0.8814 / viewing_distance_m 0.9655 / pixel_pitch_mm 0.9333 / purpose 0.7838 / brightness_min 1.0 / display_type 0.3611 |
| 检索 Series Recall@5 | 1.0（抽样 12） |
| 检索 Model Recall@10 | 0.9333（抽样 12） |
| 硬约束违规率 | 0.0 |
| 检索平均延迟 | 208.2 ms |

说明：本机沙箱**没有外网**，所以这些是"离线口径"（LLM 语义抽取不可用，走规则兜底），
不代表生产指标；但**同一环境下前后对比有效** —— 瘦身前后这些数字必须一致。
报告落在 `eval/reports/*.json`（已 gitignore）。

## 7. 每个阶段的对比纪律

```text
1. python -m pytest -q                       ← 全量（最重要）
2. python -m eval.calculator_eval            ← 计算零变化
3. python -m eval.recommendation_eval        ← 槽位/路由数字不下降
4. python -m eval.retrieval_eval --limit 12  ← 检索数字不下降
```

任何一项偏离基线，先当成"行为被改动"，定位清楚再继续。
