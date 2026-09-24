# 测试说明（入口索引）

> 计划 2.0（Phase 12-9）：测试说明集中到 `docs/testing/`，README 只留"怎么跑"。

## 1. 怎么跑

```bash
python -m pytest -q                     # 全量（唯一入口，一棵树 tests/）
python -m pytest tests/architecture -q  # 架构护栏
python -m pytest tests/dialogue -q      # 对话层
python -m pytest tests/test_vision_pipeline.py -q   # 单文件
```

- 解释器：**全局 Python 3.11.9**（`venv` 里没有 pytest）。
- 测试环境无外网：`conftest.py` 设置 `LLM_MAX_RETRIES=0` / `LLM_TIMEOUT_SECS=3`，
  LLM 不可用时走结构化兜底（失败路径本身也是被测行为）。

## 2. 目录与职责

| 目录 | 内容 |
|---|---|
| `tests/architecture/` | 架构护栏：唯一需求模型、唯一产品类型入口、唯一对话决策、层间边界、API 边界… |
| `tests/dialogue/` | 对话行为：一轮一问、不重复提问、承接、话术、产品类型链路 |
| `tests/input/` | Turn 引擎：去重 / 聚合 / 会话锁 / 状态还原 |
| `tests/memory/`、`tests/replay/`、`tests/vision/` | 会话记忆、真实日志回放、图片链路 |
| `tests/test_*.py` | 集成与端到端（API / 前端聚合 / 会话重置 / Vision 全链路…） |

> 计划 2.0 Phase 12-7 的完整功能域目录（conversation / requirement / product /
> recommendation / calculation / rag / integration / regression）是**分步迁移**目标：
> 迁移必须先把"用 `__file__` 深度推算项目根"的老样板统一掉，否则会出现
> "改了目录再修报错"。当前已完成 `tests/architecture/` 的迁移，
> 其余目录按同样方式逐个搬（见 `docs/plans/future_optimization.md`）。

## 3. 与基线对比

```bash
python -m eval.calculator_eval            # 计算（离线可跑）
python -m eval.recommendation_eval        # 需求理解 / 路由 / 推荐
python -m eval.retrieval_eval --limit 12  # 检索（抽样，快）
```

基线数字见 [`docs/refactor/baseline.md`](../refactor/baseline.md)。
