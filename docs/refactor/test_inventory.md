# 测试清单与重组方案（Phase 9）

> 现状：`tests/` 一棵树，86 个文件 / 726 条用例（`pytest.ini`: `testpaths = tests`）。
> 计划要求先**按功能分类**，不急着删；本文件给出分类结果与迁移方案。

## 1. 分类结果（按功能）

| 目标目录 | 包含的现有文件（示例） | 条数占比 |
|---|---|---|
| `product/`（产品类型与链路入口） | `dialogue/test_v293_product_type_router.py`、`dialogue/test_v294_type_*.py`（4 个）、`dialogue/test_v294_lcd_entry.py`、`dialogue/test_v294_vision_product_type.py`、`test_architecture_product_type.py` | 产品类型判断 / 锁定 / 语境 / 图片判断 |
| `conversation/`（对话行为） | `dialogue/test_ack_streak.py`、`test_one_question_per_turn.py`、`test_no_duplicate_question.py`、`test_offtopic_intent.py`、`test_final_response_single_output.py`、`test_natural_response_golden.py` 等 `dialogue/*`（30+） | 一轮一问 / 承接 / 自然措辞 / 单一出口 |
| `requirement/`（需求采集） | `dialogue/test_profile_slots.py`、`test_environment_first.py`、`test_answer_wrong_slot.py`、`test_whole_sentence_understanding.py`、`test_environment_not_reasked.py`、`input/test_continuous_information.py` | 槽位、答非所问、重复提问 |
| `recommendation/` | `test_new_questions.py`、`dialogue/test_explicit_recommendation_request.py`、`test_recommendation_request_and_price_preference.py`、`dialogue/test_quotation_request.py` | Gate / 推荐请求 / 报价 |
| `calculation/` | `test_screen_calculator.py`、`dialogue/test_pitch_resolution_context.py` | 箱体 / 模组 / 点间距解析 |
| `vision/` | `test_vision_pipeline.py`、`vision/test_image_payload_normalisation.py` | 图片抽取 → 档案 |
| `integration/` | `test_api_http.py`、`test_message_aggregation_chain.py`、`test_message_aggregator.py`、`test_frontend_turn_merge.py`、`test_session_reset.py`、`test_session_isolation.py`、`input/test_turn_engine_concurrency.py`、`memory/test_history_window.py`、`replay/test_real_log_replay.py` | 端到端 / Turn 引擎 / 会话 |
| `regression/` | `dialogue/test_v26_regression_cases.py`、`test_v27_*`（5）、`test_v28_plan_items.py`、`test_v29_plan_items.py`、`test_v291_others_cleanup.py`、`test_v292_greeting_entry.py` | 历史 bug 回归（保留，不删） |
| `architecture/`（本次新增） | `test_architecture_requirement_model.py`、`test_architecture_product_type.py`、`test_architecture_turn_decision.py`、`test_architecture_boundaries.py`、`test_architecture_agents_and_recommendation.py` | 瘦身期护栏（唯一的模型 / 入口 / 边界） |

## 2. 为什么不立刻搬文件（重要）

`tests/` 下大量文件靠 **`__file__` 的目录深度**自己推算项目根：

```python
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
```

一旦把文件搬到 `tests/product/`、`tests/conversation/` 这类**不同深度**的目录，
这些文件算出来的 `project_root` 就会错位 → 大量用例直接 import 失败。

所以搬迁必须**先统一路径注入方式**，再动文件（否则就是"改了目录、再修报错"，
正是计划 §2 禁止的路径）。

## 3. 迁移方案（按计划 §18 的五步走）

```text
第一步：统一路径注入
        把每个文件里的 sys.path 样板删掉，改用 conftest.py 里已有的
        project_root_path / data_dir 夹具（或 pytest 的 rootdir 自动注入）
        —— 这一步不改任何测试逻辑，只改 import 前置。
第二步：就地分组（先不跨目录）
        在 tests/ 下先建出目标目录，把文件**移动**后再跑全量。
第三步：跑全量 + 与基线对比（726 → 只增不减）
第四步：合并重复用例（例如多处"重新来 / 换产品"的说法表）
第五步：确认没有引用后，才允许删除历史用例
```

**本次决定**：Phase 9 完成**分类 + 方案**，物理搬迁暂缓。
原因（对应计划 §29）：搬迁是纯机械但面很广的改动，收益是"目录更整齐"，
而风险是"路径样板错位导致用例静默失败"；在无法逐条证明行为不变之前先不动。
等路径注入统一完成后（第一步本身就是一次独立、可验证的改动），再做第二、三步。

## 4. 必须保留的 Regression 清单（计划 §21）

| 场景 | 现有锚点 |
|---|---|
| 简单需求一轮一个问题 | `tests/dialogue/test_one_question_per_turn.py` |
| 一次给多个信息不重复问 | `tests/test_message_aggregation_chain.py`、`tests/input/test_collect_messages.py` |
| 答非所问仍吸收 | `tests/dialogue/test_answer_wrong_slot.py` |
| LED / LCD / IFP 链路 | `tests/dialogue/test_v293_product_type_router.py`、`tests/dialogue/test_v294_*` |
| 图片 → 档案 → 原业务链 | `tests/test_vision_pipeline.py` |
| 推荐受 Gate 控制 | `tests/test_new_questions.py`、`eval/recommendation_eval.py` |
| 计算受 Calculation Gate 控制 | `tests/test_screen_calculator.py`、`eval/calculator_eval.py` |
