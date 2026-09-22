# 评估体系（eval/）

Phase 0 建立的统一评估基线。在改动任何检索/推荐逻辑之前，先用它量化现状；
每完成一个 Phase 后重跑，用同一套用例对比是否达标。

## 文件说明

| 文件 | 作用 |
|------|------|
| `golden_dataset.json` | Golden Dataset：82 条标注用例，覆盖室内/室外、固装/租赁、会议室/教室/展厅/商场/广告/舞台/演唱会、模糊需求、多轮、多语言、指定点间距、指定尺寸、不完整与冲突需求；v2.1 追加客户决策状态边界样本（g073~g082：DELEGATED / UNKNOWN / DEFERRED / DECLINED / BLOCKED） |
| `metrics.py` | 公共指标与工具：Recall@K、MRR、Slot Accuracy、硬约束校验、产品代号解析 |
| `retrieval_eval.py` | 检索评估：Recall@5/10、MRR、硬约束违规率、延迟（Series 级 + Model 级） |
| `recommendation_eval.py` | 需求理解与推荐评估：Requirement Slot Accuracy、路由准确率、Top-1/Top-3 |
| `calculator_eval.py` | 箱体/模组计算评估：14 条计算用例，Phase 9 落地后自动生效 |
| `dialogue/run_dialogue_eval.py`、`dialogue/golden_dialogue.json` | 对话链路行为约束评测（12 个场景，独立入口） |
| `dialogue_chain_golden.json`、`naturalness_golden.json` | 同时是核心测试的数据集（`tests/test_dialogue_chain_golden.py` / `tests/test_naturalness_golden.py` 直接读） |
| `reports/` | 评测输出目录（跑一次评测才会生成，已在 `.gitignore`，不入库） |

> 2026-09-22：Phase 0 时代的 `run_eval.py` / `dataset.json` 已删除（被上面的脚本取代），
> 对应的历史报告（`baseline_v0.md` / `final_report.md` / `_smoke.json`）一并清掉。

## 运行方式

```bash
# 整体（推荐顺序）
python -m eval.recommendation_eval        # 秒级，无网络依赖
python -m eval.calculator_eval            # 秒级，无网络依赖
python -m eval.retrieval_eval             # 需要本地 BGE-M3 + 向量库（约 1~2 分钟）

# 常用参数
python -m eval.retrieval_eval --limit 10                 # 快速抽样
python -m eval.retrieval_eval --ids g025,g026            # 指定用例
python -m eval.retrieval_eval --tags outdoor             # 指定 tag
python -m eval.retrieval_eval --rebuild                  # 强制重建向量库
python -m eval.recommendation_eval --agent               # 端到端 Agent（需要 LLM 网络）
```

报告默认写入 `eval/reports/*.json`，包含汇总指标与逐条明细，便于定位回归。

## 指标口径

| 指标 | 口径 |
|------|------|
| Recall@K | 前 K 个结果覆盖了多少比例的期望 Series / Model |
| MRR | 第一个命中项的排名倒数 |
| Requirement Slot Accuracy | 期望槽位（display_type / environment / installation / purpose / viewing_distance_m / pixel_pitch_mm / brightness_min）的逐字段命中率 |
| Hard Constraint Violation | 检索或推荐结果违反硬约束（环境、安装方式、产品类型、亮度、点间距）的比例，目标 0 |
| Calculator Accuracy | 箱体列数/行数/总数、实际尺寸、模组数全部正确的比例，目标 100% |

## 两种检索模式

`retrieval_eval.py` 同时输出两种模式，用于区分"排序质量问题"和"过滤导致的漏召回"：

- `ranked`：只应用环境/安装方式/产品类型 + 亮度硬过滤 → 衡量排序质量
- `filtered`：额外应用点间距过滤（与生产 `retrieval_node` 一致）→ 衡量过滤是否可用

## 目标值（来自 `golden_dataset.json` 的 `targets`）

```
Requirement Slot Accuracy      ≥ 0.90
Recall@5                       ≥ 0.80
Recall@10                      ≥ 0.90
MRR                            ≥ 0.80
Top-1 Recommendation Accuracy  ≥ 0.85
Top-3 Recommendation Coverage  ≥ 0.95
Hard Constraint Violation      =  0.00
Calculator Accuracy            =  1.00
```
