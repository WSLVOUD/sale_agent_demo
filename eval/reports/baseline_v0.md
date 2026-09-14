# Phase 0 基线报告（Baseline v0）

- 日期：2026-09-13
- 数据集：`eval/golden_dataset.json`（72 条用例）
- 语料：`data/` 下全部 `.txt`（21 chunks，9 个 LED 系列 / 49 个型号）
- 说明：这是**改造前**的现状量化，后续每个 Phase 都用同一套用例回归对比

## 1. 需求理解与路由（`recommendation_eval`）

| 指标 | 基线 | 目标 |
|------|------|------|
| Requirement Slot Accuracy | **0.526** | ≥ 0.90 |
| Hard Constraint Capture Rate | **0.518** | ≥ 0.90 |
| Route Accuracy | **0.819** | — |

逐槽位命中率：

| 槽位 | 命中率 | 问题 |
|------|--------|------|
| `viewing_distance_m` | **0.000** | 规则提取器完全不解析观看距离 —— 而它决定点间距，是推荐链路的第一个断点 |
| `display_type` | 0.436 | 只在显式出现 "LED" 时才识别，"屏幕/显示屏" 等表述漏检 |
| `purpose` | 0.536 | 多数场景词未映射到 purpose |
| `installation` | 0.580 | 固定/租赁判断不稳 |
| `environment` | 0.793 | 相对可用 |
| `pixel_pitch_mm` | 0.833 | 相对可用 |
| `brightness_min` | 1.000 | 相对可用 |

路由混淆（前几项）：`normal→normal 52`、`normal→agent 8`、`fast→fast 6`、`fast→normal 3`、`agent→normal 2`。
主要问题是本该走 AGENT 的模糊/冲突需求被判成 NORMAL。

## 2. 检索（`retrieval_eval`）

| 指标 | ranked（无点间距过滤） | filtered（与生产一致） | 目标 |
|------|------|------|------|
| Series Recall@5 | 0.991 | 0.689 | — |
| Series MRR | 0.907 | 0.778 | — |
| **Model Recall@5** | **0.415** | **0.382** | ≥ 0.80（Recall@5 口径） |
| Model Recall@10 | 0.618 | 0.488 | ≥ 0.90 |
| Model MRR | 0.422 | 0.381 | ≥ 0.80 |
| Hard Constraint Violation Rate | **0.444**（97 条违规） | 0.000 | **0.00** |
| 空结果率 | 0.056 | 0.167 | — |
| 平均延迟 | 221 ms | 224 ms | — |

**结论（与计划文档诊断一致）**：

1. 检索以 **Series 为最小单位**，Series 级召回看起来很好（99%），但 **Model 级只有 41.5%** —— 同系列型号混杂，无法稳定定位到具体型号。
2. 环境/安装方式过滤本身是生效的；44% 的违规全部是**点间距违规**——因为 `ranked` 模式按口径不套点间距过滤，检索会返回点间距不合适的型号（例如要求 P0.9~P2.0 却召回 P3.076）。
3. 一旦按生产的做法套上点间距过滤，Series 召回从 0.99 掉到 0.69、16.7% 的用例直接空结果 —— 因为过滤作用在 **chunk 级的 pitch 区间**（一个 chunk 覆盖 P1.2~P4.0），只要区间越界就整块被删掉。这是"过滤把正确型号一起删掉"的典型症状，也是 Phase 2 改成 Model 级语料后最先被修好的问题。

## 3. 工程计算（`calculator_eval`）

状态：`pending_phase_9` —— 14 条计算用例（整除、不整除、边界、超大尺寸、不同箱体/模组）已就绪，等 Phase 9 实现 `src/tools/screen_calculator.py` 后自动生效。

## 4. 附带发现（Phase 0 顺带修掉）

- `src/tools/product_tool.py`、`src/tools/search_tool.py` 从 `src.rag.retriever` 导入不存在的 `HybridSearch`，导致 `import src.tools` 直接失败（已修正为 `src.rag.fusion`）。
- `load_all_product_files()` 会把 `data/company_profile.txt` 当作产品索引进向量库，并因文中出现 "Rental LED Display" 被打上 `is_rental=True`。该文件不应进入产品语料，Phase 2 会一并解决。
- 旧向量库（约 1000 条）与实际语料（21 chunks）不一致，属于历史遗留的固定长度分块产物；基线运行已按实际语料重建。

## 5. 复现命令

```bash
python -m eval.recommendation_eval      # → eval/reports/recommendation_report.json
python -m eval.retrieval_eval           # → eval/reports/retrieval_report.json
python -m eval.calculator_eval          # → eval/reports/calculator_report.json
```
