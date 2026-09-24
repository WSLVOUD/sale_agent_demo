# 未来优化清单（Phase 12 执行期间记录，不在本轮动手）

> 依据：计划 2.0 §十六「禁止顺便优化」 —— 执行 Phase 12 时发现的"这里其实可以优化一下"，
> 一律记录到本文件，等专门一轮再处理。
>
> 记录时间：2026-09-24

## A. 架构残留（本轮已核对，未动）

| # | 位置 | 现象 | 建议 | 为什么不本轮做 |
|---|---|---|---|---|
| A1 | `src/dialogue/natural_response.py`、`src/dialogue/natural_continuation.py` | 两个"表达辅助"模块（合计约 240 行），只被 `dialogue/__init__` 与 orchestrator 使用 | 并入 `ResponseGenerator`（计划 §12-3 的"如果只是表达辅助 → 合并"） | 纯搬迁，会改动包出口与 import 面；需要单独一轮 + 全量回归 |
| A2 | `src/agents/solution/nodes/requirement.py` | 无档案时的兼容分支仍会写旧 `requirement` 字典（已标 `LEGACY-COMPAT`） | 等所有调用方都带 `RequirementProfile` 后删除该分支 | 生产已走档案路径；删除会影响"无档案旧调用"的兜底行为 |
| A3 | `src/rag/recommendation_engine.py` | 没有 `profile` 时自己解析一次（已标 `LEGACY-COMPAT`，护栏钉死只允许 1 处） | 改成返回 `REQUIREMENT_NOT_READY` | 需要先确认没有外部调用方裸调引擎 |
| A4 | `src/dialogue/product_router.py:132` | `LEDPolicy.get_next_question` 在函数内延迟导入 `agents.sales.question_planner`（dialogue → agents 反向依赖，护栏钉死只允许 1 处） | 把问题计划下沉到共享层 | 牵动 LED 链路的问题顺序，属于业务面改动 |
| A5 | `src/models/requirement.py` | 函数内延迟导入 rag 的两个辅助函数 + 一个 Gate（3 处，护栏钉死数量） | 把 `_detect_purpose` / `environment_from_purpose` 下沉到 models 或 core | 搬迁会牵动 RAG 的用途标准化逻辑 |
| A6 | `tests/` 其余功能域目录 | 计划 2.0 §12-7 要求 conversation / requirement / product / recommendation / calculation / rag / integration / regression | 逐个搬（`tests/architecture/` 已完成） | 老测试用 `__file__` 深度推算项目根，先统一注入再搬 |

## B. 行为/话术（客户实测发现，已单独修复的不在此列）

| # | 现象 | 说明 |
|---|---|---|
| B1 | LCD 入口确认后每轮重复同一句 "Let's go with LCD video wall…" | 计划 §十五「LCD 需求链留白」阶段的已知表现：入口语应当只说一次。设计 LCD 需求链时一起处理 |
| B2 | 客户对需求问题回 "no"（不是否认产品类型）时只回一句承接 | 应按计划 §3「客户答不上来 → 换下一问」处理 |
| B3 | 多轮里"客户给了裸尺寸（如 129,2cm）"的方向确认 | 现由 `size_axis` 单独确认，属于既有口径；若要改顺序需客户确认 |

## C. 文档

| # | 内容 |
|---|---|
| C1 | README 的历史章节已迁到 `docs/history/CHANGELOG_archive.md`；后续版本历史一律走 Git（`git log`），不再往 README 累积 |
| C2 | `eval/reports/*.json` 每次评测自动生成（已 gitignore），需要留档时手动复制到 `docs/refactor/` |
