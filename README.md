# LED RAG 智能销售系统

> 基于大模型（DeepSeek）的 LED/LCD/IFP 全品类显示产品智能销售助手，采用多 Agent 协作 + 混合检索（RAG）架构，为销售团队提供实时产品推荐和技术咨询能力。

> **版本标签：v2.9.6（2026-09-24，历史版本）**。下方 2026-09-28 的 810 条测试结果也是历史快照，不代表当前工作树。
>
> 当前架构与本轮真实验收结果见下方「当前架构与验收」；历史架构收口和指标保留在后文，便于追溯。

## 当前架构与验收（2026-10-09）

本轮按职责边界拆分模块，不改变 LED/LCD/IFP 路由、唯一推荐 Gate、API 契约或最终回复出口。

| 模块 | 当前职责 |
|------|----------|
| `src/orchestrator.py` / `src/orchestrator_state.py` | Turn 编排与对话状态服务 |
| `src/agents/sales/nodes/requirement.py` | 需求流程协调；对话意图、确认回应、上下文维护及 LCD 适配在相邻辅助模块 |
| `src/rag/query_understanding.py` | 查询理解协调；语言、测量解析、槽位抽取及检索查询在独立模块 |
| `src/rag/readiness.py` | 唯一推荐准入与计算准入 Gate；尺寸提示格式化位于 `readiness_measurements.py` |
| `src/dialogue/product_type_router.py` | 唯一 `route_display_type()` 决策入口；LCD / IFP 下一步由 `lcd_decision.py` 统一决定 |
| `src/rag/reply_composer.py` | 兼容入口与必要协调；语言安全、产品事实、商业回答和兜底文案各归其职责模块 |
| `src/api.py` | HTTP 路由、参数校验与服务调用；向量库校验/复用、会话快照、重建任务分别位于 `vectorstore_service.py`、`session_state_service.py`、`rebuild_service.py` |
| `tests/architecture/` | 保护唯一决策入口、层间边界和核心架构契约；行为回归仍由业务测试验证 |

**多屏维护风险（本轮禁改）**：`src/rag/multi_screen.py` 中仍有两个同名 `_share_common_facts()` 定义，后一个会覆盖前一个。本轮未修改或迁移该模块；如需处理，应另开专项并补跑多屏回归。

### 本轮实测

- `python -m pytest -q`：**695 passed, 1 warning**（501.12 秒）。唯一警告是 LangGraph 提醒未来版本会调整 `allowed_objects` 默认值；本轮未涉及该配置。
- `tests/test_new_questions.py` 和 `tests/test_question_phrasing.py` 的旧安装问句断言已按现行客户口径更新为“固定安装 vs 快装快拆”，并明确禁止回复中出现 “rental”；生产文案未改动。
- `python -m eval.recommendation_eval`：需求槽位准确率 **0.9914**，硬约束捕获 **0.9896**，派生捕获 **0.4003**，产品类型准确率 **1.0**。
- `python -m eval.calculator_eval`：**14/14（1.0）**。
- `python -m eval.retrieval_eval`：82 cases；过滤后 Recall@10 **0.9511**、MRR **0.9267**、Model Recall@10 **0.9286**、硬约束违规 **0**。
- `python -m eval.dialogue.run_dialogue_eval`：12 cases；一轮一问、action 准确率及问题槽位准确率均为 **1.0**，评估通过。

## 架构收口（2026-09-24 ~ 09-28）

目标：**不改业务行为，只收敛职责**。三轮收口的结果如下。

### 一、职责收敛（架构瘦身）

```text
1. 产品类型词汇只保留一处定义（turn_kind）；ProductTypeRouter 是唯一类型决策入口
2. "一轮最多一个问题"的预算只保留一处定义（action.MAX_QUESTIONS_PER_TURN）
3. 删除零引用的旧适配层 / 旧入口：
   core/sales_requirement_adapter.py、core/solution_requirement_adapter.py、
   input/turn_payload.py（旧 Turn 合并入口）、memory/enhanced.py
4. 新增架构护栏测试 tests/architecture/（70 条）：唯一需求模型 / 唯一产品类型入口 /
   唯一对话决策 / 层间边界（RAG 不碰对话、计算层独立、Vision 只抽需求）/
   Sales 与 Solution 共用同一套需求系统 / 推荐单一入口 / API 不执行业务
```

### 二、需求链路收口（Solution 不再理解需求）

```text
RequirementExtractor → RequirementProfile（唯一真值）
    → ProductTypeRouter（唯一产品类型判断）
    → ParameterInference（唯一工程推断出口，档案优先）
    → Recommendation Gate（唯一推荐准入）
    → Solution Agent（只执行方案）
    → RAG / Recommendation / Calculator → Validation → Response

Solution 侧删掉的东西：
  · understand_node 改为**纯 Adapter**（读档案 → 确定性 Gate → 返回；无档案标记
    REQUIREMENT_NOT_READY）—— 旧文件 663 → 268 行
  · 删除 Solution 内部 LLM 需求提取（_build_requirement_prompt / _parse_requirement_response）
  · 删除 infer_display_type()（"IFP 否则 LED"的默认判断）
  · clarify_node 只保留"档案 → Gate → 一个问题"，删掉按面积推视距/尺寸、
    按室内外推亮度、按点间距推分辨率等与 parameter_inference 重复的启发式
```

### 三、路由收口（删除 Fast 业务路由）

```text
删除：solution/runner.py 里的 classify_complexity + QueryRoute.FAST / NORMAL / AGENT 三层分流
保留：结构化产品查询能力（型号 / 点间距 / 亮度 / 防护等级）
改名：src/rag/fast_path.py → src/rag/structured_product_query.py
接线：Solution 直接判断"是不是结构化产品查询"并调用它（route = product_query），
      会话里已有场景级需求时不走这条（避免绕过"环境+视距→点间距"规则表）
```

### 四、事实模型（字段来源可解释）

```text
RequirementProfile 每个字段都带来源，归一到五类口径：
    CUSTOMER_EXPLICIT  客户明确说的
    SCENARIO_DERIVED   场景 / 图片直接可见推出来的
    SYSTEM_INFERRED    算法估算 / 图片推测
    DEFAULT            系统默认值
    UNKNOWN            没有值 / 没有依据
（fact_class() / RequirementProfile.fact_source() / fact_sources()）
评估据此执行"显式错才扣分"：合法推断记为 fp_derived，不计入错误。
```

### 五、评估体系（指标真正测生产链路）

```text
· Golden Dataset 标签按"客户原话证据"拆分：hard（客户明确）vs derived（推断+默认）、
  slots vs slots_derived —— 此前 244 个字段被误标成"客户硬约束"，是指标偏低的真因
· 新增逐字段 TP / FP / FN / Accuracy / Recall（7 个字段）
· Product Type Accuracy：直接用 ProductTypeRouter 判分（IFP 按 LCD 子类型口径计对）
· fast / normal / agent 降级为 legacy_route 历史字段，不再作为质量门槛
· --agent 端到端评测走生产链路（先建 RequirementProfile 再调 Solution），
  Top-1 / Top-3 只统计 Gate READY 子集，需追问单独统计
```

### 六、验收（2026-09-28 实测）

```text
指标                       收口前      现在      目标
Requirement Slot Accuracy  0.7032  →  0.9914    ≥0.90 ✅
Hard Constraint Capture    0.5923  →  0.9896    ≥0.90 ✅
Product Type Accuracy      0.8333  →  1.0000    ≥0.90 ✅
Calculator Accuracy        1.0     →  1.0       =1.00 ✅
Hard Constraint Violation  0       →  0         =0    ✅

python -m pytest -q                → 810 passed
python -m eval.recommendation_eval → 上述 Slot / Hard / ProductType
python -m eval.calculator_eval     → 1.0（14/14）
python -m eval.retrieval_eval      → series@5 1.0 / model@10 0.9333 / MRR 0.6708 / 违规 0
```
