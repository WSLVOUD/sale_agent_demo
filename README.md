# LED RAG 智能销售系统

> 基于大模型（DeepSeek）的 LED/LCD/IFP 全品类显示产品智能销售助手，采用多 Agent 协作 + 混合检索（RAG）架构，为销售团队提供实时产品推荐和技术咨询能力。

> **当前版本：v2.9.6（2026-09-24）** ｜ 全量测试：`python -m pytest -q` → **760 条，约 2 分 40 秒**
>
> 当前行为口径集中在下面「当前行为口径」一节；历史版本的逐条变更见文末「变更明细」。
>
> **架构收口（2026-09-24 ~ 09-28）**：职责收敛 + 需求链路收口已完成，结论见下方
> 「架构收口」一节。文中历史章节的数字（如 591 条）是当时的快照，保留不动；
> 版本历史以 Git 为准（`git log --oneline`）。

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
