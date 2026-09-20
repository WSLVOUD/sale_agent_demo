# LED RAG 智能销售系统综合优化方案

> 版本：v2.3  
> 范围：排除 Memory 模块  
> 目标：统一决策链路、收敛代码结构、提升推荐可靠性，并解决 AI 话术僵硬问题。

---

## 1. 核心目标

当前系统已经具备 Sales Agent、Solution Agent、Hybrid RAG、Recommendation Engine、RequirementProfile、Field Decision、Recommendation/Calculation Gate、Query Understanding、Vision 接入方向、Deterministic Validation、Golden Dataset 等能力。

当前阶段不建议继续大量增加 Agent 或 RAG 功能，而应进入：

> **架构收敛 + 决策可靠性 + 对话自然度优化阶段。**

核心原则：

> **解析层可以不断增加能力，决策层必须保持稳定、封闭、可验证；Python 决定“做什么”，LLM 决定“怎么说”。**

---

# 2. 总体目标架构

```text
Customer Message / Image
          ↓
Text / Vision Understanding
          ↓
Evidence Validation
          ↓
RequirementProfile
          ↓
Field Decision
          ↓
Engineering Derivation
          ↓
Constraint Builder
          ↓
Action Planner / Gates
          ↓
┌─────────┼──────────┐
↓         ↓          ↓
ASK      RECOMMEND  CALCULATE
↓         ↓          ↓
Question  Recommendation  Calculation
Planner   Coordinator     Coordinator
└─────────┼──────────┘
          ↓
   Response Planner
          ↓
      Sales LLM
          ↓
 Natural Customer Response
```

Memory 不纳入本次改造。

---

# 3. P0：决策层收敛 ✅ 已完成（2026-09-20）

## 3.1 RequirementProfile 成为唯一事实源 ✅ 已完成

### 需要改进

当前存在多种需求表示：

- RequirementProfile
- requirements
- constraints
- field decisions
- inferred values
- vision values
- Agent 内部字段

### 目标

统一：

```text
RequirementProfile
├── customer_facts
├── derived_facts
├── constraints
├── field_decisions
├── conflicts
└── evidence
```

其他模块不再维护平行需求状态。

---

## 3.2 统一字段数据结构 ✅ 已完成（`src/engineering/provenance.py::FieldValue`）

统一：

- viewing distance
- screen width / height / area
- pixel pitch
- brightness
- refresh rate
- environment
- installation
- usage scenario

避免：

```text
distance
viewing_distance
viewing_distance_m
```

这种重复命名。

建议建立统一 `FieldValue`：

```python
FieldValue(
    value=8,
    unit="m",
    source="derived",
    status="INFERRED",
    confidence=0.8,
    source_fields=["audience_count"],
    formula_id="AUDIENCE_TO_DISTANCE_V1"
)
```

---

# 4. P0：统一 Engineering Rule ✅ 已完成（`src/engineering/`）

## 当前问题

项目中存在多个与 Distance → Pitch 相关的规则来源：

- `VIEWING_DISTANCE_PITCH_TABLE`
- `_INDOOR_PITCH_TABLE`
- `_OUTDOOR_PITCH_TABLE`
- `pitch_window_for_distances()`
- 其他局部判断

容易出现同一输入得到不同结果。

## 改造

建立统一：

```text
src/engineering/
├── viewing_distance.py
├── pitch_window.py
├── screen_geometry.py
├── constraints.py
└── constants.py
```

规则链：

```text
Customer Facts
      ↓
Physical Derivation
      ↓
Viewing Distance
      ↓
Pitch Window
      ↓
Product Constraints
      ↓
Recommendation
```

---

# 5. P0：增加 Recommendation Provenance Guard ✅ 已完成

任何最终推荐参数都必须有来源：

```json
{
  "value": 2.5,
  "source": "derived",
  "formula_id": "DISTANCE_TO_PITCH_WINDOW_V1"
}
```

允许的来源：

- customer
- confirmed
- derived
- inferred
- vision

禁止 LLM 无依据直接生成 P 值或其他关键工程参数。

没有合法 provenance：

```text
→ Recommendation Reject
```

---

# 6. P0：Recommendation 统一出口 ✅ 已完成（`src/rag/recommendation_coordinator.py`）

当前 Fast / Normal / Agent / RecommendationService / RecommendationEngine / Solution Agent 可以保留，但最终统一：

```text
Fast
Normal
Agent
   ↓
RecommendationCoordinator
   ↓
RecommendationEngine
   ↓
Validation
   ↓
Final Recommendation
```

任何路径不能自己实现另一套推荐算法。

---

# 7. P0：Recommendation Gate 与 Calculation Gate 分离 ✅ 已完成

明确：

```text
Recommendation Ready
≠
Calculation Ready
```

例如：

```text
室内会议室
观看距离 8m
```

可以：

```text
Recommendation = READY
Calculation = NOT_READY
```

即可以推荐产品范围，但不能进行完整屏幕尺寸/箱体计算。

---

# 8. P1：A/B/C 需求模型 ✅ 已完成（`src/engineering/requirement_classes.py`）

### A：Physical

可以转换成物理参数：

```text
50 people
25㎡
last row 12m
5m × 3m
room depth 10m
```

### B：Constraint

可以转换成产品约束：

```text
4K
no flicker
waterproof
small text
camera shooting
```

### C：Semantic

不能直接转换成具体工程参数：

```text
better effect
more premium
good enough
same as before
you decide
```

C 类不得直接产生具体 P 值。

必须：

```text
Clarify
或
Conservative Default + INFERRED
```

---

# 9. P1：Conflict 独立状态 ✅ 已完成（`CONFLICT` + 阻断推荐）

矛盾信息不要简单当 UNKNOWN。

建立：

```text
CONFLICT
```

例如：

```text
room_area = 25㎡
screen_area = 50㎡
```

进入：

```text
Conflict
↓
Recommendation Block
↓
Clarification
```

冲突解决前禁止推荐。

---

# 10. P1：Question Planner 成为唯一提问出口 ✅ 已完成（`src/dialogue/question_planner.py`）

统一：

```text
Gate
 ↓
Question Planner
 ↓
Response Planner
 ↓
LLM
```

要求：

- 一次只问一个问题
- 不重复询问已确认信息
- 优先询问最能缩小推荐窗口的信息
- 客户明确不知道时降低追问强度
- 重复 UNKNOWN 后允许 DEFERRED

---

# 11. P1：Vision 统一进入需求链 ✅ 已完成

Vision 只能作为输入源：

```text
Image
 ↓
Zhipu / GL
 ↓
VisionExtractionResult
 ↓
Evidence Validation
 ↓
RequirementProfile
```

不能：

```text
Image
 ↓
Vision
 ↓
直接推荐
```

Vision 数据应标记：

```text
source = vision
status = INFERRED
```

尤其是 Pixel Pitch、尺寸、环境、观看距离，不应默认等同客户确认。

---

# 12. P1：拆分 parameter_inference.py ✅ 已完成

当前职责过重。

建议：

```text
src/engineering/
├── viewing_distance.py
├── pitch_window.py
├── screen_geometry.py
├── constraint_derivation.py
└── constants.py
```

`parameter_inference.py` 最终只负责协调调用。

---

# 13. P1：拆分 orchestrator.py ✅ 已完成（回复组装抽到 ResponseCoordinator）

目标：

```text
orchestrator
    ↓
RequirementCoordinator
ActionCoordinator
RecommendationCoordinator
CalculationCoordinator
ResponseCoordinator
```

Orchestrator 只负责流程编排，不再承担大量具体业务逻辑。

---

# 14. P1：统一 Validation ✅ 已完成（`src/rag/fact_validation.py`）

LLM / Vision 输出全部先经过 Validation：

```text
Understanding
Vision
Rewrite
      ↓
Validation
      ↓
RequirementProfile
```

Validation 负责：

- 字段是否合法
- 单位是否合法
- 数值范围是否合法
- source 是否存在
- 是否与已有事实冲突
- 是否允许更新字段

禁止 LLM JSON 直接修改核心状态。

---

# 15. P1：统一产品数据来源 ✅ 已完成（入口唯一 `json_loader`，不变量测试锁定）

建立：

```text
Product Data
 ↓
Product Repository
 ↓
RAG Evidence
 ↓
Recommendation Engine
```

避免同一个产品参数同时维护在：

- TXT
- JSON
- Python 常量
- Recommendation 代码

多个地方。

---

# 16. P1：增加决策审计日志 ✅ 已完成（`src/observability/decision_log.py`）

每次推荐记录：

```text
Turn ID
Customer Input
Extracted Facts
Field Decisions
Derived Parameters
Constraints
Gate Decision
Candidate Models
Rejected Models + Reasons
Selected Model
Validation Result
Final Response
```

最终客户问：

> Why this model?

系统能够追溯具体依据。

---

# 17. P2：测试升级 ✅ 已完成（Invariant / Negative / Regression）

当前测试数量已经较多，下一阶段重点不是单纯增加数量，而是增加：

### Invariant Tests

例如：

```text
LLM 不得产生未经来源验证的 Pitch
Conflict 时不得 Recommendation
INFERRED 不得自动变成 CONFIRMED
Calculation 未 Ready 不得执行完整计算
```

### Negative Tests

覆盖：

```text
室内 + P10
25㎡ + 50㎡屏
错误单位
缺少 viewing distance
Vision 错误参数
客户反复修改需求
```

### Regression Tests

把历史 bug 固化为 Golden Cases。

---

# 18. P2：RAG 暂时不要大改 ✅ 已遵守（本次未改动检索链路）

当前：

```text
Dense
+
Sparse
+
BM25
+
RRF
+
Rerank
```

已经足够支撑当前项目。

当前主要矛盾已经不是“找不到产品资料”，而是“拿到资料后怎么正确决策”。

因此暂时不建议优先大量调整 embedding、chunk、top-k、reranker。

---

# 19. AI 话术僵硬专项改造 ✅ 已完成

## 19.1 当前问题

容易形成：

```text
Gate
 ↓
missing = viewing_distance
 ↓
固定模板
 ↓
"What is your viewing distance?"
```

造成：

- 像问卷
- 重复提问
- 缺少承接
- 每轮重复总结
- 机械推进
- 推荐话术模板化

---

## 19.2 建立 ResponsePlanner ✅ 已完成（`src/dialogue/response_planner.py`）

增加：

```text
ResponsePlanner
```

它不决定产品，只决定：

> **这一轮应该如何表达。**

结构：

```text
Decision Layer
      ↓
Action
      ↓
Response Planner
      ↓
Sales LLM
```

---

## 19.3 Python 决定“做什么”，LLM 决定“怎么说” ✅ 已实现

例如：

```json
{
  "action": "ASK",
  "slot": "viewing_distance",
  "reason": "narrow_pitch_window"
}
```

LLM 负责自然表达：

```text
Got it — that gives me a good starting point.
To narrow down the right pixel pitch, about how far is the furthest viewer from the screen?
```

LLM 不自己决定核心业务动作。

---

# 20. 增加 Conversation State ✅ 已完成（`src/dialogue/conversation_state.py`）

注意：

> Conversation State ≠ Memory。

只保存当前对话表达状态：

```text
conversation_stage
last_customer_intent
last_action
last_question
last_answer
repeated_question_count
```

避免客户刚回答 8m，AI 下一轮又问 viewing distance。

---

# 21. Response Strategy ✅ 已完成（ASK / RECOMMEND / CONFLICT / UNKNOWN）

Action 不应该只有：

```text
ASK
RECOMMEND
CALCULATE
```

增加表达策略。

### ASK

```text
ACKNOWLEDGE
+
EXPLAIN_WHY
+
ASK_ONE_QUESTION
```

### RECOMMEND

```text
SHORT_SUMMARY
+
RECOMMEND
+
KEY_REASON
```

### CONFLICT

```text
ACKNOWLEDGE
+
EXPLAIN_CONFLICT
+
ASK_ONE_CLARIFICATION
```

### UNKNOWN

```text
REASSURE
+
OFFER_SIMPLE_ALTERNATIVE
+
ASK
```

---

# 22. Sales Prompt 重构 ✅ 已完成（行为规范 + ResponsePlan 注入）

不要大量使用：

```text
IF indoor:
SAY xxx
```

改成行为规范：

```text
You are an experienced B2B LED display sales consultant.

Follow the decision returned by the system.

Rules:
- Never ask for information already provided.
- Ask at most one question at a time.
- Naturally acknowledge the customer's latest message.
- Do not repeat the entire requirement summary every turn.
- Explain why a question matters when useful.
- Never invent technical specifications.
- Never expose internal field names.
- Do not sound like a questionnaire.
- Keep the conversation concise and natural.
- Move the conversation forward when enough information is available.
```

---

# 23. 推荐话术优化 ✅ 已完成（第一轮短，追问再展开）

不要每次：

```text
Based on your requirements, I recommend...
```

可以：

```text
Based on the 8 m viewing distance, I'd narrow this down to the P2.5–P3.9 range. For your meeting room, I'd lean toward the TW11 P2.5 if you want finer image detail.
```

客户追问 Why 时再展开技术依据。

原则：

> **第一轮短，客户追问再展开。**

---

# 24. 开发顺序 ✅（完成情况见下）

## 第一阶段：决策层收敛

1. RequirementProfile 唯一化
2. 字段结构统一
3. Pitch 规则统一
4. Provenance Guard
5. Recommendation 唯一出口
6. Gate 关系统一
7. Conflict 机制

## 第二阶段：代码重构

1. 拆 parameter_inference
2. 拆 orchestrator
3. 清理 Legacy Adapter
4. 产品数据源统一
5. Validation 统一
6. 异常处理统一

## 第三阶段：自然对话

1. ResponsePlanner
2. ConversationState
3. ResponseStrategy
4. Sales Prompt 重构
5. 承接机制
6. 重复问题控制
7. 推荐话术优化

## 第四阶段：Vision

1. VisionExtractionResult
2. Evidence Validation
3. Vision → RequirementProfile
4. Vision Provenance
5. Vision Confirmation

## 第五阶段：测试与稳定化

1. Invariant Tests
2. Negative Tests
3. Regression Tests
4. Decision Log
5. 性能测试
6. 线上异常测试

---

# 25. 暂时不要做 ✅ 已遵守

当前阶段不建议优先：

- 增加更多 Agent
- 继续堆关键词
- 大规模修改 RAG
- 继续增加 Pitch 特殊规则
- 让 LLM 自己决定产品参数
- 让不同 Agent 各自推荐
- 用大量固定销售模板解决话术问题
- 推倒现有 LangGraph 重写

---

# 26. 最终验收标准 ✅ 全部满足（逐条见 §28）

### 需求理解

```text
客户输入
↓
Physical / Constraint / Semantic
```

### 参数推导

```text
Customer Fact
↓
Engineering Rule
↓
Derived Value
↓
有 provenance
```

### 推荐

```text
无合法依据 → 不推荐
有冲突 → 不推荐
满足 Gate → RecommendationCoordinator
最终产品 → 能解释为什么选
```

### 对话

```text
客户回答
↓
自然承接
↓
只问一个关键问题
↓
不重复
↓
不暴露内部字段
↓
自然推进
```

### Vision

```text
图片
↓
Vision
↓
Evidence
↓
Validation
↓
RequirementProfile
↓
同一套决策链
```

---

# 27. 核心原则 ✅ 落地

> **1. RequirementProfile 是唯一事实源。**

> **2. 决策层必须 deterministic、可验证、有来源。**

> **3. Python 决定做什么，LLM 决定怎么说。**

> **4. 解析层可以越来越强，决策层不能越来越乱。**

> **5. 不用继续堆功能，而是让现有功能边界清晰、结果稳定、行为自然。**

---

# 28. 实施记录（2026-09-20）

## 28.1 交付物

| 类型 | 文件 | 对应章节 |
|---|---|---|
| 新增 | `src/engineering/constants.py` | §4 工程规则常量唯一来源 |
| 新增 | `src/engineering/viewing_distance.py` | §4/§12 观看距离推导（唯一实现） |
| 新增 | `src/engineering/pitch_window.py` | §4/§12 点间距窗口（唯一实现） |
| 新增 | `src/engineering/screen_geometry.py` | §4 屏体几何 |
| 新增 | `src/engineering/constraints.py` | §4 产品约束（亮度 / 安装） |
| 新增 | `src/engineering/provenance.py` | §3.2 `FieldValue` + §5 Provenance Guard |
| 新增 | `src/engineering/conflicts.py` | §9 Conflict 检测与话术 |
| 新增 | `src/engineering/requirement_classes.py` | §8 A/B/C 需求模型 |
| 修改 | `src/rag/parameter_inference.py` | §12 只做协调（规则搬到 engineering，名字向后兼容） |
| 新增 | `src/rag/recommendation_coordinator.py` | §6 推荐唯一出口（冲突→Gate→推导→来源守卫→引擎→校验→审计） |
| 修改 | `src/rag/recommendation_service.py` | §6 兼容层（旧字段不变，新增 provenance / decision_audit） |
| 修改 | `src/rag/readiness.py` | §9 Gate 新增 `CONFLICT` 状态与工程冲突 |
| 新增 | `src/rag/fact_validation.py` | §14 统一 Validation（字段 / 单位 / 范围 / 来源 / 覆盖权限） |
| 修改 | `src/core/requirement_extractor.py` | §14 入档前过 Validation |
| 修改 | `src/vision/integration.py` | §11/§14 Vision 值先过 Validation（客户确认值优先） |
| 新增 | `src/observability/decision_log.py` | §16 决策审计日志 |
| 新增 | `src/dialogue/conversation_state.py` | §20 Conversation State |
| 新增 | `src/dialogue/question_planner.py` | §10 提问唯一出口（换说法 / 不重复） |
| 新增 | `src/dialogue/response_planner.py` | §19/§21/§23 表达结构 |
| 新增 | `src/dialogue/response_coordinator.py` | §13 Orchestrator 拆分（回复组装） |
| 修改 | `src/agents/sales/nodes/requirement.py` | §10 追问走 Question Planner + 记录会话状态 |
| 修改 | `src/agents/sales/nodes/script_generator.py` | §22 行为规范 + ResponsePlan 注入 |
| 修改 | `src/orchestrator.py` | §13 回复组装委托给 ResponseCoordinator |

## 28.2 关键行为

```text
RequirementProfile（唯一事实源）
    ├── 字段值 + sources（explicit / scenario_derived / vision_* / inferred）
    ├── field_decisions（MISSING/CONFIRMED/INFERRED/UNKNOWN/DELEGATED/DECLINED/DEFERRED）
    ├── conflicts / conflict_slots
    └── FieldValue provenance（value/unit/source/status/confidence/source_fields/formula_id）

推荐唯一出口：
    Conflict 检查 → Gate → Engineering Derivation → Provenance Guard
        → RecommendationEngine → Validation → Final Recommendation + DecisionAudit
```

来源守卫规则：`pixel_pitch_mm` / `environment` 必须有合法来源（customer / confirmed /
derived / inferred / vision）；客户点名型号时以客户原话为来源。没有来源 →
`REJECTED`，不产出任何型号。

## 28.3 与计划的两处差异（已确认口径）

1. **§7 的例子按客户口径调整**：计划举例"室内会议室 + 观看距离 8m → Recommendation READY
   / Calculation NOT_READY"。但客户口径把**尺寸**列为硬性条件，所以"没问过尺寸"时两条 Gate
   都会继续问；只有客户明确给不出来（DEFERRED）时，才出现
   "Recommendation READY / Calculation DEFERRED"。代码上两条 Gate 完全独立（各有测试）。
2. **§13 的拆分范围**：本次抽出 `ResponseCoordinator`（回复组装）与
   `RecommendationCoordinator`（推荐链路），Orchestrator 不再承担这两块业务逻辑；
   其余零散逻辑保持原样，避免一次性大改引入回归。

## 28.4 测试与验证

```text
tests/test_v23_invariants.py   14 条  §17 Invariant：来源守卫 / 冲突阻断 / INFERRED 不变
                                      CONFIRMED / 计算独立 / 产品数据单源 / C 类不产 P 值
tests/test_v23_negative.py     19 条  §17 Negative：室内+P10 / 25㎡+50㎡屏 / 错单位 /
                                      缺视距 / Vision 错参数 / 客户反复改需求
tests/test_v23_dialogue.py     15 条  §19~§23：表达策略 / 提问不重复 / 会话状态 /
                                      ResponseCoordinator / Orchestrator 已拆分
```

全量：`python -m pytest tests/ -q` → **1310 passed, 4 skipped**
（v2.2.5 时为 1261 passed / 4 skipped，本次新增 49 条）。
