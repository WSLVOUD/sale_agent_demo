# LED RAG 对话决策与 ResponseGenerator 架构收口改造计划

> 版本：v1.0  
> 目标：彻底收口 `QuestionPlanner / QuestionFlow / DialoguePolicy / script_generator / ResponseGenerator` 的职责，解决多个模块同时决定“问什么”、多个模块生成最终回复造成的决策冲突、问题错位和话术僵硬问题。  
> 仓库：`WSLVOUD/sale_agent_demo`

---

## 进度看板

| Phase | 内容 | 状态 | 完成日期 | 证据 |
|---|---|---|---|---|
| 1 | 建立唯一决策对象（审计 + 映射 + 一致性测试） | ✅ 已完成 | 2026-09-23 | `tests/dialogue/test_action_consistency.py`（5 条）；全量 596 passed |
| 2 | QuestionPlanner 降级为候选生成 | ⏳ 待做 | | |
| 3 | QuestionFlow 收口为状态管理 | ⏳ 待做 | | |
| 4 | ResponseContext 重构（QuestionSpec） | ⏳ 待做 | | |
| 5 | ResponseGenerator 唯一出口 | ⏳ 待做 | | |
| 6 | script_generator 瘦身 | ⏳ 待做 | | |
| 7 | 旧回复链退出（先替换 → 测试 → 日志 → 再删除） | ⏳ 待做 | | |

### ✅ Phase 1：建立唯一决策对象（2026-09-23 完成）

**1）审计：谁在写这四个"决策键"**（`rg` 统计 + 人工分类）

| 决策键 | 出现处 | 真正的决策点 | 派生 / 兼容（不算决策） |
|---|---|---|---|
| `pending_question` | 20 | `agents/sales/nodes/requirement.py:1298`、`agents/solution/nodes/requirement.py:296/310/531/560/634`、`core/sales_requirement_adapter.py:62` | `runner.py` 透传、`orchestrator.py:685/1055/1060`（收口与闸门改写） |
| `pending_slot` | 10 | `agents/sales/nodes/requirement.py:1299`、`core/sales_requirement_adapter.py:61` | `runner.py` 透传、`orchestrator.py:586/1056/1061` |
| `next_action` | 61 | `nodes/classify.py`（6）、`nodes/requirement.py`（6）、`nodes/router.py`（1）、`script_generator.py`（**20 处** if/else 分支） | `runner.py`、`orchestrator.py`（first_contact / 路由分支）、solution 侧 4 处 |
| `should_generate_solution` | — | `agents/sales/nodes/requirement.py`（Gate 结果） | `script_generator.py` 多处只读判断 |

**结论：本轮之前有 3.5 个决策中心** —— ① QuestionPlanner/QuestionFlow（requirement 节点）、
② DialoguePolicy（`dialogue_action`）、③ script_generator 的 20 处 `next_action` if/else、
④ 编排器收口层（重复提问闸门 + 兼容字段）。这正是计划 §2 描述的现象。

**2）重复决策点（Phase 1 实测抓到的真 bug）**

```text
DialoguePolicy:  ASK(viewing_distance)     ← 决策对象
销售层准备:       pending_slot=pixel_pitch ← 第二个决策
收口结果（修复前）: 最终问了 pixel_pitch，日志里 selected_action 却写着 viewing_distance
                    → 计划 §16.2 直接违约（问题错位）
```

**3）收口映射（本轮落地）**

| 旧字段 | 归属 | 处理 |
|---|---|---|
| `dialogue_action.target_slot` | DialoguePolicy | **唯一决策**：定了槽位就以它为准 |
| `pending_slot` / `pending_question` | 销售层准备 | 降级为"候选"：只有 Policy 没定槽位时才允许竞争 |
| `next_action` | 路由兼容 | 本轮不动（Phase 6 随 `script_generator` 一起瘦身） |
| 重复提问闸门改问 | 同层防重 | 例外保留：闸门 `duplicate_question*` 改的问不能反过来被覆盖 |

**4）代码改动**

- `src/orchestrator.py::_select_turn_action()`：Policy 已定 `target_slot` 时，
  不再把销售层 `pending_question` 当竞争者（Phase 1 的"后续模块不得改决定"）。
- `src/orchestrator.py::_finalize_turn_response()`：新增 Action 一致性对齐 ——
  若 Policy 槽位 ≠ 销售层准备问句的槽位，以 Policy 为准：换掉正文里那句问错的问句、
  生成该槽位的标准问句（措辞仍交给 LLM/模板），并记录 `result["action_consistency"]`
  与 `[ActionConsistency]` 警告日志。
- 例外：`duplicate_check` 以 `duplicate_question` 开头（闸门刚故意改问）时不对齐。

**5）验收（计划 §16.1 / §16.2 / §16.3）**

```bash
python -m pytest tests/dialogue/test_action_consistency.py -q   # 5 passed
python -m pytest -q                                             # 596 passed（全量）
```

- 一个 Turn 只有一个 `selected_action`，其余进 `discarded_actions`（候选留痕）
- `DialogueAction.question_slot` == `FinalResponse` 实际问的 slot（Policy 优先）
- `RECOMMEND` 动作本轮问 0 个需求问题
- 闸门改问不被对齐覆盖

---

## 1. 改造目标

当前：

```text
              ┌─ QuestionPlanner
              │
Requirement ──┼─ QuestionFlow
              │
              ├─ DialoguePolicy
              │
              └─ script_generator
                       ↓
                 ResponseGenerator
```

目标：

```text
Requirement
    │
    ├── QuestionPlanner → 候选问题
    ├── QuestionFlow   → 需求收集状态
    │
    └──────────────┐
                   ↓
             DialoguePolicy
                   ↓
             DialogueAction
                   ↓
            ResponseContext
                   ↓
          ResponseGenerator
                   ↓
          ResponseValidator
                   ↓
             FinalResponse
```

核心原则：

> **DialoguePolicy 决定“做什么”，ResponseGenerator 决定“怎么说”。**

---

## 2. 当前问题

### 2.1 QuestionPlanner 权限过大

`src/agents/sales/question_planner.py` 当前将自己定义为“问什么”的唯一来源，并通过 `plan_next_question()` 直接产生 slot、question、priority 等。

**改造：**

QuestionPlanner 降级为 Candidate Question Generator，只负责提供候选，不负责最终选择。

---

## 3. QuestionFlow 收口

保留 QuestionFlow，但只负责：

- 当前需求收集进度
- 已确认 / 缺失字段
- 已问问题
- 字段状态
- 是否允许再次询问

禁止：

- 最终决定本轮问哪个问题
- 生成最终客户话术
- 修改 DialogueAction
- 直接生成 `state["response"]`

---

## 4. DialoguePolicy 成为唯一决策中心

当前 `src/dialogue/action.py` 已经存在 `DialogueDecision`、`decide_dialogue_action()`、`select_single_action()`，作为本次改造核心。

最终一轮只允许一个 Action：

```text
DIRECT_ANSWER
ASK
ANSWER_AND_ASK
RECOMMEND
CLARIFY
CONFIRM
ACK_ONLY
```

例如：

```python
DialogueAction(
    action="ASK",
    question_slot="viewing_distance",
    reason="required_for_recommendation"
)
```

一旦确定 `ASK(viewing_distance)`，后续模块不得改成 `ASK(pixel_pitch)`。

逐步减少 `pending_question`、`pending_slot`、`next_action`、`should_generate_solution` 作为独立决策来源；它们只能由最终 DialogueAction 派生或兼容。

---

## 5. QuestionPlanner → DialoguePolicy

改为：

```text
Requirement
     ↓
QuestionFlow
     ↓
QuestionPlanner
     ↓
CandidateQuestions
     ↓
DialoguePolicy
     ↓
唯一 DialogueAction
```

例如：

```json
[
  {"slot": "viewing_distance", "priority": 1},
  {"slot": "installation", "priority": 2}
]
```

最终：

```json
{
  "action": "ASK",
  "question_slot": "viewing_distance"
}
```

QuestionPlanner 不再与 DialoguePolicy 抢最终决策权。

---

## 6. ResponseContext 结构化

当前 `src/dialogue/response_context.py` 已经是正确方向，但 `question`、`question_slot`、`question_intent`、`required_question` 存在重复。

建议逐步统一：

```text
QuestionSpec
├── slot
└── intent
```

目标：

```python
ResponseContext(
    action="ASK",
    question=QuestionSpec(
        slot="viewing_distance",
        intent="collect approximate viewing distance"
    ),
    known_facts=[...],
    missing_facts=[...],
    business_goal="...",
    recommendation=...,
    engineering_result=...,
    recent_dialogue="...",
    restrictions=[...]
)
```

原则：

```text
slot     = Python 决策
intent   = Python 决策
wording  = LLM 生成
```

---

## 7. ResponseGenerator 成为唯一正常文本出口

当前 `src/dialogue/response_generator.py` 的方向正确。

最终：

```text
DialogueAction
      +
ResponseContext
      ↓
ResponseGenerator
      ↓
LLM Native Generation
      ↓
ResponseValidator
      ↓
FinalResponse
```

ResponseGenerator 只负责：

- 自然语言表达
- 上下文衔接
- B2B 销售语气
- 多语言表达
- 简洁程度
- 避免机械复述
- 避免固定模板味

禁止：

- 修改 Action
- 修改 question_slot
- 自己选择另一个问题
- 自己决定是否推荐
- 修改推荐型号
- 修改计算结果
- 创造产品参数、价格、交期事实

---

## 8. script_generator 大幅瘦身

当前 `src/agents/sales/nodes/script_generator.py` 仍承担报价、交期、售后 FAQ、图片确认、off-topic、需求采集、greeting、need_query、objection、product_question、others、closing 等大量判断。

目标：

```text
业务判断 → DialoguePolicy
结构化上下文 → ResponseContext
自然语言生成 → ResponseGenerator
```

`script_generator` 最终只负责把已有的 DialogueAction / ResponseContext 交给 ResponseGenerator，逐步退出核心决策与文本生成。

禁止其重新决定：

- 问哪个 slot
- 是否追加问题
- 是否改变 action
- 是否重新推荐

---

## 9. `_natural_reply()` 改造

当前 `_natural_reply()` 会根据 `answer/question` 重新推导：

```text
ANSWER_AND_ASK / DIRECT_ANSWER / ASK
```

这会造成第二个决策中心。

改为：

```python
action = state["dialogue_action"]
```

然后：

```text
DialogueAction
    ↓
build_context()
    ↓
generate_response()
```

ResponseGenerator 不再重新推导业务动作。

---

## 10. `compose_requirement_reply()` 降级为 fallback

当前 `src/rag/reply_composer.py` 的 `compose_requirement_reply()` 仍被真实流程调用。

目标正常路径：

```text
ResponseGenerator
    ↓
LLM Native
    ↓
Validator
    ↓
FinalResponse
```

只有以下情况才进入 fallback：

- LLM 不可用
- LLM 超时
- 生成为空
- Validator 无法修复

旧模板链路先标记 Legacy，再逐步删除，不能一次性粗暴删除。

---

## 11. ResponseCoordinator 改造

不要：

```text
生成回复
  ↓
再追加 FAQ
  ↓
再追加 Vision confirmation
  ↓
再加工文本
```

改为：

```text
Service FAQ
Vision
Recommendation
Engineering
Requirement
        ↓
ResponseContext
        ↓
ResponseGenerator
        ↓
FinalResponse
```

所有事实一次性进入 ResponseContext，由 ResponseGenerator 统一组织。

---

## 12. ResponseValidator 改造

### HARD：必须阻止

```text
wrong_action
wrong_question_slot
too_many_questions
invented_fact
invented_number
invented_model
changed_engineering_result
unsupported_price
unsupported_delivery_time
internal_term
```

### SOFT：只记录指标

```text
generic_ack
repeated_connector
repeated_phrasing
unnecessary_echo
```

原则：

```text
HARD → Repair / Fallback
SOFT → 记录指标
```

避免轻微语言问题导致大量 LLM 重试。

---

## 13. 统一语言策略

当前 `_natural_reply()` 存在 `language="en"` 硬编码。

统一使用项目已有：

```text
response_language_rule()
reply_language()
RESPONSE_LANGUAGE_POLICY
```

最终：

```text
auto → 跟随客户语言
en   → 英文
zh   → 中文
```

---

## 14. Strategy A / B 收口

短期可以保留用于 A/B 测试。

验证完成后：

```text
Native Generation
```

成为唯一正常路径。

旧：

```text
Template → LLM Polish
```

退出正常生产链。

---

# 15. 实施阶段

## Phase 1：建立唯一决策对象

任务：

- 梳理所有产生 `pending_question`
- 梳理所有产生 `pending_slot`
- 梳理所有产生 `next_action`
- 梳理所有产生 `should_generate_solution`
- 找出重复决策点
- 统一映射到 DialogueAction
- 增加 Action 一致性测试

验收：

```text
一个 Turn = 一个最终 DialogueAction
```

---

## Phase 2：QuestionPlanner 降级

任务：

- `plan_next_question()` 改为候选生成
- 删除“问什么唯一来源”的定位
- 不直接写最终 response
- 不覆盖 DialoguePolicy
- 保留 priority、blocking、state 等候选信息

验收：

```text
QuestionPlanner 只能产生 Candidate
DialoguePolicy 才能选择最终 slot
```

---

## Phase 3：QuestionFlow 收口

任务：

- 统一字段状态读取
- 统一已问记录
- 统一 UNKNOWN / CONFIRMED 等状态
- 删除 QuestionFlow 中的最终回复逻辑
- 删除 QuestionFlow 中最终问题选择逻辑

验收：

```text
QuestionFlow = 状态管理
```

---

## Phase 4：ResponseContext 重构

任务：

- 合并重复 question 字段
- 建立 QuestionSpec
- Action 与 QuestionSpec 强绑定
- 推荐、工程计算、FAQ、Vision 全部结构化
- 删除成句文本作为核心决策字段

验收：

```text
ResponseContext 不负责决定业务
只描述已经决定好的内容
```

---

## Phase 5：ResponseGenerator 唯一出口

任务：

- 所有正常客户回复统一进入 ResponseGenerator
- LLM 只负责自然表达
- 禁止修改 Action / slot / model / calculation
- Validator 统一接收生成结果
- 统一 response_source

验收：

```text
正常路径只有一个文本生成出口
```

---

## Phase 6：script_generator 瘦身

任务：

- 将业务决策迁移到 DialoguePolicy
- 将结构化上下文交给 ResponseGenerator
- 删除主要的 `state["response"]` 直接生成路径
- 删除重复问题选择
- 删除重复回复生成
- 保留必要兼容层

验收：

```text
script_generator 不再拥有独立的“问什么 / 怎么做”决策权
```

---

## Phase 7：旧回复链退出

逐步处理：

```text
compose_requirement_reply()
reply_composer
template polish
script_generator direct response
ResponseCoordinator text mutation
```

原则：

```text
先替换
→ 测试
→ 日志验证
→ 再删除
```

---

# 16. 必须新增/重构的测试

## 16.1 Action 一致性

```text
QuestionPlanner = A
DialoguePolicy = B
最终只能执行 B
```

## 16.2 Slot 一致性

```text
DialogueAction:
ASK(viewing_distance)

FinalResponse 必须询问 viewing_distance
不能询问 pixel_pitch / installation / environment
```

## 16.3 一轮一个问题

```text
ASK → 1 question
ANSWER_AND_ASK → 1 question
DIRECT_ANSWER → 0 requirement questions
RECOMMEND → 0 requirement questions
```

## 16.4 ResponseGenerator 不得改决策

输入：

```text
ASK(viewing_distance)
```

不能输出：

```text
What pixel pitch do you need?
```

## 16.5 推荐锁定

已确定型号后，ResponseGenerator 不得输出其他型号。

## 16.6 工程结果锁定

ResponseGenerator 不得修改：

- 屏幕尺寸
- 箱体数量
- 分辨率
- 实际尺寸
- 计算结果

## 16.7 FAQ + ASK

必须一次性生成：

```text
FAQ事实 + 一个合理需求问题
```

不得生成后再硬拼第二段。

## 16.8 Vision + ASK

Vision confirmation 与需求问题统一进入 ResponseContext，再由 ResponseGenerator 一次生成。

---

# 17. 日志要求

每轮记录：

```text
turn_id
customer_message
candidate_questions
selected_action
selected_question_slot
response_source
validation_result
final_response
```

推荐格式：

```text
[DialogueDecision]
candidates=[environment, viewing_distance]
selected=ASK
slot=viewing_distance

[ResponseGenerator]
action=ASK
slot=viewing_distance
source=llm

[ResponseValidator]
hard_errors=0
soft_issues=1

[FinalResponse]
...
```

这样可以完整追踪：

```text
Candidate
→ Policy
→ Action
→ Context
→ Response
```

---

# 18. 禁止继续增加的东西

```text
❌ 第二个 QuestionPlanner
❌ 第二套 DialoguePolicy
❌ 第二个 ConversationState
❌ 第二套 Turn System
❌ 在 script_generator 中继续堆 if/else 决策
❌ 在 ResponseGenerator 中重新判断业务
❌ 让 LLM 自己决定下一问题
❌ ResponseGenerator 后继续大量拼接文本
❌ 用更多 Guard 掩盖职责冲突
```

核心目标：

> **减少决策中心，而不是继续增加规则。**

---

# 19. 最终验收标准

### 决策

```text
一个 Turn
→ 一个 DialogueAction
```

### 问题

```text
一个 Turn
→ 最多一个需求问题
```

### 问题一致性

```text
DialogueAction.question_slot
=
FinalResponse 实际询问的 slot
```

### 推荐一致性

```text
Recommendation Engine
=
DialogueAction
=
ResponseContext
=
FinalResponse
```

### 文本出口

```text
正常路径
→ ResponseGenerator
→ Validator
→ FinalResponse
```

### LLM 权限

```text
LLM
只能改变表达方式

不能改变：
Action
Question Slot
Recommendation
Engineering Result
Grounded Facts
```

---

# 20. 最终职责定义

```text
QuestionPlanner
= 提供候选

QuestionFlow
= 管理状态

DialoguePolicy
= 做最终业务决定

script_generator
= 逐步退出核心决策与文本生成

ResponseGenerator
= 把最终决定自然地说出来
```

最终核心边界：

```text
                 “做什么”
                     │
                     ▼
              DialoguePolicy
                     │
                     ▼
              DialogueAction
                     │
                     ▼
                 “怎么说”
                     │
                     ▼
             ResponseGenerator
```

本次改造的最终目的不是增加代码，而是让：

> **一个问题只有一个最终负责人，一轮只有一个最终 Action，一条客户回复只有一个正常文本出口。**
