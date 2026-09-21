# LED RAG 智能销售系统：消息聚合与自然对话链路优化计划

> 版本：v1.0  
> 日期：2026-09-21  
> 基准：当前 `main` 分支最新代码 + 最新运行日志  
> 目标：解决连续消息逐条触发 Agent、客户主动提问后仍机械推进需求采集、话术僵硬、LLM 补充未授权业务事实、LLM 调用统计不准确等问题。  
> 原则：不破坏现有推荐引擎、RAG、工程计算、Requirement Profile、First Contact 等已经工作的核心能力。

---

## 一、问题结论

当前问题不是单纯 Prompt 不够自然，而是**消息接收层、对话理解层、需求决策层、回复生成层没有形成真正的“对话回合”机制**。

当前实际链路更接近：

```text
客户一条消息
    ↓
完整 Sales Agent Run
    ↓
提取需求
    ↓
检查缺失字段
    ↓
QuestionFlow
    ↓
马上问下一个字段
    ↓
回复客户
```

因此容易出现：

```text
客户：P3
AI：P3 works well... Should I optimise for the best price or quality?

客户：price
AI：What's the screen for, and where will it be used?

客户：你的交付日期是多久
AI：回答交付周期 + 再问 installation
```

这不是自然销售对话，而是“需求问卷 + LLM 话术包装”。

---

# 二、日志暴露出的核心问题

## 2.1 连续消息没有真正聚合

当前每一条客户消息都会独立触发：

```text
Processing
→ Intent
→ Requirement Extraction
→ Readiness
→ QuestionFlow
→ ScriptGenerator
→ Response
```

日志中的典型耗时：

| 消息 | 单轮耗时 |
|---|---:|
| `i need a led display` | 9877.7ms |
| `3*5` | 7824.2ms |
| `indoor` | 7112.8ms |
| `p3` | 6990.3ms |
| `price` | 7455.8ms |
| `你的交付日期是多久` | 7809.1ms |

说明当前系统没有稳定的：

```text
Message Buffer
+
Debounce
+
Conversation Turn
```

机制。

---

## 2.2 QuestionFlow 只关注“缺什么”，不够关注“客户刚刚做了什么”

例如客户输入：

```text
p3
```

系统识别到了：

```text
pixel_pitch = 3.0
```

但随后直接：

```text
missing = installation
QuestionFlow → price_preference
```

最后生成：

```text
P3 works well for that size indoors.
Should I optimise for the best price, or for the best quality?
```

问题不是这个问题本身一定错误，而是：

> 客户刚刚回答了一个需求，系统没有根据这个回答形成“接话 → 更新状态 → 判断下一步”的完整对话动作。

---

## 2.3 客户主动提问时，需求采集不应该拥有绝对优先级

例如：

```text
客户：你的交付日期是多久
```

这是一个明显的业务问题。

系统应该：

```text
识别客户主动问题
→ 优先回答
→ 更新当前需求状态
→ 如果仍然需要继续收集需求，再选择是否追问
```

而不是：

```text
回答交付周期
+
强行追加 installation 问题
```

当前日志已经出现：

```text
Delivery / lead-time reply
+
No jargon then — is it fixed in place...
```

这种“回答客户 A + 强行追问 B”的结构。

---

# 三、总体改造目标

最终将当前链路：

```text
Message
 ↓
Requirement Extraction
 ↓
Missing Field
 ↓
QuestionFlow
 ↓
ScriptGenerator
```

升级为：

```text
Customer Messages
        ↓
┌─────────────────────┐
│ Message Aggregator  │
│ 600~1800ms          │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Conversation Turn   │
│ Builder             │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Conversation        │
│ Understanding       │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Requirement State   │
│ Update              │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Dialogue Policy     │
│ 决定下一步做什么     │
└──────────┬──────────┘
           ↓
    ┌──────┴──────┐
    ↓             ↓
 ANSWER          ASK ONE
    ↓             ↓
    └──────┬──────┘
           ↓
┌─────────────────────┐
│ ResponseContext     │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ ResponseGenerator   │
│ LLM只负责表达       │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ FinalResponseGuard  │
└──────────┬──────────┘
           ↓
        Customer
```

---

# 四、第一阶段：建立 Message Aggregator ✅ 已完成（v2.5++++）

## 4.1 目标

客户连续快速发送多条消息时，不应该每条都立即触发完整 Agent。

例如客户：

```text
3*5
indoor
p3
price
```

应该尽量形成一个 Conversation Turn：

```text
3*5
indoor
p3
price
```

然后只执行一次 Sales Agent。

---

## 4.2 推荐参数

初版建议：

```python
DEBOUNCE_MIN_MS = 600
DEBOUNCE_MAX_MS = 1800
```

行为：

```text
收到第一条
 ↓
等待约 600ms

600ms 内继续收到消息
 ↓
重新等待

持续输入
 ↓
继续聚合

达到 1800ms 上限
 ↓
强制提交
```

目标不是“固定等待 3 秒”。

而是：

> 尽量等待客户把这一轮话说完，同时不让单条消息产生明显延迟。

---

## 4.3 聚合器职责

Message Aggregator 只负责：

- 接收消息
- 按 session_id / conversation_id 分组
- 保存短时间内连续消息
- 判断 quiet period
- 形成 Conversation Turn
- 将 Turn 交给 Sales Agent

不要在 Aggregator 内做：

- 产品推荐
- RAG
- LLM 销售话术
- Requirement Gate
- 业务决策

---

# 五、第二阶段：建立 Conversation Turn ✅ 已完成（v2.5++++）

当前系统以：

```text
message
```

作为核心输入单位。

需要升级成：

```text
conversation_turn
```

示例：

```json
{
  "turn_id": "turn_xxx",
  "session_id": "session_xxx",
  "messages": [
    "3*5",
    "indoor",
    "p3"
  ],
  "merged_text": "3*5 indoor p3",
  "created_at": "...",
  "message_count": 3
}
```

这样后面的 Agent 看到的是：

> 客户这一轮完整表达了什么。

而不是：

> 客户刚刚发送了哪一个字段。

---

# 六、第三阶段：加入 Conversation Understanding ✅ 已完成（SpeechAct）

不要再单纯依赖：

```text
recommendation
need_query
others
objection
```

需要识别客户当前“说话行为”。

建议建立：

```python
SpeechAct
```

至少支持：

```text
ANSWER_REQUIREMENT
NEW_REQUIREMENT
CUSTOMER_QUESTION
PRICE_QUESTION
DELIVERY_QUESTION
PRODUCT_QUESTION
OBJECTION
CONFIRMATION
CORRECTION
CASUAL
MULTI_INTENT
```

---

## 6.1 示例：回答需求

客户：

```text
P3
```

输出：

```json
{
  "speech_act": "ANSWER_REQUIREMENT",
  "field": "pixel_pitch",
  "value": "P3"
}
```

系统知道：

> 客户是在回答上一轮问题。

因此应该优先完成：

```text
确认
→ 更新 profile
→ 评估影响
→ 决定下一步
```

---

## 6.2 示例：客户主动提问

客户：

```text
How long is delivery?
```

应该：

```text
speech_act = DELIVERY_QUESTION
```

然后进入：

```text
Answer Customer Question
```

而不是：

```text
找一个缺失字段
→ 强制继续问
```

---

## 6.3 示例：多意图

客户：

```text
P3. Also, how long is delivery?
```

识别：

```json
{
  "speech_act": "MULTI_INTENT",
  "requirements": {
    "pixel_pitch": "P3"
  },
  "customer_questions": [
    "delivery"
  ]
}
```

处理顺序：

```text
更新 P3
 ↓
优先回答 delivery
 ↓
再判断是否需要下一问题
```

---

# 七、第四阶段：重构 Dialogue Policy ✅ 已完成（优先级 P0~P5）

这是本次最重要的改造。

当前逻辑：

```text
missing field
→ ask missing field
```

改成：

```text
当前客户行为
+
当前 Requirement State
+
当前业务状态
+
客户刚刚提供的信息
+
Recommendation Gate
→ Dialogue Action
```

---

## 7.1 Dialogue Action

建议统一：

```python
class DialogueAction:
    action: Literal[
        "answer",
        "acknowledge",
        "ask",
        "recommend",
        "clarify",
        "wait"
    ]

    target_slot: Optional[str]
    reason: str
    priority: int
```

---

## 7.2 优先级

建议：

```text
P0  客户明确问题
P1  客户纠正 / 冲突
P2  客户刚刚提供的需求需要确认
P3  Recommendation Gate 必须条件
P4  业务上最有价值的下一问题
P5  普通补充信息
```

因此：

```text
客户主动询问交付
```

优先级必须高于：

```text
系统还缺 installation
```

---

# 八、第五阶段：重新定义“下一问题”的选择机制 ✅ 已完成（候选 → 过滤 → 价值排序 → 同分才随机）

不要继续使用：

```text
missing
→ random_pass
→ ask
```

改成：

```text
Candidate Questions
        ↓
过滤无意义问题
        ↓
过滤客户已经回答的问题
        ↓
过滤当前不重要的问题
        ↓
根据业务价值排序
        ↓
如果多个问题价值相同
        ↓
才允许随机
        ↓
只选一个
```

---

## 8.1 环境问题保持硬优先

如果环境没有被可靠确定：

```text
environment = unknown
```

则：

```text
第一优先级：
indoor / outdoor
```

但如果客户已经明确：

```text
indoor
```

或者根据你现有业务规则可以可靠从场景推断：

```text
environment = indoor
```

则不要重复问。

必须保留现有：

> 明显场景可以推断 indoor / outdoor。

---

# 九、第六阶段：解决“回答后马上追问”的僵硬问题 ✅ 已完成（answer_only）

核心原则：

> **不是每一轮回答客户以后都必须继续问问题。**

应该允许：

```text
ANSWER_ONLY
```

例如：

```text
客户：What's the delivery time?

AI：
Our usual lead time is around 15–30 days after order confirmation...
```

这一轮可以结束。

下一轮客户继续说：

```text
Ok, it's for indoor use.
```

再继续需求采集。

---

## 9.1 推荐增加 action

```text
answer_only
answer_then_ask
ask_only
clarify_only
recommend_only
```

其中：

### answer_only

客户主动问题优先，当前没有必要继续追问。

### answer_then_ask

客户的问题已经回答，并且确实存在一个高价值、必要的下一问题。

### ask_only

客户本轮只是提供需求，没有需要解释的内容。

### clarify_only

客户输入存在冲突或无法判断。

### recommend_only

Requirement Gate 已满足，可以进入推荐。

---

# 十、第七阶段：自然语言生成层重构 ✅ 已完成

最终原则：

```text
Python 决定：
“做什么”

LLM 决定：
“怎么说”
```

LLM 不应该决定：

- 下一步问哪个字段
- 是否进入推荐
- 产品参数
- 工程计算
- 交付事实
- 公司事实
- 未提供的业务事实

---

# 十一、建立 GroundedFact ✅ 已完成

ResponseContext 中增加结构化事实：

```python
GroundedFact(
    field="pixel_pitch",
    value="P2.9",
    source="recommended"
)
```

例如：

```python
GroundedFact(
    field="environment",
    value="indoor",
    source="customer"
)

GroundedFact(
    field="viewing_distance",
    value="≈5m",
    source="inferred_from_pitch"
)

GroundedFact(
    field="screen_size",
    value="3m × 5m",
    source="customer"
)
```

允许：

```text
customer
inferred
calculated
retrieved
recommended
```

但 LLM 不得自己创造：

```text
unknown
```

事实。

---

# 十二、第八阶段：解决 P3 → P2.9 之类的表达问题 ✅ 已完成

Requirement / Decision 层必须保存：

```python
requested_pitch
resolved_pitch
pitch_match_type
pitch_resolution_reason
```

例如：

```json
{
  "requested_pitch": "P3",
  "resolved_pitch": "P2.9",
  "pitch_match_type": "NEAREST_AVAILABLE",
  "pitch_resolution_reason": "P3 is not an exact available pitch"
}
```

LLM 才可以自然表达：

> P3 isn't an exact available pitch in this range, so P2.9 is the closest match.

而不是：

```text
前一句：P3
后一句：P2.9
```

造成客户感觉系统前后矛盾。

---

# 十三、第九阶段：统一 Final Response ✅ 已完成

必须确保所有客户可见输出最终经过：

```text
ResponseContext
 ↓
ResponseGenerator
 ↓
ResponseValidator
 ↓
FinalResponseGuard
 ↓
Customer
```

不能再存在：

```text
Legacy Composer
Legacy Script Generator
特殊 Intent 分支
ResponseCoordinator
```

各自直接输出客户文本的情况。

---

# 十四、One Question Guard ✅ 已完成

最终输出增加全局检查：

```python
question_count <= 1
```

但不能只是简单地把第二个 `?` 删除。

正确做法：

```text
检测到多个问题
 ↓
保留最高优先级问题
 ↓
删除低优先级追问
 ↓
重新生成 / 结构化修复
```

例如：

错误：

```text
How long is delivery?
What is the installation type?
What is your budget?
```

最终：

```text
How long is delivery?
```

如果本轮客户主动问交付，就优先保留客户问题。

---

# 十五、第十阶段：修复 LLM 调用统计 ✅ 已完成（LLMCallTracker）

日志中出现：

```text
PERF ... llm_calls=0
```

但同一轮实际上出现了多次：

```text
POST https://api.deepseek.com/chat/completions
```

说明：

> PerfTracker 的 LLM 调用统计没有覆盖所有 LLM 调用入口。

需要统一 LLM 调用统计。

建议：

```python
LLMCallTracker
```

统一记录：

```text
call_id
session_id
turn_id
agent
node
model
start_time
end_time
latency_ms
prompt_tokens
completion_tokens
total_tokens
success
error
```

最终：

```text
turn_llm_calls
turn_llm_latency
turn_total_latency
```

必须真实反映一轮的调用情况。

---

# 十六、第十一阶段：测试体系 ✅ 已完成（tests/test_message_aggregation_chain.py）

新增以下测试。

## 16.1 连续消息

输入：

```text
3*5
indoor
P3
```

要求：

```text
Agent Run = 1
Customer Response = 1
```

---

## 16.2 单条消息

输入：

```text
I need an indoor LED screen.
```

要求：

```text
不因为 Aggregator 引入明显等待
```

---

## 16.3 客户回答问题

```text
AI: What pixel pitch do you have in mind?

Customer: P3
```

要求：

```text
更新 pixel_pitch
自然接话
最多一个下一问题
```

---

## 16.4 客户主动询价

```text
Customer: How much?
```

要求：

```text
优先处理价格问题
不能机械跳到随机需求字段
```

---

## 16.5 客户主动问交付

```text
Customer: How long is delivery?
```

要求：

```text
优先回答 delivery
不得无理由追加 installation / purpose
```

---

## 16.6 多意图

```text
P3. How long is delivery?
```

要求：

```text
pixel_pitch = P3
delivery_question = true
```

最终：

```text
先回答 delivery
```

---

## 16.7 P3 → P2.9

要求：

```text
requested_pitch = P3
resolved_pitch = P2.9
```

回复必须解释原因。

---

## 16.8 Unsupported Fact

测试：

```text
客户没有提供 viewing distance
```

LLM 不得自行说：

```text
This is suitable for your 5m viewing distance.
```

除非系统存在：

```text
inferred_from_pitch
```

等合法 provenance。

---

# 十七、实施顺序

## Phase 1：Message Aggregator ✅

优先级：P0

修改：

```text
API / message entry
session handling
```

完成：

```text
600~1800ms debounce
消息合并
Conversation Turn
```

---

## Phase 2：Conversation Understanding ✅

优先级：P0

增加：

```text
SpeechAct
CustomerQuestion
MultiIntent
```

---

## Phase 3：Dialogue Policy ✅

优先级：P0

将：

```text
Missing Field → Ask
```

替换为：

```text
Customer Behavior
+
Requirement State
+
Business Priority
→ DialogueAction
```

---

## Phase 4：Response Pipeline 收口 ✅

优先级：P1

统一：

```text
ResponseContext
→ ResponseGenerator
→ Validator
→ FinalResponseGuard
```

逐步移除旧的直接回复路径。

---

## Phase 5：GroundedFact ✅

优先级：P1

统一：

```text
customer
inferred
calculated
retrieved
recommended
unknown
```

解决 LLM 自己补业务事实的问题。

---

## Phase 6：LLM Metrics ✅

优先级：P1

统一统计：

```text
LLM calls
latency
tokens
errors
```

---

## Phase 7：回归测试 ✅

优先级：P1

目标：

```text
原有测试全部通过
+
新增对话测试全部通过
```

---

# 十八、建议修改的代码区域

实施前必须以 `main` 最新代码实际结构为准，不直接假定文件已经存在。

重点检查：

```text
src/api.py
src/orchestrator.py

src/agents/sales/runner.py

src/agents/sales/nodes/classify.py
src/agents/sales/nodes/requirement.py
src/agents/sales/nodes/script_generator.py

src/agents/sales/question_planner.py

src/dialogue/action.py
src/dialogue/response_context.py
src/dialogue/response_generator.py
src/dialogue/response_validator.py
src/dialogue/response_coordinator.py
src/dialogue/question_flow.py

src/rag/readiness.py

src/memory/
```

同时重点搜索所有：

```text
POST /chat
Processing:
QuestionFlow
QuestionPlanner
script_generator
compose_requirement_reply
reply_composer
ResponseGenerator
ResponseCoordinator
```

确认是否仍存在多条客户可见输出路径。

---

# 十九、必须保留的现有业务逻辑 ✅ 全部保留

本次优化不能破坏：

### 1. Environment

保留：

```text
明显场景 → 可以推断 indoor/outdoor
```

但如果无法确定：

```text
第一优先问题 = environment
```

---

### 2. Pitch ↔ Viewing Distance

保留现有：

```text
pixel pitch → approximate viewing distance
viewing distance → approximate pixel pitch
```

不要为了 provenance 保护而删除这个业务能力。

关键是：

```text
inferred_from_pitch
```

必须明确记录来源。

---

### 3. Recommendation Gate

继续保证：

```text
条件不足
→ 不推荐
```

---

### 4. Deterministic Recommendation Engine

继续保证：

```text
Python 决定产品
LLM 不重新选择产品
```

---

### 5. Screen Calculator

继续保持：

```text
Python 计算
LLM 解释
```

---

### 6. RAG

继续保持：

```text
RAG 提供事实证据
```

不要让自然语言优化破坏事实边界。

---

# 二十、最终验收标准 ✅ 逐条满足（见文末实施记录）

本次优化完成后，系统必须达到：

## 消息层

```text
连续消息：
尽量合并成一轮

单条消息：
不产生明显额外等待
```

---

## 对话层

客户：

```text
P3
```

AI：

```text
自然接住 P3
→ 更新状态
→ 如确实需要，只问一个问题
```

不能：

```text
P3
→ 机械随机问另一个字段
```

---

## 主动问题

客户：

```text
How much?
How long is delivery?
Can you do 4K?
```

必须：

```text
优先处理客户主动问题
```

而不是强行推进需求问卷。

---

## 多意图

客户：

```text
P3 and how long is delivery?
```

必须：

```text
一次理解
一次回答
```

---

## 自然度

不能出现大量固定：

```text
Right...
That helps...
Got it...
By the way...
```

形成：

```text
ACK
+
Echo
+
Connector
+
Question
```

的固定模板。

自然表达必须由：

```text
Conversation Context
+
Dialogue Action
+
Grounded Facts
```

驱动。

---

## 事实安全

LLM：

```text
可以自然表达已有事实
可以表达合法推断
可以表达计算结果
可以表达 RAG 检索事实
可以表达推荐结果
```

但是：

```text
不得凭空创造业务事实
```

---

## 性能

最终日志应该能够真实看到：

```text
turn_total_ms
message_count
aggregated = true/false
llm_calls
llm_latency_ms
```

例如：

```text
[Turn]
messages=3
aggregated=true
llm_calls=2
total_ms=...
```

而不能出现：

```text
实际调用多个 DeepSeek
但：
llm_calls=0
```

---

---

## 实施记录（v2.5++++，2026-09-21）

### 一、逐项交付

| 计划章节 | 交付物 | 说明 |
|---|---|---|
| §四 Message Aggregator | `src/input/message_aggregator.py`、`static/app.js`、`src/api.py` | debounce/max window 按计划改成 **600ms / 1800ms**（可用 `LED_RAG_TURN_DEBOUNCE_MS` / `LED_RAG_TURN_MAX_WINDOW_MS` 或前端 localStorage 覆盖）；API 把「这一轮由几条消息聚合而来」传给编排器 |
| §五 Conversation Turn | `src/input/user_turn.py`、`turn_payload.py`（v2.5 已有）+ API `messages[]` | 一次请求 = 一个 turn，日志按 turn 输出 |
| §六 Conversation Understanding | `src/dialogue/speech_act.py`（新增） | 11 种 SpeechAct；`MULTI_INTENT` 同时给出需求与客户问题；纯规则不额外调 LLM |
| §七 Dialogue Policy | `src/dialogue/policy.py`（新增） | P0 客户问题 > P1 纠正/冲突 > P2 刚给的需求 > P3 Gate 必须 > P4 最有价值问题 > P5 补充 |
| §八 下一问题选择 | `policy.question_candidates()` + `question_flow.pass1_pending()` | 候选 → 过滤（已问过/已答/无意义）→ **业务价值排序** → 同价值才用会话随机；环境未定仍是硬优先 |
| §九 answer_only | `policy.should_append_requirement_question()` + `ResponseCoordinator` + `script_generator` | 客户问价格 / 交期 → 先答完，**不再硬塞**需求问题；待问项留到下一轮继续采集 |
| §十 自然语言生成层 | v2.5++ 已完成的 ResponseGenerator（唯一出口）| 本次补上 SpeechAct/Policy 上下文与事实约束 |
| §十一 GroundedFact | v2.5+++ 的 `src/dialogue/grounded_facts.py` | customer / inferred / calculated / retrieved / recommended；unknown 不允许出现 |
| §十二 P3 → P2.9 | v2.5+++ 的 `src/rag/pitch_resolution.py` | `requested/resolved/match_type/reason`，不一致时把解释句放进推荐理由 |
| §十三 统一 Final Response | `ResponseGenerator → Validator → FinalResponseGuard`（v2.5++/+++）| 所有客户可见文本（含附加气泡）都经过 Guard |
| §十四 One Question Guard | `src/dialogue/final_guard.py` | 多问题时保留最高优先级，其余下一次再问；内部术语句直接删 |
| §十五 LLM 统计 | `src/observability/llm_tracker.py`（新增）+ `src/core/llm.py` + `perf.py` + `orchestrator.py` | 在 **LLM 调用入口**记账：call_id/turn_id/model/latency/tokens/success；每轮结束打 `[Turn] messages=.. aggregated=.. llm_calls=.. llm_latency_ms=.. total_ms=..` |
| §十六 测试 | `tests/test_message_aggregation_chain.py`（新增 16 条）| 覆盖 §16.1~§16.8 全部场景 |

### 二、实测（本轮新增测试）

```
§16.1 连续消息 "3*5"→"indoor"→"P3"        → 聚合成 1 个 turn，只发一次（Agent Run = 1）
§16.2 单条消息                             → 最多只等 600ms（不会等满 1800ms）
§16.3 客户答 "P3"                          → ANSWER_REQUIREMENT（field=pixel_pitch）→ 只问一个下一问题
§16.4 "How much?"                          → PRICE_QUESTION    → answer_only（不追加需求问题）
§16.5 "How long is delivery?" / "交付多久"  → DELIVERY_QUESTION → answer_only
§16.6 "P3. Also, how long is delivery?"    → MULTI_INTENT（需求 + 客户问题），先回答 delivery
§16.7 P3 → TW11-IR-P2.9                    → NEAREST_AVAILABLE + 解释句进入推荐理由
§16.8 LLM 说 "your 5m viewing distance"     → 无 grounded fact → ungrounded_numeric 拦截
§15   一轮里的 LLM 调用                     → llm_calls / llm_latency_ms / tokens 真实统计（不再是 0）
```

### 三、验收（计划 §20）

| 项目 | 标准 | 结果 |
|---|---|---|
| 连续消息 | 尽量合并成一轮 | ✅ 600/1800ms 聚合，测试证明 Agent Run=1 |
| 单条消息 | 无明显额外等待 | ✅ 只等 debounce（600ms） |
| 客户回答问题 | 接话 + 更新 + 最多一个问题 | ✅ SpeechAct + Policy + One Question Guard |
| 主动提问（价格 / 交期） | 优先处理，不强行推进问卷 | ✅ answer_only |
| 多意图 | 一次理解、一次回答 | ✅ MULTI_INTENT |
| 自然度 | 不再固定 ACK+Echo+Connector+Question | ✅ v2.5++ 起模板腔只作兜底 |
| 事实安全 | 不凭空创造业务事实 | ✅ GroundedFact + Fact/Numeric Guard |
| 性能 | 日志真实反映 turn 的 messages / aggregated / llm_calls | ✅ `[Turn]` 统计 |

### 四、保留（计划 §19）

明显场景推断 indoor/outdoor ✅、Pitch ↔ Viewing Distance 双向推断 ✅（并标 `inferred_from_pitch` /
`inferred_from_distance`）、Recommendation Gate ✅、确定性推荐引擎 ✅、Screen Calculator ✅、RAG 事实证据 ✅
—— 本次一行未删。Memory 未改动。

### 五、补充修复（实测反馈，2026-09-21）

**现象**：客户在拿到推荐后说"你还能给我推荐其他的吗？"，系统回
"Understood, let me re-check your requirements for this one."，然后把**刚答过的
室内外 / 尺寸又重问了一遍**（客户回"和刚刚一样的参数""3*5"之后，又被问了一次 Indoor/Outdoor）。

**根因**：`session_switch.detect_requirement_reset()` 把"推荐 + 其他/别的"判成了
`new_inquiry`（换产品）→ 全量清空需求档案 → 重新采集。

**修复**：
- `_OTHER_MODEL_REQUEST_RE` / `_MORE_OPTIONS_GUARDS` 覆盖"推荐 + 其他/别的/更多"的常见口语顺序
  （你还能给我推荐其他的吗 / 还能再推荐几款吗 / 有没有其他推荐 / 给我推荐别的 /
  can you recommend others / any other options / more models），这类一律**不算重置**；
- 真正换产品 / 重新来（"我要换一款产品""I want a different product""算了重新来"）仍然**必须重置**；
- 顺带：客户说"和刚刚一样的参数 / same as before"时，按现有需求再推荐一次，不再追问已答过的字段。

**验证**：`tests/test_session_reset.py` 新增 13 条（8 条"更多型号不重置" + 4 条"真换产品仍重置"
+ 1 条整轮行为），全量 `pytest -q` 通过。

# 二十一、最终架构原则

整个系统最终遵循：

```text
                 ┌──────────────────┐
                 │ Customer Message │
                 └────────┬─────────┘
                          ↓
                 Message Aggregator
                          ↓
                 Conversation Turn
                          ↓
                 Conversation Understanding
                          ↓
                 Requirement State
                          ↓
                 Dialogue Policy
                          ↓
                 ┌────────┴────────┐
                 ↓                 ↓
              Answer            Ask One
                 ↓                 ↓
                 └────────┬────────┘
                          ↓
                  ResponseContext
                          ↓
                  ResponseGenerator
                          ↓
                  ResponseValidator
                          ↓
                 FinalResponseGuard
                          ↓
                      Customer
```

核心职责严格保持：

```text
Message Aggregator
    = 等客户这一轮说完

Conversation Understanding
    = 判断客户这句话是什么意思

Requirement State
    = 当前已经知道什么

Dialogue Policy
    = 下一步应该做什么

Python Engineering
    = 产品 / 参数 / 计算 / 推荐 / 约束

RAG
    = 提供事实证据

LLM
    = 把已经决定好的内容自然地说出来

Final Guard
    = 防止最终输出越界
```

最终目标不是让 AI “多说一点”，而是让它真正做到：

> **先听懂客户这一轮在说什么，再决定是否需要继续问；客户主动问问题时先回答客户，而不是为了填表强行推进。**

这才是从“需求采集机器人”向“真正的 AI 销售对话系统”转换的核心。
