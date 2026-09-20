# LED RAG 智能销售系统：客户决策状态与灵活追问优化实施计划

> 项目：`WSLVOUD/sale_agent_demo`  
> 版本：v2.1  
> 目标：解决“客户授权 AI 决定仍被反复追问”和“硬性参数问不出来后无限追问”两类问题。  
> 原则：**不推倒现有 LangGraph、RequirementProfile、Recommendation Ready Gate、Calculation Ready Gate 和 Deterministic Recommendation Engine。**

---

## 1. 背景与现状

当前项目已经具备：

- `RequirementProfile`
- `query_understanding`
- `requirement_extractor`
- `question_planner`
- `Recommendation Ready Gate`
- `Calculation Ready Gate`
- `Recommendation Engine`
- `ask_counts`
- `unknown_reasons`
- `last_asked_slot`
- `EASIER_QUESTIONS`
- `READY / CONTINUE_ASKING / DEGRADED_READY`

现阶段的主要问题不是缺少模块，而是**需求字段状态表达不完整**。

当前逻辑比较接近：

```text
字段有值       → 使用
字段没值       → 缺失
字段缺失       → 继续问
```

这会导致两个问题：

### 问题 A：客户把决策权交给 AI

例如：

```text
AI：
What's the approximate width?

Customer：
No range, you recommend.
```

当前容易变成：

```text
width = None
↓
width missing
↓
继续问 width
```

但客户实际上已经明确表示：

> “这个参数我不知道，由你决定。”

应该进入：

```text
width = DELEGATED
↓
允许 AI / Python 根据已有条件推导
↓
不再追问 width
```

### 问题 B：硬性参数问了一次仍然问不出来

例如：

```text
AI：
What's the viewing distance?

Customer：
I don't know.
```

如果系统只判断：

```text
viewing_distance = None
```

就会继续把它视为普通 missing slot。

最终形成：

```text
问
↓
不知道
↓
再问
↓
还是不知道
↓
继续问
```

这不是销售 Agent 应有的行为。

---

# 2. 本次优化目标

本次 v2.1 不改变核心推荐架构，只升级“需求状态管理”。

最终实现：

```text
客户明确提供
    ↓
CONFIRMED

客户不知道
    ↓
UNKNOWN
    ↓
降低回答门槛再问一次
    ↓
仍然不知道
    ↓
DEFERRED

客户说“你决定 / 你推荐”
    ↓
DELEGATED

客户明确不提供
    ↓
DECLINED
```

核心原则：

> **一个字段缺失，不等于整个销售流程必须停止。**

进一步定义：

> **字段只阻塞依赖该字段的 Action，而不是阻塞整个 Sales Workflow。**

---

# 3. 最终状态模型

## 3.1 字段状态

建议统一使用以下状态：

| 状态 | 含义 | 后续动作 |
|---|---|---|
| `MISSING` | 从未获得该信息 | 可以询问 |
| `CONFIRMED` | 客户明确提供 | 正常使用 |
| `INFERRED` | 系统推导得到 | 按现有规则使用，不能伪装成客户确认 |
| `UNKNOWN` | 客户明确表示不知道 | 可以降低门槛再问一次 |
| `DELEGATED` | 客户授权 AI 决定 | 不再追问，允许推导 |
| `DECLINED` | 客户明确不愿提供 | 不再追问 |
| `DEFERRED` | 当前不再追问，暂时延后 | 只阻塞依赖该字段的 Action |

注意：

`UNKNOWN` 不等于 `DEFERRED`。

推荐流程：

```text
MISSING
  ↓
第一次询问
  ↓
UNKNOWN
  ↓
降低回答门槛
  ↓
再次不知道
  ↓
DEFERRED
```

---

# 4. 核心设计原则

## 4.1 同一个字段禁止无限追问

规则：

```text
同一字段：
正常询问最多 1 次
如果客户不知道：
最多再进行 1 次低门槛询问
仍然不知道：
DEFERRED
```

禁止：

```text
ASK
→ ASK
→ ASK
→ ASK
→ ASK
```

---

## 4.2 “客户不知道”和“客户让 AI 决定”必须区分

例如：

```text
I don't know.
```

应该：

```text
UNKNOWN
```

而：

```text
You decide.
You recommend.
Whatever you think is suitable.
```

应该：

```text
DELEGATED
```

二者绝对不能都保存成：

```text
value = None
```

否则后续 Gate 无法知道客户真实意图。

---

## 4.3 “客户拒绝提供”也不能等同于不知道

例如：

```text
I don't want to provide that.
```

应该：

```text
DECLINED
```

后续不再询问。

---

## 4.4 缺失参数只阻塞需要它的 Action

例如：

```text
viewing_distance = DEFERRED
```

不应该自动意味着：

```text
Recommendation = BLOCKED
Calculation = BLOCKED
Sales Conversation = BLOCKED
```

应该分别判断。

例如：

```text
Recommendation:
READY

Calculation:
DEFERRED
```

---

# 5. Phase 1：新增 Customer Response Intent ✅ 已完成

## 5.1 新增模块

建议新增：

```text
src/core/customer_response.py
```

职责：

> 识别客户针对上一轮问题的回答意图。

---

## 5.2 输入

至少包括：

```python
current_message
last_asked_slot
conversation_context
```

---

## 5.3 输出

建议结构：

```json
{
  "slot": "width",
  "intent": "delegated",
  "evidence": "you recommend",
  "confidence": 0.98
}
```

---

## 5.4 支持的 Intent

### UNKNOWN

典型表达：

```text
I don't know.
No idea.
Not sure.
I'm not sure.
I don't have the measurements.
I don't know yet.
```

---

### DELEGATED

典型表达：

```text
You decide.
You recommend.
You choose.
Whatever you recommend.
You can decide.
Just recommend something suitable.
```

---

### DECLINED

典型表达：

```text
I don't want to provide that.
That doesn't matter.
I don't care about that.
I don't want to specify it.
```

注意：

“Doesn't matter”必须结合上下文判断。

---

### CONFIRMED

客户直接提供参数：

```text
Around 5 meters.
About 100 people.
It's indoors.
Fixed installation.
```

---

### CORRECTION

客户修改之前的信息：

```text
Actually, it's 5 meters, not 3.
```

应该更新原字段，而不是触发重复询问。

---

# 6. Phase 2：扩展 RequirementProfile ✅ 已完成

修改：

```text
src/models/requirement.py
```

增加：

```python
field_decisions: Dict[str, str]
```

示例：

```python
field_decisions = {
    "width": "delegated",
    "viewing_distance_m": "deferred",
    "pixel_pitch": "confirmed"
}
```

---

## 6.1 增加统一方法

建议增加：

```python
def field_decision(self, slot: str) -> str:
    ...

def is_delegated(self, slot: str) -> bool:
    ...

def is_deferred(self, slot: str) -> bool:
    ...

def is_declined(self, slot: str) -> bool:
    ...

def is_exhausted(self, slot: str) -> bool:
    ...
```

---

## 6.2 保留现有字段

不要删除：

```text
ask_counts
unknown_reasons
last_asked_slot
sources
```

本次是扩展，不是重构掉原来的数据结构。

---

# 7. Phase 3：修改 Query Understanding / Requirement Extraction ✅ 已完成

当前：

```text
客户消息
↓
提取字段
```

升级为：

```text
客户消息
+
上一轮问题
+
last_asked_slot
↓
提取所有有效需求
+
识别客户对当前字段的决策态度
```

---

## 7.1 必须支持“一句话多个结果”

例如：

```text
I don't know the width, just recommend something suitable.
```

输出：

```json
{
  "width": {
    "value": null,
    "status": "DELEGATED"
  }
}
```

---

## 7.2 另一个例子

```text
I don't know the viewing distance either, you can decide the pitch.
```

输出：

```json
{
  "viewing_distance_m": {
    "value": null,
    "status": "UNKNOWN"
  },
  "pixel_pitch": {
    "value": null,
    "status": "DELEGATED"
  }
}
```

不能因为当前问题是 viewing distance，就只处理 viewing distance。

---

# 8. Phase 4：修改 Question Planner ✅ 已完成

修改：

```text
src/agents/sales/question_planner.py
```

当前逻辑：

```text
字段没有值
↓
进入问题计划
```

升级为：

```text
CONFIRMED
→ SKIP

INFERRED
→ SKIP

DELEGATED
→ SKIP

DECLINED
→ SKIP

DEFERRED
→ SKIP

EXHAUSTED
→ SKIP

MISSING
→ ASK

UNKNOWN
→ EASIER ASK
```

---

## 8.1 同一个字段最多两次接触

建议：

```text
第一次：
正常问题

第二次：
更低门槛问题

第二次仍失败：
DEFERRED
```

例如：

### 第一次

```text
What is the approximate viewing distance?
```

客户：

```text
I don't know.
```

### 第二次

```text
No problem. Would it be roughly 3–5m, 5–10m, or farther?
```

客户：

```text
Still not sure.
```

最终：

```text
viewing_distance = DEFERRED
```

之后：

> 不允许 Question Planner 再选择 viewing_distance。

---

# 9. Phase 5：修改 Recommendation Ready Gate ✅ 已完成

重点修改：

```text
src/rag/readiness.py
```

当前核心问题：

```python
if not profile.has_target_size:
    hard_missing.append("size")
```

这种判断只关注：

```text
有没有值
```

应该升级为：

```text
有没有值
+
客户是否授权 AI 决定
+
是否已经 Deferred
+
是否存在可替代推导路径
```

---

# 10. 建立 Field Policy ✅ 已完成（`src/rag/field_policy.py`）

建议新增：

```text
src/rag/field_policy.py
```

或者放入现有 readiness 相关模块。

示例：

```python
FIELD_POLICIES = {
    "environment": {
        "required_for_recommendation": True,
        "can_infer": True,
        "can_delegate": False,
        "on_exhausted": "BLOCK",
    },

    "installation": {
        "required_for_recommendation": True,
        "can_infer": False,
        "can_delegate": True,
        "on_exhausted": "DEGRADED",
    },

    "viewing_distance_m": {
        "required_for_recommendation": False,
        "required_for_calculation": True,
        "can_infer": False,
        "can_delegate": True,
        "on_exhausted": "DEFER",
    },

    "pixel_pitch": {
        "required_for_recommendation": False,
        "can_infer_from_distance": True,
        "can_delegate": True,
        "on_exhausted": "DEFER",
    },

    "size": {
        "required_for_recommendation": False,
        "required_for_calculation": True,
        "can_infer": True,
        "can_delegate": True,
        "on_exhausted": "DEFER_CALCULATION",
    }
}
```

---

# 11. Gate 判断顺序 ✅ 已实现（`field_action()` 按此顺序判断）

每一个缺失字段按照下面顺序处理：

```text
字段缺失
   ↓
客户是否 DELEGATED？
   ↓ YES
允许 AI / Python 推导
   ↓ NO
是否可以可靠 INFER？
   ↓ YES
进入推导
   ↓ NO
是否 DEFERRED？
   ↓ YES
根据当前 Action 决定是否阻塞
   ↓ NO
是否值得再次 ASK？
   ↓ YES
询问
   ↓ NO
BLOCK / DEGRADED
```

---

# 12. Recommendation Gate 的新规则 ✅ 已实现（READY / DEGRADED_READY / CONTINUE_ASKING / BLOCKED）

最终：

```text
Recommendation Gate
```

应该返回：

```text
READY
DEGRADED_READY
CONTINUE_ASKING
BLOCKED
```

其中：

### READY

所有推荐必需条件满足。

### DEGRADED_READY

部分非关键参数未知，但仍可合理推荐。

### CONTINUE_ASKING

存在一个真正值得询问、且没有达到询问上限的关键参数。

### BLOCKED

参数无法推导、客户没有授权、已经无法继续，并且该参数确实是推荐不可绕过的硬性条件。

---

# 13. Calculation Gate 的新规则 ✅ 已实现（与 Recommendation 完全独立判断）

Recommendation 和 Calculation 必须继续保持独立。

例如：

```text
screen_size = DEFERRED
```

结果：

```text
Recommendation:
READY

Calculation:
DEFERRED
```

而不是：

```text
Recommendation:
BLOCKED
```

---

# 14. “DELEGATED”必须进入确定性推导 ✅ 已实现（Python 推导，LLM 只负责表达）

例如：

```text
width = DELEGATED
```

不要让 LLM 随便编一个宽度。

应该：

```text
width = DELEGATED
↓
Recommendation Engine / Parameter Inference
↓
根据：
- viewing distance
- people
- scene
- product constraints
- cabinet dimensions
- installation constraints
↓
计算合理候选
```

LLM 只负责最终销售表达。

---

# 15. Recommendation Engine 的处理规则 ✅ 已实现

现有确定性推荐引擎继续保留。

新增：

```python
if profile.is_delegated("width"):
    width = infer_width(profile)
```

以及：

```python
if profile.is_delegated("pixel_pitch"):
    pixel_pitch = infer_pixel_pitch(profile)
```

禁止：

```python
LLM → 猜一个参数 → 当作事实
```

---

# 16. “硬性参数”的重新定义 ✅ 已实现（字段策略表见 `src/rag/field_policy.py`）

不要再简单定义：

```text
hard parameter = missing → BLOCK
```

改成：

```text
hard parameter = 某个 Action 对该字段有依赖
```

例如：

| 参数 | 推荐 | 计算 | 缺失策略 |
|---|---|---|---|
| 室内/室外 | 强依赖 | 强依赖 | 无法可靠判断时 BLOCK |
| 固装/租赁 | 强依赖 | 相关 | 可根据场景/客户授权处理 |
| Viewing Distance | 非绝对阻塞 | 强依赖 | DEFER |
| Width | 非绝对阻塞 | 强依赖 | DELEGATED/DEFER |
| Height | 非绝对阻塞 | 强依赖 | DELEGATED/DEFER |
| Pixel Pitch | 可推导 | 强相关 | 根据距离/场景推导 |
| Purpose | 强相关 | 非绝对阻塞 | 优先询问，但不无限追问 |
| Content Type | 非硬性 | 非硬性 | 不阻塞推荐 |
| Budget | 非硬性 | 非硬性 | 不阻塞推荐 |

最终目标：

> **不是所有“缺失”都阻塞推荐。**

---

# 17. 推荐的 Action Planner ✅ 已实现（`plan_actions()` / `ask_candidates()`）

可以增加一个统一的：

```text
Action Planner
```

逻辑：

```text
RequirementProfile
       ↓
Action Planner
       ↓
┌─────────────┬──────────────┬──────────────┐
│ ASK         │ RECOMMEND    │ CALCULATE    │
└─────────────┴──────────────┴──────────────┘
```

它判断：

```text
现在应该：
1. 问客户
2. 自己推导
3. 推荐
4. 暂缓计算
5. 真正阻塞
```

这样以后增加报价、方案书、产品对比等功能也可以复用。

---

# 18. 典型场景测试 ✅ 已覆盖（Case 1~5 见 `tests/test_recommendation_gate_v21.py`）

## Case 1：客户授权 AI 决定尺寸

输入：

```text
AI:
What's the approximate width?

Customer:
No range, you recommend.
```

预期：

```text
width.status = DELEGATED
```

并且：

```text
Question Planner:
不再问 width
```

---

## Case 2：客户不知道宽度

```text
AI:
What's the approximate width?

Customer:
I don't know.
```

第一次：

```text
width = UNKNOWN
ask_count = 1
```

再次询问：

```text
Could it be roughly 3m, 5m, or 8m?
```

仍然：

```text
I have no idea.
```

最终：

```text
width = DEFERRED
```

以后：

```text
Question Planner:
不再问 width
```

---

## Case 3：Viewing Distance 不知道，但已有 Pitch

```text
pixel_pitch = P2.5
viewing_distance = DEFERRED
```

结果：

```text
Recommendation:
READY
```

但：

```text
Calculation:
根据需要决定是否 DEFERRED
```

---

## Case 4：客户同时授权多个参数

输入：

```text
Indoor church, fixed installation.
You decide the size and pitch.
```

应该得到：

```text
environment = CONFIRMED
purpose = CONFIRMED
installation = CONFIRMED
size = DELEGATED
pixel_pitch = DELEGATED
```

然后：

```text
Recommendation Engine
↓
开始确定性推导
```

---

## Case 5：真正无法继续

客户：

```text
I need an LED display.
```

AI：

```text
Indoor or outdoor?
```

客户：

```text
I don't know.
```

再次低门槛询问后：

```text
Still not sure.
```

如果没有任何可靠场景线索：

```text
environment = DEFERRED
```

且环境是当前产品硬过滤必需条件：

```text
Recommendation Gate = BLOCKED
```

此时只需要说明：

```text
To narrow down the suitable models, I need to know whether this will be used indoors or outdoors, because the product requirements are different.
```

不继续追问其他无关字段。

---

# 19. 单元测试计划 ✅ 已完成

新增：

```text
tests/test_customer_response_intent.py
tests/test_requirement_decision_state.py
tests/test_deferred_fields.py
tests/test_delegated_fields.py
tests/test_recommendation_gate_v21.py
```

至少覆盖：

- `I don't know` → UNKNOWN
- `Not sure` → UNKNOWN
- `You decide` → DELEGATED
- `You recommend` → DELEGATED
- `I don't care` → DECLINED
- 客户直接提供参数 → CONFIRMED
- 客户修改之前参数 → CORRECTION
- UNKNOWN → easier question
- UNKNOWN 第二次 → DEFERRED
- DEFERRED → 禁止再次询问
- DELEGATED → 禁止再次询问
- DELEGATED → 进入 Python 推导
- DEFERRED 参数只阻塞依赖它的 Action
- 真正不可绕过参数 → BLOCKED

---

# 20. 日志与调试 ✅ 已实现（`[FieldDecision]` / `[ActionPlanner]` 结构化日志）

每次 Sales Agent 决策增加结构化日志：

```json
{
  "slot": "viewing_distance_m",
  "previous_state": "UNKNOWN",
  "new_state": "DEFERRED",
  "ask_count": 2,
  "action": "skip_question",
  "reason": "customer_does_not_know"
}
```

对于授权：

```json
{
  "slot": "width",
  "previous_state": "MISSING",
  "new_state": "DELEGATED",
  "action": "infer",
  "reason": "customer_authorized_ai"
}
```

Gate：

```json
{
  "gate": "recommendation",
  "status": "DEGRADED_READY",
  "blocked_slots": [],
  "deferred_slots": ["viewing_distance_m"]
}
```

这样以后出现：

> “为什么 AI 没问？”

或者：

> “为什么 AI 还在问？”

可以直接从日志定位。

---

# 21. 不建议做的事情 ✅ 已遵守（未堆 Prompt、未让 LLM 决定参数、DEFERRED 不当 CONFIRMED）

## 不要 1：继续堆 Prompt

不要试图通过：

```text
“如果客户说 I don't know 就不要一直问”
```

解决全部问题。

因为最终还是状态管理问题。

---

## 不要 2：把所有硬性参数都改成非必填

这会造成：

```text
客户只说：
I need an LED screen.
```

系统也开始推荐。

这会破坏你之前建立的 Recommendation Ready Gate。

---

## 不要 3：让 LLM 自己决定参数

禁止：

```text
客户不知道尺寸
↓
LLM 随便猜一个
↓
进入推荐
```

必须：

```text
DELEGATED
↓
确定性推导
↓
Recommendation Engine
```

---

## 不要 4：把 DEFERRED 当成 CONFIRMED

例如：

```text
viewing_distance = DEFERRED
```

绝不能在后续输出中说：

```text
Based on your 5m viewing distance...
```

因为客户从未确认过 5m。

---

# 22. 最终架构 ✅ 已落地（见第 26 节实施记录）

优化后的完整链路：

```text
Customer Message
       │
       ▼
Query Understanding
       │
       ▼
Requirement Extraction
       │
       ▼
Customer Response Intent
       │
       ├── CONFIRMED
       ├── UNKNOWN
       ├── DELEGATED
       ├── DECLINED
       └── CORRECTION
       │
       ▼
RequirementProfile
       │
       ▼
Field State Manager
       │
       ▼
Action Planner
       │
       ├──────────────┬───────────────┐
       ▼              ▼               ▼
      ASK           INFER          RECOMMEND
       │              │               │
       ▼              ▼               ▼
 Question         Python          Recommendation
 Planner          Inference         Engine
       │              │               │
       └──────────────┴───────┬───────┘
                              ▼
                     Calculation Gate
                              │
                              ▼
                       Screen Calculator
                              │
                              ▼
                         Validation
                              │
                              ▼
                         Final LLM
```

---

# 23. 实施顺序（✅ = 本次已完成）

建议严格按照下面顺序开发：

```text
Phase 1
Customer Response Intent
        ↓
Phase 2
RequirementProfile 状态扩展
        ↓
Phase 3
Query Understanding / Extraction 接入
        ↓
Phase 4
Question Planner 状态化
        ↓
Phase 5
Recommendation Gate 改造
        ↓
Phase 6
Calculation Gate 改造
        ↓
Phase 7
Recommendation Engine Delegated 推导
        ↓
Phase 8
完整测试
        ↓
Phase 9
Golden Dataset 增加边界样本
        ↓
Phase 10
线上日志验证
```

### 完成情况

```text
Phase 1  Customer Response Intent                ✅ 已完成
Phase 2  RequirementProfile 状态扩展              ✅ 已完成
Phase 3  Query Understanding / Extraction 接入    ✅ 已完成
Phase 4  Question Planner 状态化                  ✅ 已完成
Phase 5  Recommendation Gate 改造                 ✅ 已完成
Phase 6  Calculation Gate 改造                    ✅ 已完成
Phase 7  Recommendation Engine Delegated 推导      ✅ 已完成
Phase 8  完整测试                                 ✅ 已完成（全量 1135 passed / 4 skipped）
Phase 9  Golden Dataset 增加边界样本               ✅ 已完成（g073~g082）
Phase 10 线上日志验证                              ⏸ 本次不实施（用户指定排除线上验证环境）
```

---

# 24. 验收标准

完成后必须满足：

### 客户授权 AI

```text
“You decide.”
```

结果：

```text
DELEGATED
```

不能再次询问同一字段。

---

### 客户不知道

第一次：

```text
UNKNOWN
```

第二次仍不知道：

```text
DEFERRED
```

不能第三次询问同一字段。

---

### 推荐

`DEFERRED` 参数不应自动阻塞所有推荐。

---

### 计算

只有真正需要该参数的计算才被延迟。

---

### 真正硬阻塞

只有：

```text
无法推导
+
没有替代信息
+
客户没有授权 AI 决定
+
该参数确实是当前 Action 的必要条件
```

才进入：

```text
BLOCKED
```

---

### 确定性

LLM 不得把：

```text
UNKNOWN
DEFERRED
DELEGATED
```

直接变成客户确认事实。

---

# 25. 最终目标

本次 v2.1 的核心不是让 AI “少问几个问题”。

而是让 Sales Agent 从：

```text
字段采集机器人
```

升级成：

```text
客户决策状态驱动的销售 Agent
```

最终实现：

```text
客户知道
→ 使用

客户不知道
→ 降低门槛问一次

客户还是不知道
→ DEFERRED

客户说你决定
→ DELEGATED

客户拒绝提供
→ DECLINED

可以计算
→ Python 推导

不能计算
→ 延迟计算

真正无法继续
→ BLOCKED

整个过程中
→ 同一个问题不无限追问
```

**最重要的工程原则：**

> **Missing 不等于 Blocked。**
>
> **Unknown 不等于 Ask Forever。**
>
> **Delegated 不等于 Guess。**
>
> **Deferred 不等于 Confirmed。**
>
> **一个字段只应该阻塞依赖它的 Action。**

---

# 26. 实施记录（2026-09-20）

## 26.1 交付物

| 类型 | 文件 | 说明 |
|---|---|---|
| 新增 | `src/core/customer_response.py` | Phase 1：客户回答意图（纯规则，零 LLM） |
| 新增 | `src/rag/field_policy.py` | Phase 5 / 10 / 17：Field Policy + Action Planner + 跨槽位依赖 |
| 修改 | `src/models/requirement.py` | Phase 2：`field_decisions` 状态 + 槽位名归一 + 合并规则 |
| 修改 | `src/agents/sales/nodes/requirement.py` | Phase 3：意图接入 + `[FieldDecision]` 结构化日志 |
| 修改 | `src/agents/sales/question_planner.py` | Phase 4：状态化（CONFIRMED/INFERRED/DELEGATED/DECLINED/DEFERRED → 不问） |
| 修改 | `src/rag/readiness.py` | Phase 5 + 6：Recommendation / Calculation Gate 改造 |
| 修改 | `src/rag/parameter_inference.py` | Phase 7：`suggest_screen_size()` 确定性推导 |
| 修改 | `src/rag/recommendation_engine.py` | Phase 7：DELEGATED 参数走确定性推导并留日志 |
| 修改 | `src/agents/solution/nodes/recommend.py` | Phase 7：用推导出的参考尺寸算箱体，不写入客户事实 |
| 新增 | `tests/test_customer_response_intent.py` | Phase 8：41 条 |
| 新增 | `tests/test_requirement_decision_state.py` | Phase 8：15 条 |
| 新增 | `tests/test_deferred_fields.py` | Phase 8：10 条 |
| 新增 | `tests/test_delegated_fields.py` | Phase 8：12 条 |
| 新增 | `tests/test_recommendation_gate_v21.py` | Phase 8：13 条（含计划第 18 节 Case 1~5） |
| 修改 | `eval/golden_dataset.json` | Phase 9：新增 g073~g082 边界样本 + `decision` 字段说明 |

## 26.2 字段策略（落地口径）

| 字段 | 推荐 Gate | 计算 Gate | 问不出来时 | 可授权 AI |
|---|---|---|---|---|
| environment | 强依赖 | 强依赖 | **BLOCKED**（附英文说明，只问这一项） | 否 |
| installation | 强依赖 | 相关 | DEGRADED（按场景默认固装） | 是 |
| pixel_pitch | 可推导（观看距离 / 环境 + 场景） | 强相关 | DEFER | 是 |
| viewing_distance_m | 不阻塞 | 相关 | DEFER | 是 |
| size / width / height | 不阻塞（授权时用观看距离推导参考尺寸） | **强依赖** | DEFER_CALCULATION（只挂计算） | 是 |
| purpose | 非绝对阻塞（不再作为 Gate 的追问项） | 不阻塞 | DEGRADED | 是 |
| content_type / price_preference / budget | 不阻塞（从不主动问） | 不阻塞 | 不阻塞 | 是 |

## 26.3 与旧口径的差异（重要）

1. **尺寸 / 点间距 / 观看距离问满两次仍拿不到 → DEFERRED**：不再无限追问，推荐照常
   （状态可能是 `DEGRADED_READY`），只有依赖尺寸的**计算**被挂起。旧口径"硬性条件没拿到
   就一直问"由本计划第 4.1 / 16 节取代；**没问过（MISSING）的硬性条件仍然一定会问**
   （保留"客户说了一堆无关的话、最后要推荐时必须再问一次缺的硬性条件"）。
2. **室内外仍然是不可绕过的硬阻塞**：客户说"你决定"也不能替他决定，两次问不出来 →
   `BLOCKED`，且只说明"需要先确认室内还是室外"。
3. **场景（purpose）不再阻塞推荐**：客户口径「尺寸 + P 值 + 室内外 + 固装租赁齐了就推荐，
   别再问别的」，所以 purpose 只参与打分。
4. **点间距与观看距离成对处理**：先问 P 值；客户说不知道 → 转问观看距离；
   客户给了 P 值（或授权 AI 决定 P 值）→ 不再问观看距离。
   客户明确给了观看距离而没给 P 值 → 直接用观看距离推导点间距，不再回头问 P 值。
5. **DELEGATED 一律走 Python 确定性推导**：LLM 不参与参数取值；推导结果只用于工程计算与
   话术参考，**不写进客户的确认事实**（`derived_size_m` 只存在于本轮 Gate 结果里）。

## 26.4 结构化日志（第 20 节）

```text
[FieldDecision] {"slot": "pixel_pitch", "new_state": "DEFERRED", "ask_count": 1, "action": "skip_question", "reason": "unknown", "evidence": "I don't know"}
[ActionPlanner] {"actions": {...}, "blocked": [], "deferred": ["pixel_pitch"], "askable": ["viewing_distance"], "infer": []}
[RecommendationGate] status=CONTINUE_ASKING missing=['viewing_distance'] unknown=[]
```

## 26.5 验证结果

```text
python -m pytest tests/ -q
→ 1135 passed, 4 skipped（基线 1044 passed / 4 skipped，新增 91 条 v2.1 测试）
```

## 26.6 未实施

```text
Phase 10 线上日志验证  ⏸ 按用户要求本次排除（不在本地/线上日志环境跑真实会话）
```

---

## 26.7 追加：观看距离确定性推导 + 点间距物理窗口（v2.2，2026-09-20）

实测 bug：客户说 "indoor permanent 10x5m wall for around 100 viewers" 时，系统推荐了
**TW11-3216-P1.2**（该系列最细最贵的型号）。根因：没有观看距离 → `preferred_pitch_for_environment`
返回空 → `_score` 里 `applicable` 过滤掉 `pitch` 维度 → 同系列所有型号完全平分 →
排序兜底 `item.model.pixel_pitch_mm`（"点间距小的优先"）挑中最细的型号。

**修法（不再按客户说法加规则，而是折算成物理量）**

| 位置 | 改动 |
|---|---|
| `src/rag/query_understanding.py` | 新增 `_extract_space_facts()`：从原话解析人数 / 面积 / 进深（纯规则） |
| `src/models/requirement.py` | 新增 `audience_count` / `room_area_sqm` / `room_depth_m` 三个**事实**字段（只作为观看距离的来源，不参与 Gate） |
| `src/rag/parameter_inference.py` | 新增 `estimate_viewing_distance()`（4 个通用公式 + 优先级）、`pitch_window_for_distances()`（物理窗口）、`fallback_pitch_band()`（按环境的保守兜底档）；`infer_technical_parameters()` 增加窗口收口与来源标记 |
| `src/rag/recommendation_engine.py` | 新增 `_pitch_tiebreak()`：并列时取接近窗口首选值的型号，没有首选值时**取偏粗的一端**，不再"点间距小的优先"；`_size_fit()` 只允许宽高都齐时才调用计算器 |

**验收（5 组输入，全部不再出现 P1.x）**

```text
10x5m + 100 人      → 距离 6.2~8.8m   窗口 P3.0~P5.9  → TW11-3216-P3.0
10x5m + 50 人       → 距离 4.3~6.1m   窗口 P3.0~P4.1  → TW11-3216-P3.0
10x5m + 进深 8m     → 距离 5.3~7.5m   窗口 P3.0~P5.0  → TW11-3216-P3.0
10x5m + 什么都不说   → 无            兜底 P2.5~P4.0  → TW11-3216-P3.0
25㎡ + 3m 宽屏       → 距离 2.5~7.8m   窗口 P1.6~P2.5  → TW11-3216-P2.5
DELEGATED 点间距 + 无视距 → 兜底档 → TW11-3216-P3.0（不再是 P1.2）
```

测试：`tests/test_viewing_distance_derivation.py`（36 条）；全量 `python -m pytest tests/ -q`
→ **1171 passed, 4 skipped**。

### 26.7.1 追问循环兜底（同日追加，源自客户实测日志）

实测日志：客户连回两次 `i dont konw`（know 拼错），系统把同一个问题问了第三、第四遍；
而且客户已经说了"大约 50 个人需要看的屏幕"，系统还在问"他们站多远"。三处根因：

| 根因 | 修法 |
|---|---|
| `i dont konw` 拼写错误 → `UNKNOWN_RE` / `detect_no_answer` 全不命中 → 状态机学不到"不知道" | 新增 `normalize_customer_text()`（konw→know、idk/dunno/no clue/nt sure 归一化），`customer_response` 与 `unknown_detector` 共用；UNKNOWN 正则补充简写与中文变体 |
| "问满两次就不许再问"只在意图识别命中时才生效 → 漏判即无限追问 | `field_policy.field_action()` 增加**记账守卫**：MISSING 且 `ask_count >= MAX_ASKS_PER_SLOT` → 直接按 `on_exhausted` 处理（DEFER / DEGRADE / BLOCK），与意图识别解耦 |
| 客户已给人数/面积/进深，观看距离仍被追问 | `field_action()` 对 `viewing_distance_m`：只要有场地几何事实 → `INFER`（推导），不再提问；`estimate_viewing_distance()` 支持"只有人数、没有屏尺寸"（座位区宽深比 2:1 → 排数 = √(人数÷2)） |

复刻实测对话后的走向：

```text
问 P 值      → "i dont konw" → DEFERRED（不再问 P 值）
问观看距离    → "i dont konw" → 降门槛再问一次（允许的第二次）
客户："大约50个人需要看的屏幕" → 记录 50 人 → 观看距离转为推导，直接问屏尺寸
问屏尺寸      → "i dont konw" → 降门槛再问一次 → DEFERRED
→ DEGRADED_READY：推荐 TW11-3216-P3.0（备选 TW21-3216-P3.0 / TW11-3216-P4.0）
```

测试：`tests/test_customer_response_intent.py`（拼写归一化）、`tests/test_deferred_fields.py`
（问满上限守卫 / 环境问满转 BLOCK / 场地事实替代距离问题）、`tests/test_viewing_distance_derivation.py`
（只有人数也能推）；全量 `python -m pytest tests/ -q` → **1190 passed, 4 skipped**。

### 26.7.2 场地信息的解析补全（同日追加）

客户回答"场地多大"时的说法必须接得住，而且**不能和屏体尺寸混在一起**：

| 客户说法 | 解析结果 | 推导 | 推荐 |
|---|---|---|---|
| `the room is 25 sqm` / `大概 30 平` / `25㎡` / `25 m2` | 面积 | 无屏宽时进深 ≈ √(面积÷2) → 最远 3.0m | TW11-3216-P2.0 |
| `the room is 8m x 5m` / `场地 8米x5米` / `room 8m by 5m` | 面积 40㎡ + 进深 8m | 最远 7.5m | TW11-3216-P3.0 |
| `the hall is 10m wide and 6m deep` / `5米宽8米深` | 面积 60㎡ + 进深 6m | 最远 5.5m | TW11-3216-P3.0 |
| `about 50 people` / `大约50个人` | 人数 | 座位区 2:1 → 5 排 → 最远 7.0m | TW11-3216-P3.0 |
| `the screen is 10m wide and 6m deep` | **屏体尺寸**（不是场地） | 屏高 → 最近/最远 | 按屏尺寸推 |

新增解析能力：面积支持 `平米 / 平方米 / ㎡ / m2 / sqm / square metres / 平（口语）`；
场地尺寸支持"成对写法（x / × / by / 乘）"和"带方向词（wide / deep / 宽 / 深）"两种，
并用"最近的上下文词"判断这句话说的是**房间**还是**屏幕**（`room/hall/场地/大厅` vs
`screen/display/屏幕`）。命中场地时不写入屏体尺寸，屏体尺寸继续问客户。

测试：`tests/test_viewing_distance_derivation.py`（新增 10 条，含"屏体尺寸不被误判为场地"
与反向校验）；全量 `python -m pytest tests/ -q` → **1197 passed, 4 skipped**。

### 26.7.3 追问话术不再漏出内部槽位名（同日追加）

实测日志（客户回"需要50人观看的屏幕"之后）：系统回了两句，
一句问尺寸，另一句是

```text
To recommend the right products for you, could you tell me: distance?
```

两个问题：

| 问题 | 根因 | 修法 |
|---|---|---|
| 把内部槽位名 `distance` 抛给客户 | Solution 侧 `clarify_node` 的旧分支拿 LLM 的 `missing_info` 拼问句（`f"... could you tell me: {missing[0]}?"`） | `clarify_node` 先做确定性判断：有 `RequirementProfile` → 一律用 Gate 的问句；Gate 已 ready → 本轮不问；旧分支保留但槽位名必须经 `human_label()` 翻译成人话。`question_for()` 对未知槽位退回通用问句，**任何**情况下都不会输出字段名 |
| 客户已经说了"50 人"，还在追问观看距离 | 同上，旧分支用的是 LLM 的 missing_info 而不是字段决策状态 | 断言加在 `tests/test_question_safety.py`：`需要50人观看的屏幕 / about 50 people / 50 viewers` → Gate 的 `missing` 里不得出现 `viewing_distance` |

测试：`tests/test_question_safety.py`（32 条：槽位名不漏出、clarify 优先用 Gate、
Gate 就绪时不追问、人数已给时不问距离）；全量 `python -m pytest tests/ -q`
→ **1229 passed, 4 skipped**。
