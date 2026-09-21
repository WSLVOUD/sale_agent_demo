# LED RAG 智能销售系统：僵硬话术优化工程计划

> 版本：v1.0  
> 目标：解决当前 Sales Agent 回复“像问卷、像模板、像机器人”的问题，让客户侧回复更接近真实 B2B LED 销售人员。  
> 核心原则：**Python 决定“说什么”，LLM 决定“怎么自然地说”。**

---

## 1. 当前问题

当前虽然已经有 `DialogueAction`、`ResponseContext`、`ResponseGenerator`、`ResponseValidator`，但实际主链路仍存在较强的模板驱动：

```text
业务逻辑
  ↓
固定回复模板
  ↓
ACK / Echo / Connector / Bridge
  ↓
LLM Polish
  ↓
Validator
  ↓
客户
```

因此即使增加更多 ACK、连接词和同义句，客户感知仍然是：

> AI 按照问卷一题一题地问，而不是在进行真实销售沟通。

典型结构是：

```text
“Got it / Thanks...”
→ 重复客户信息
→ “Based on that / By the way / So...”
→ 继续问问题
```

**本次不继续堆模板，而是调整回复生成架构。**

---

# 2. 优化目标

最终客户回复链路改成：

```text
Customer Message
      ↓
Message Aggregator
      ↓
Requirement Extraction
      ↓
RequirementProfile
      ↓
Engineering / Decision
      ↓
DialogueAction
      ↓
ResponseContext
      ↓
ResponseGenerator
      ↓
ResponseValidator
      ↓
Customer
```

核心变化：

```text
旧：Business Logic → Template → LLM Polish → Customer

新：Business Context → LLM Native Generation → Validator → Customer
```

`ResponseGenerator` 成为**唯一客户可见文本出口**。

---

# 3. 核心设计原则

## 3.1 不再让模板决定句子结构

不要继续通过：

```python
ACK + ECHO + CONNECTOR + QUESTION
```

拼装客户话术。

`_ACK_ECHO_LEADS`、`_CONNECTORS`、`_BRIDGES`、`_ACK_GENERIC` 等不再承担最终话术结构职责。

---

## 3.2 Python 决定业务，LLM 决定表达

Python 负责：

- 已知信息
- 缺失信息
- 当前 Action
- 必须询问的问题
- 是否允许推荐
- 推荐结果
- 工程约束
- 不能改变的事实

LLM 负责：

- 是否需要 ACK
- 是否重复客户信息
- 句子长度
- 表达方式
- 是否直接进入问题
- 自然语言组织

原则：

> **LLM 不能改变业务决策，但可以自由决定自然表达方式。**

---

# 4. 第一阶段：重构 ResponseContext ✅ 已完成（v2.5++）

`ResponseContext` 改为纯结构化上下文，不保存拼好的销售句子。

建议：

```python
ResponseContext(
    action="ASK",
    required_question="viewing_distance",
    known_facts=[
        "indoor",
        "church",
        "screen_size_4m_x_2m"
    ],
    missing_facts=["viewing_distance"],
    business_goal="narrow_pixel_pitch_window",
    recommendation=None,
    engineering_constraints=[],
    language="en",
    restrictions=[
        "ask_only_one_question",
        "do_not_invent_facts",
        "do_not_repeat_customer_unnecessarily"
    ],
    style="natural_b2b_sales"
)
```

禁止：

```python
draft = "Thanks for sharing..."
```

正确方式：

```python
required_question = "viewing_distance"
```

---

# 5. 第二阶段：重构 ResponseGenerator ✅ 已完成（v2.5++）

## 5.1 成为唯一客户文本出口

统一：

```text
DialogueAction
    ↓
ResponseContext
    ↓
ResponseGenerator
    ↓
final_text
```

其他模块不得直接负责最终销售回复。

## 5.2 删除 Draft → Polish

不要：

```text
固定话术
  ↓
LLM 润色
```

改成：

```text
结构化业务上下文
  ↓
LLM 直接生成最终回复
```

因为如果输入本身已经是模板句子，LLM 再怎么润色也很容易有“AI 在努力装成人”的感觉。

---

# 6. 第三阶段：重写 ResponseGenerator Prompt ✅ 已完成（v2.5++）

建议核心约束：

```text
You are an experienced B2B LED sales consultant.

Respond naturally, like a real salesperson having a normal conversation.

The business decision has already been made by the system.
Preserve all provided facts, recommendations, constraints and required questions.

You are NOT required to:
- acknowledge every customer message
- repeat what the customer just said
- use a transition phrase
- explain why you are asking
- follow a fixed sentence structure

Sometimes the best reply is one short question.
Sometimes it is a direct answer.
Sometimes it is a short comment followed by one question.

Do not sound like a questionnaire.
Do not sound like a scripted chatbot.
Do not use sales filler.
Do not force phrases such as:
"Got it", "Thanks", "Based on that", "By the way", "So", "That said".

Ask no more than one question.

Do not change the business decision.
Do not invent specifications, prices, models, lead times,
engineering results, or commitments.

Keep the response concise and conversational.
```

---

# 7. 第四阶段：重新定义 DialogueAction ✅ 已完成（v2.5++，Action 只决定"做什么"）

保留：

```text
DIRECT_ANSWER
ASK
ANSWER_AND_ASK
RECOMMEND
CLARIFY
CONFIRM
ACK_ONLY
```

但明确：

> Action 只决定“做什么”，不决定“怎么说”。

例如：

```text
Action = ASK
Question = viewing_distance
```

不意味着必须生成：

```text
Thanks for the information. Based on that, may I ask...
```

可以直接：

```text
How far will the audience be from the screen?
```

表达方式交给 LLM。

---

# 8. 第五阶段：收缩 reply_composer.py 职责 ✅ 已完成（v2.5++，定位改为 Fact / Format Utility）

`reply_composer.py` 可以保留：

- 产品事实格式化
- 数值格式化
- 单位格式化
- 语言工具
- 安全清洗
- 输出辅助

逐步退出：

- ACK 选择
- Connector 选择
- Bridge 选择
- 客户信息 Echo
- Question 前置话术
- 销售模板拼接

最终定位为：

```text
Fact / Format Utility
```

而不是：

```text
Dialogue Sentence Composer
```

---

# 9. 第六阶段：修改 script_generator.py ✅ 已完成（v2.5++，主链路改走 Context → Generator）

主路径不要继续：

```text
compose_requirement_reply()
→ _polish_question_message()
→ final response
```

改为：

```text
Requirement / Decision
→ DialogueAction
→ ResponseContext
→ ResponseGenerator
```

`script_generator.py` 负责：

- 销售决策上下文
- 问题规划
- 销售场景信息

不负责：

- ACK
- Connector
- Bridge
- 具体句子拼接
- Question wording

---

# 10. 第七阶段：调整 ResponseValidator ✅ 已完成（v2.5++，只守事实/边界，不点评话术）

Validator 应该是：

> **事实、安全、业务边界守门员**

必须检查：

- 是否回答客户明确问题
- 是否出现错误产品事实
- 是否出现不存在的型号
- 是否改变工程结论
- 是否虚构价格
- 是否虚构交期
- 是否超过一个问题
- 是否泄露内部字段
- 是否违反当前 Action

不应该强制：

- 必须 ACK
- 必须 Connector
- 必须 Bridge
- 必须复述客户
- 必须有销售过渡句

---

# 11. 第八阶段：建立自然回复策略 ✅ 已完成（v2.5++）

## 11.1 不要求每轮 ACK

客户：

```text
Indoor.
```

不要固定：

```text
Thanks for sharing. Got it — since this is for indoor use...
```

可以：

```text
Is it a fixed installation or a rental setup?
```

## 11.2 不机械复述

客户：

```text
It's for a church, around 200 people.
```

不要：

```text
Thanks. So this is an indoor church project for around 200 people.
```

可以：

```text
Roughly how far will the audience be from the screen?
```

## 11.3 不强制连接词

不要为了“自然”强行加入：

```text
So
By the way
Based on that
That said
Now
```

## 11.4 允许极短回复

如果只缺一个字段：

```text
What’s the approximate viewing distance?
```

直接问即可，不人为增加两三句。

---

# 12. 第九阶段：按场景决定表达 ✅ 已完成（v2.5++，7 种 Action 各自口径）

### ASK

只需要继续收集需求时，可以只问一个问题。

### DIRECT_ANSWER

客户明确提问时直接回答，不强行继续问。

### ANSWER_AND_ASK

先回答客户问题，再自然提出一个关键问题。

### RECOMMEND

满足条件后自然给出推荐，不继续机械询问。

### CLARIFY

客户表达含糊时，只问一个能转化成工程参数的问题。

例如：

```text
Customer:
I want something with a better effect.

不要：
What is your application?
What is your screen size?
What is your viewing distance?

应该：
Do you mean better image detail, or higher brightness?
```

---

# 13. 第十阶段：增加自然度指标 ✅ 已完成（v2.5++，10 项指标）

新增日志：

```text
generic_ack_rate
question_repeat_rate
connector_repeat_rate
customer_echo_rate
questionnaire_pattern_rate
customer_question_answer_rate
one_question_compliance
unsupported_fact_rate
internal_term_leak_rate
response_length
```

重点指标：

### Generic ACK Rate

统计：

```text
Got it / Thanks / Sure / Understood / I see
```

不是要求为 0，而是避免每轮固定出现。

### Customer Echo Rate

统计 AI 是否机械重复客户刚提供的信息。

### Connector Repeat Rate

统计：

```text
So / By the way / Based on that / That said / Now
```

等词的重复。

### Question Repeat Rate

统计是否重复询问客户已经回答的信息。

### Customer Question Answer Rate

客户明确提问时，系统是否真正回答了问题。

---

# 14. 第十一阶段：建立 Naturalness Golden Dataset ✅ 已完成（v2.5++）

新增专门测试“僵硬话术”的数据集。

例如：

```json
{
  "input": "Indoor.",
  "expected_action": "ASK",
  "required_question": "installation_type"
}
```

重点不是要求固定句子。

而是验证：

```text
✓ Action 正确
✓ 问题正确
✓ 不机械重复客户
✓ 不强制 ACK
✓ 不强制 Connector
✓ 不超过一个问题
✓ 无内部术语
✓ 无虚假事实
✓ 像正常销售对话
```

Golden Dataset 测：

> **行为约束，而不是固定文案。**

---

# 15. 第十二阶段：A/B Test ✅ 已完成（v2.5++，两套策略同批对比）

保留两套策略进行对比。

### Strategy A

```text
Template → LLM Polish
```

### Strategy B

```text
ResponseContext → LLM Native Generation
```

使用相同客户输入比较：

```text
ACK Rate
Echo Rate
Connector Rate
Question Repeat Rate
Question Answer Rate
Unsupported Fact Rate
平均回复长度
人工自然度评分
```

确认 Strategy B 稳定后再全量切换。

---

# 16. 第十三阶段：代码改造顺序 ✅ 已按此顺序执行（旧链路保留为回退）

严格按照：

```text
1. ResponseContext
      ↓
2. ResponseGenerator
      ↓
3. script_generator.py
      ↓
4. reply_composer.py
      ↓
5. ResponseValidator
      ↓
6. Naturalness Metrics
      ↓
7. Golden Dataset
      ↓
8. A/B Test
      ↓
9. 全量切换
```

不要一开始就删除旧模块，先建立新链路并保留旧链路作为回退方案。

---

# 17. 最终架构

```text
Customer
   ↓
Message Aggregator
   ↓
Requirement Extraction
   ↓
RequirementProfile
   ↓
Engineering / Decision
   ↓
DialogueAction
   ↓
ResponseContext
   ↓
┌──────────────────────┐
│  ResponseGenerator   │
│  ONE LLM TEXT OUTPUT │
└──────────┬───────────┘
           ↓
ResponseValidator
   ↓
Customer
```

---

# 18. 最终验收标准

优化完成后必须满足：

### 18.1 不再固定 ACK ✅

不是每条消息都：

```text
Got it.
Thanks.
I see.
```

### 18.2 不再固定复述 ✅

客户说：

```text
Indoor, church, 200 people.
```

不需要完整复述客户信息。

### 18.3 不再固定过渡词 ✅

不强制：

```text
So / By the way / Based on that / That said / Now
```

### 18.4 一次只问一个真正有价值的问题 ✅

如果只缺：

```text
viewing_distance
```

就只问：

```text
How far will the audience be from the screen?
```

### 18.5 客户提问必须优先回答 ✅

客户：

```text
Can this support 4K?
```

不能为了继续收集字段而直接问：

```text
What is your viewing distance?
```

应该先处理客户问题。

### 18.6 自然表达不能突破事实边界 ✅

LLM 可以改变：

```text
句式
长度
语气
连接方式
```

不能改变：

```text
型号
参数
推荐结果
工程计算
分辨率结论
价格
交期
```

---

---

## 实施记录（v2.5++，2026-09-21）✅ 全部阶段完成

### 一、代码改造清单（逐项对应本计划）

| 阶段 | 交付物 | 说明 |
|---|---|---|
| §4 第一阶段 | `src/dialogue/response_context.py` | `ResponseContext` 重构为**纯结构化上下文**：`action` / `known_facts` / `missing_facts` / `business_goal` / `required_question` / `engineering_constraints` / `language` / `restrictions` / `style`；`question` / `answer` 只承载"必须传达的内容"，不再保存拼好的销售句子；`prompt_block()` 只输出结构化决策 |
| §5 第二阶段 | `src/dialogue/response_generator.py` | `generate_response()` 成为**唯一客户文本出口**；Strategy B（默认）= 结构化上下文 → LLM 原生生成（**没有草稿**）；Strategy A = 旧"草稿 → 润色"，只用于 A/B 对比与回退；两条路都必须过 Validator，不合格退回结构化拼装 |
| §6 第三阶段 | `response_generator.NATIVE_SYSTEM_PROMPT` | 系统提示按计划原文落地：不要求 ACK / 不复述 / 不强制过渡词 / 不解释为什么问 / 一轮最多一个问题 / 不得改业务决策、不得编造参数价格交期 |
| §7 第四阶段 | `src/dialogue/action.py` | 7 种 `DialogueAction` 保留，职责明确为"只决定做什么"；表达方式全部交给 LLM |
| §8 第五阶段 | `src/rag/reply_composer.py` | 文件头写明新定位 **Fact / Format Utility**；ACK/Connector/Bridge/Echo/问题前置话术不再承担最终结构职责 —— `compose_requirement_reply()` 标注为 **Legacy · 仅作兜底** |
| §9 第六阶段 | `src/agents/sales/nodes/script_generator.py` | 新增 `_natural_reply()`（唯一出口）：需求采集、无关话题、交付/档期、异议与价格四类回复全部改走 `DialogueAction → ResponseContext → ResponseGenerator`；旧模板只在"没有 LLM / 生成失败 / 需求重置轮 / 带图核对轮"兜底 |
| §10 第七阶段 | `src/dialogue/response_validator.py` | 只守事实与边界：客户问题是否被回答、是否编造参数/型号/交期、是否改动工程结论、是否超一个问题、是否泄漏内部术语、是否违反 Action；**不再**要求 ACK / 连接词 / 过渡句 / 复述 |
| §11 第八阶段 | 同上 + `_opening_for()` | 不再要求每轮 ACK、不机械复述（泛客套与复述句直接不再进入上下文）、不强制连接词、允许极短回复 |
| §12 第九阶段 | `compose_from_context()` + Action | 各 Action 有独立表达口径：ASK 只问一个；DIRECT_ANSWER 直接答；ANSWER_AND_ASK 先答再问；RECOMMEND 给结论不再机械追问；CLARIFY 只问一个能转成工程参数的问题 |
| §13 第十阶段 | `response_validator.compute_metrics()` | 10 项自然度指标：`generic_ack_rate` / `question_repeat_rate` / `connector_repeat_rate` / `customer_echo_rate` / `questionnaire_pattern_rate` / `customer_question_answer_rate` / `one_question_compliance` / `unsupported_fact_rate` / `internal_term_leak_rate` / `response_length` |
| §14 第十一阶段 | `eval/naturalness_golden.json` + `tests/test_naturalness_golden.py` | 10 条黄金用例，逐条验证"Action 正确 / 问题正确 / 不复述 / 不强制 ACK / 不超一个问题 / 无内部术语 / 无虚假事实"，并统计整份数据集的指标 |
| §15 第十二阶段 | `src/dialogue/ab_test.py` + 同名测试 | `compare_strategies()` 用同一批输入跑 A/B，输出指标对比；测试断言 B 的 ACK / 问卷腔为 0、长度更短、一轮一问合规 |
| §16 第十三阶段 | 全流程 | 严格按顺序改（Context → Generator → script_generator → reply_composer → Validator → Metrics → Golden → A/B）；**旧链路没有删除**，保留为回退（`compose_requirement_reply` + 旧润色函数仍在） |

### 二、现在的客户回复链路

```text
Customer Message
    ↓  Message Aggregator（v2.5）
    ↓  Requirement Extraction → RequirementProfile
    ↓  Engineering / Decision（Gate / 可行性 / 推荐）
    ↓  DialogueAction（只决定"做什么"）
    ↓  ResponseContext（结构化：已知/缺失/目标/必问/约束）
    ↓  ResponseGenerator —— Strategy B：LLM 原生生成（默认）
    │                        Strategy A：草稿→润色（对比/回退）
    │                        无 LLM / 生成不合格 → 结构化拼装 → 旧模板兜底
    ↓  ResponseValidator（事实/安全/业务边界守门员）
    ↓  Customer
```

### 三、测试与验证

```text
tests/test_dialogue_naturalness.py    Action 判定、Context 结构、生成回退、校验器、10 项指标
tests/test_naturalness_golden.py      黄金数据集 10 条行为约束 + 数据集指标 + A/B 对比
tests/test_reply_composer.py          旧模板仍可用；"church" 那类场景不再复述客户原话

python -m pytest tests/ -q  →  1459 passed, 4 skipped
（本次优化：1446 → 1459，新增 13 条；无回归）
```

### 四、本阶段没做 / 保留

没有删除旧模块（`reply_composer` 的组合函数、`_polish_question_message` 都保留为回退）；
没有让 LLM 决定业务（问什么 / 推什么 / 能不能做仍然全部由 Python 决定）；
没有放开事实边界（编造参数、型号、价格、交期一律被 Validator 拦回结构化兜底）。

LLM 连续失败 2 次会触发 60s 熔断（`reset_llm_breaker()` 可手动恢复），熔断期间自动走结构化拼装，
避免每轮都等超时。

# 19. 最终结论

本次优化最重要的不是：

> “把模板写得更像人。”

而是：

> **让系统不再要求模型按照模板说话。**

最终形成清晰职责：

```text
Python
  ↓
决定：
“现在应该做什么？”

LLM
  ↓
决定：
“怎样像真实销售一样把这件事说出来？”

Validator
  ↓
保证：
“没有越过事实和业务边界。”
```

这才是解决当前“AI 很聪明，但说话很僵硬”的根本方案。
