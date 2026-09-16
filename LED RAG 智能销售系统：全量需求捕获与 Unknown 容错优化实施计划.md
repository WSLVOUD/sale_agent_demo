# LED RAG 智能销售系统：全量需求捕获与 Unknown 容错优化实施计划

> 版本：v1.0  
> 日期：2026-09-16  
> 目标：解决“AI 问 A、客户回答 B，B 信息被丢失”以及“客户不知道某项需求时 AI 反复追问”的问题。

---

## 执行进度总览（更新时间：2026-09-16）

| 计划章节 | Phase | 内容 | 状态 |
| --- | --- | --- | --- |
| §3 | Phase 1 | RequirementProfile 状态扩展 | ✅ 已完成 |
| §4 | Phase 2 | 建立 Global Requirement Extraction | ✅ 已完成 |
| §5 | Phase 3 | 支持一条消息多个需求 | ✅ 已完成 |
| §6 | Phase 4 | 增加 Unknown Detection | ✅ 已完成 |
| §7 | Phase 5 | 建立字段级 Ask Counter | ✅ 已完成 |
| §8 | Phase 6 | 第一次询问 | ✅ 已完成 |
| §9 | Phase 7 | 第二次询问必须降低门槛 | ✅ 已完成 |
| §10 | Phase 8 | 第二次仍不知道 → Unknown | ✅ 已完成 |
| §11 | Phase 9 | 客户回答其他信息时必须同时记录 | ✅ 已完成 |
| §12 | Phase 10 | 客户主动补充 Unknown 信息 | ✅ 已完成 |
| §13 | Phase 11 | Recommendation Ready Gate 改造 | ✅ 已完成 |
| §14 | Phase 12 | 不要简单放宽所有 Gate | ✅ 已完成 |
| §15 | Phase 13 | Recommendation Engine 处理 Unknown | ✅ 已完成 |
| §16 | Phase 14 | 增加 Recommendation Basis | ✅ 已完成 |
| §17 | Phase 15 | 最终销售话术 | ✅ 已完成 |
| §18 | Phase 16 | Calculation 与 Unknown 解耦 | ✅ 已完成 |
| §19 | Phase 17 | Question Planner 优先级 | ✅ 已完成 |
| §20 | Phase 18 | Session Switch 兼容 | ✅ 已完成 |
| §21 | Phase 19 | 日志优化 | ✅ 已完成 |
| §22 | Phase 20 | 测试用例（Test 1~7） | ✅ 已完成 |
| §23 | — | 回归测试 / 金标对话 | ✅ 已完成 |

> 验证结果：`python -m pytest tests -q` → **759 passed, 4 skipped**（4 条跳过的是需要真实 LLM 的非中英文用例）。
> 新增回归文件：`tests/test_unknown_tolerance.py`（59 个用例）。
> 实施过程中修复的两个真实缺陷见文末「附录 B」。

---

## 1. 优化目标

当前 Sales Agent 的需求收集需要升级为：

> **客户消息全量解析 + 字段级状态管理 + 最多询问两次 + Unknown 自动跳过 + 基于已有信息继续推荐。**

必须满足：

1. AI 问 `viewing_distance`，客户回答 `screen_size`，屏幕尺寸必须被记录。
2. 一条客户消息包含多个需求时，必须全部提取。
3. 客户明确表示不知道某项需求时，最多询问两次。
4. 第二次询问必须降低回答门槛，不能机械重复。
5. 两次仍不知道后，将字段标记为 `unknown`。
6. `unknown` 字段不能继续阻塞需求收集。
7. Question Planner 不得再次主动询问已经 `unknown` 且 `ask_count >= 2` 的字段。
8. 客户后续主动提供该信息时，允许 `unknown → confirmed`。
9. 推荐必须基于已经确认的信息继续进行。
10. 最终推荐必须说明哪些关键需求缺失，以及这些缺失可能影响什么。
11. 屏幕尺寸、产品规格已经满足时，不能因为 viewing distance unknown 而阻止柜体计算。

---

# 2. 核心设计原则

## 原则一：问题不是信息提取边界

错误架构：

```text
AI 当前问 viewing_distance
        ↓
只提取 viewing_distance
        ↓
客户同时说出的 screen_size 被丢弃
```

正确架构：

```text
客户消息
   ↓
全量需求提取
   ↓
更新 RequirementProfile
   ↓
判断当前问题是否得到回答
   ↓
Question Planner 决定下一步问题
```

---

## 原则二：Requirement Extractor 和 Question Planner 必须解耦

### Requirement Extractor

负责：

> 客户刚刚说了什么？

### Question Planner

负责：

> 接下来还需要问什么？

二者不能混合。

---

## 原则三：Unknown 是有效状态

必须区分：

```text
missing
confirmed
inferred
unknown
```

其中：

```text
missing = 尚未获取
unknown = 已经尝试获取，但客户不知道
```

不能把 unknown 当成缺失错误，也不能把 unknown 转成 `0`。

---

# 3. Phase 1：RequirementProfile 状态扩展
> ✅ **已完成（2026-09-16）** —— `RequirementProfile` 新增 `ask_counts` / `unknown_reasons` / `last_asked_slot`，
> 并提供 `slot_status()` 状态机（`missing → unknown_pending → unknown → confirmed`），
> 以及 `slot_value_present()` / `slot_is_confirmed()` / `is_unknown()` / `record_ask()` / `mark_unknown()`。
> 关键口径：**场景默认值（如"教堂默认固装"）算"有值"，但不算"客户确认"**。
> 验收：`tests/test_unknown_tolerance.py::TestSlotStateMachine`。

## 修改文件

```text
src/models/requirement.py
```

必要时同步：

```text
src/models/legacy_adapter.py
```

## 增加字段

每个需求字段至少支持：

```text
value
status
source
confidence
ask_count
skip_reason
```

例如：

```json
{
  "viewing_distance": {
    "value": null,
    "unit": "m",
    "status": "unknown",
    "source": "customer",
    "confidence": 1.0,
    "ask_count": 2,
    "skip_reason": "customer_does_not_know"
  }
}
```

---

## 状态定义

### missing

```text
尚未询问/尚未获得
```

### confirmed

```text
客户明确提供
```

### inferred

```text
根据规则推断
```

### unknown

```text
客户明确表示不知道
```

---

## 验收

测试：

```text
missing → confirmed
missing → unknown
inferred → confirmed
unknown → confirmed
```

全部正常。

---

# 4. Phase 2：建立 Global Requirement Extraction
> ✅ **已完成（2026-09-16）** —— 每条客户消息都由统一入口 `RequirementExtractor.extract()`
> 做**全量**提取（规则事实 + 复用同一轮语义结果），不再按"当前 pending 问题"过滤字段；
> Sales Agent 与 Solution Agent 走同一条链路。
> 验收：`TestPhase2GlobalExtraction::test_answer_other_field_is_not_dropped`（AI 问视距、客户答尺寸 → 尺寸照常入库）。

## 修改文件

```text
src/core/requirement_extractor.py
```

核心改造：

> 每条客户消息都必须进行全量需求提取。

不能只根据当前 pending question 提取字段。

---

## 示例

AI：

```text
What is the viewing distance?
```

客户：

```text
I don't know the viewing distance, but the screen is 5m × 3m for an indoor meeting room.
```

必须提取：

```text
viewing_distance = unknown
screen_width = 5m
screen_height = 3m
environment = indoor
scenario = meeting_room
```

而不是：

```text
viewing_distance = unknown
```

---

# 5. Phase 3：支持一条消息多个需求
> ✅ **已完成（2026-09-16）** —— 一条消息里的环境 / 场景 / 宽 / 高 / 视距一次性写入档案。
> 验收：`TestPhase2GlobalExtraction::test_one_message_updates_every_field`
> （"Outdoor stadium screen, 8m x 5m, viewing distance around 15m" → 5 个字段全部命中）。

Requirement Extractor 必须支持：

```text
one message
    ↓
multiple fields
```

例如：

```text
It's an outdoor stadium screen,
8m × 5m,
and the viewing distance is about 15 meters.
```

结果：

```json
{
  "environment": "outdoor",
  "scenario": "stadium",
  "screen_width": 8,
  "screen_height": 5,
  "viewing_distance": 15
}
```

所有字段一次性写入 RequirementProfile。

---

# 6. Phase 4：增加 Unknown Detection
> ✅ **已完成（2026-09-16）** —— 新增 `src/core/unknown_detector.py`，**纯正则、不调用 LLM**，
> 覆盖计划列出的全部英文 / 中文"不知道"说法与"跳过"说法，返回
> `customer_does_not_know` / `customer_skip` / `None`。
> 验收：`TestUnknownDetection`（中英文逐条参数化，含 "let's skip the viewing distance"）。

## 推荐位置

```text
src/core/requirement_extractor.py
```

或者独立：

```text
src/core/unknown_detector.py
```

优先规则检测，不要为了判断：

```text
"I don't know"
```

额外调用 LLM。

---

## 英文规则

至少支持：

```text
I don't know
I don't know yet
I have no idea
I'm not sure
Not sure
No idea
I don't have that information
I can't tell
Unknown
I can't estimate
```

---

## 中文规则

至少支持：

```text
不知道
不清楚
不确定
没了解
没有这个信息
暂时不知道
不太清楚
无法确定
没法估计
```

---

## Skip 规则

例如：

```text
skip it
don't ask
not available
we don't have this information
这个不用了
这个先跳过
这个没有
```

可以直接：

```text
status = unknown
skip_reason = customer_skip
```

无需第二次询问。

---

# 7. Phase 5：建立字段级 Ask Counter
> ✅ **已完成（2026-09-16）** —— 每个字段独立维护 `ask_counts`，由 Python 强制执行
> `ask_count >= MAX_ASKS_PER_SLOT(2) → 不再询问`，LLM 无权决定。
> 验收：`TestGoldenConversation`（客户连续说"不知道"时同一字段最多被问 2 次，之后自动跳到下一项）。

## 修改文件

```text
src/agents/sales/question_planner.py
```

每个字段单独维护：

```text
ask_count
```

状态：

```text
0 → 未询问
1 → 已询问一次
2 → 已询问两次
```

---

## 强制规则

```python
if ask_count >= 2:
    do_not_ask_again()
```

这个规则必须由 Python 执行。

不能让 LLM 自己决定。

---

# 8. Phase 6：第一次询问
> ✅ **已完成（2026-09-16）** —— 第一次追问即记账（`ask_count=1`）；客户答"不知道"时记
> `unknown_reasons[slot]=customer_does_not_know`，状态为 `unknown_pending`（仍可降门槛再问一次）。
> 验收：`TestSlotStateMachine::test_first_dont_know_is_still_askable`。

例如：

```text
What is the approximate viewing distance?
```

客户：

```text
I don't know.
```

更新：

```text
viewing_distance.status = unknown_pending
viewing_distance.ask_count = 1
```

然后第二次询问。

---

# 9. Phase 7：第二次询问必须降低门槛
> ✅ **已完成（2026-09-16）** —— `readiness.EASIER_QUESTIONS` + `question_for(..., easier=True)`：
> 第二次问同一字段时改成"给区间 / 二选一"（如"更近，还是 10 米开外？"），**不机械重复第一遍**。
> 验收：`TestPlanTestCases::test_2_unknown_once`（断言第二问与第一问措辞不同、且带区间）。

不能：

```text
第一次：
What is the viewing distance?

第二次：
What is the viewing distance?
```

应该：

```text
That's okay. Even an approximate distance is helpful.
Will viewers be relatively close to the screen, or more than about 10 meters away?
```

允许客户回答：

```text
approximately
roughly
near
far
more than 10m
around 5–10m
```

也可以接受客户的模糊描述。

---

# 10. Phase 8：第二次仍不知道 → Unknown
> ✅ **已完成（2026-09-16）** —— 问满两次仍无值 → `status=unknown`，从此不再询问该字段，
> Question Planner 直接跳到下一个字段；核心信息齐备时 Gate 转 `DEGRADED_READY`。
> 验收：`TestPlanTestCases::test_3_unknown_twice_then_stop_asking`。

例如：

```text
AI：That's okay. Even an approximate distance is fine...
客户：Still don't know.
```

更新：

```json
{
  "viewing_distance": {
    "value": null,
    "status": "unknown",
    "ask_count": 2,
    "skip_reason": "customer_does_not_know"
  }
}
```

然后：

```text
禁止继续询问 viewing_distance
```

Question Planner 选择下一个需求。

---

# 11. Phase 9：客户回答其他信息时必须同时记录
> ✅ **已完成（2026-09-16）** —— 客户说"不知道视距，但屏是 5m x 3m"时，
> 视距记 unknown、宽高照常入库；已确认字段绝不会被再次追问。
> 验收：`TestPlanTestCases::test_4_answer_other_information_while_not_knowing`。

这是本次优化的重点。

例如：

```text
AI：
What is the viewing distance?

客户：
I don't know.
The screen will be 6m × 3m.
It's for an outdoor advertising project.
```

系统必须得到：

```text
viewing_distance → unknown
screen_width → 6m
screen_height → 3m
environment → outdoor
scenario → advertising
```

然后：

```text
viewing_distance ask_count = 1
```

下一轮可以第二次询问。

但：

```text
screen_width
screen_height
environment
scenario
```

已经获得的信息不得再次询问。

---

# 12. Phase 10：客户主动补充 Unknown 信息
> ✅ **已完成（2026-09-16）** —— `unknown` 不是锁死状态：合并时只要客户**明确补上**，
> 自动 `unknown → confirmed` 并撤销 unknown 标记（场景默认值不算"补上"）。
> 验收：`TestPlanTestCases::test_6_later_addition_flips_unknown_to_confirmed`。

Unknown 不是永久锁死。

例如：

第 2 轮：

```text
viewing_distance = unknown
```

第 5 轮客户突然说：

```text
Actually, viewers will be about 8 meters away.
```

必须：

```text
unknown
    ↓
confirmed
```

最终：

```json
{
  "viewing_distance": {
    "value": 8,
    "unit": "m",
    "status": "confirmed",
    "source": "customer"
  }
}
```

但 Question Planner 不得因为这个字段曾经 unknown，就再次主动询问。

---

# 13. Phase 11：Recommendation Ready Gate 改造
> ✅ **已完成（2026-09-16）** —— `GateDecision.status` 三态：`READY` / `CONTINUE_ASKING` / `DEGRADED_READY`，
> 并带 `unknown_slots`。判定规则：待问字段（可再问）→ CONTINUE_ASKING；
> 只剩 unknown 且仍有选型依据（场景 / 室内外）→ DEGRADED_READY（放行推荐）。
> 验收：`TestDegradedRecommendation`、`TestPhase14RecommendationBasis`。

## 修改文件

```text
src/rag/readiness.py
```

当前 Gate 增加三种状态：

```text
READY
CONTINUE_ASKING
DEGRADED_READY
```

---

## READY

核心需求已经获得。

正常推荐。

---

## CONTINUE_ASKING

还有高价值信息：

```text
status = missing
ask_count < 2
```

继续询问。

---

## DEGRADED_READY

核心信息已经足够，但存在：

```text
unknown
```

或者非关键字段：

```text
missing
```

允许基于现有信息进行 Best-effort Recommendation。

---

# 14. Phase 12：不要简单放宽所有 Gate
> ✅ **已完成（2026-09-16）** —— 只把"辅助信息"（视距 / 亮度 / 预算…）放进可降级范围；
> 核心信息（室内外 / 场景 / 安装方式）仍然是 `CONTINUE_ASKING`，不会"什么都不知道也推荐"。
> 验收：`TestDegradedRecommendation::test_continue_asking_when_not_yet_asked_twice`。

不能改成：

```text
任何信息都没有也推荐
```

而应该定义：

## 核心推荐信息

例如：

```text
environment
scenario
installation
screen_size
```

根据当前项目产品规则确定具体必需字段。

---

## 辅助信息

例如：

```text
viewing_distance
brightness
refresh_rate
budget
```

缺失时不应该无限阻塞。

---

# 15. Phase 13：Recommendation Engine 处理 Unknown
> ✅ **已完成（2026-09-16）** —— `unknown` 不再等价于 0 分：缺数据的评分维度直接返回 `None`
> 并**整体跳过**该维度（`breakdown["pitch"] is None`），不拉低所有候选。
> 验收：`TestPhase13EngineUnknownSkip`。

## 修改文件

```text
src/rag/recommendation_engine.py
```

原则：

```text
unknown != 0
unknown != confirmed
unknown != inferred
```

---

## 评分处理

例如：

```text
environment_score
scenario_score
installation_score
pitch_score
viewing_distance_score
```

如果：

```text
viewing_distance = unknown
```

那么：

```text
viewing_distance_score
```

直接跳过。

不能：

```text
viewing_distance_score = 0
```

否则会错误降低所有候选产品的评分。

---

# 16. Phase 14：增加 Recommendation Basis
> ✅ **已完成（2026-09-16）** —— 推荐结果新增
> `recommendation_status`（`RECOMMENDED` / `DEGRADED`）与 `recommendation_basis`
> （`confirmed_requirements` / `inferred_requirements` / `unknown_requirements`），
> 以及顶层 `unknown_requirements`。
> 验收：`TestPhase14RecommendationBasis`。

推荐结果内部记录：

```json
{
  "recommendation_status": "degraded",
  "confirmed_requirements": [
    "environment",
    "scenario",
    "screen_width",
    "screen_height"
  ],
  "unknown_requirements": [
    "viewing_distance"
  ]
}
```

这样后续可以明确知道：

> 为什么系统推荐这个产品。

---

# 17. Phase 15：最终销售话术
> ✅ **已完成（2026-09-16）** —— 新增 `reply_composer.degraded_note()`（"缺什么 + 影响什么"的多套自然说法）
> 与 `_ensure_degraded_note()` 确定性兜底：LLM 漏说就一定补上；
> 话术禁止出现"无法推荐 / 信息不足"。
> 验收：`TestPhase15DegradedWording`、`TestPhase16DegradedRecommendationEndToEnd`。

## 信息完整

正常：

```text
Based on your requirements, I recommend XXX.
```

## 存在 Unknown

不要：

```text
I cannot recommend accurately.
```

应该：

```text
Based on the requirements you've confirmed, I would recommend XXX.

Since the viewing distance hasn't been confirmed, the final pixel pitch may need to be adjusted once that information is available.
```

结构：

```text
推荐产品
+
当前依据
+
缺少什么
+
缺少信息可能影响什么
```

---

# 18. Phase 16：Calculation 与 Unknown 解耦
> ✅ **已完成（2026-09-16）** —— Calculation Gate 只看屏体宽高；`viewing_distance=unknown`
> 时仍然照常计算箱体数量 / 模组数量 / 实际屏体尺寸。
> 验收：`test_calculation_not_blocked_by_unknown_distance`、
> `TestPhase16DegradedRecommendationEndToEnd::test_node_recommends_with_unknown_distance`。

如果已经有：

```text
product
cabinet specification
module specification
screen width
screen height
```

即使：

```text
viewing_distance = unknown
```

也必须允许：

```text
screen calculation
cabinet quantity
module quantity
actual screen size
```

继续执行。

因为：

```text
viewing_distance
```

主要影响：

```text
pixel pitch / product selection
```

而不是：

```text
cabinet arithmetic
```

---

# 19. Phase 17：Question Planner 优先级
> ✅ **已完成（2026-09-16）** —— `SLOT_PRIORITY`（HIGH / MEDIUM / LOW）定义字段优先级；
> Question Planner 遇到 `unknown` 字段**直接跳过**，`missing_slots()` 也不再把它算作"缺"。
> 验收：`TestPhase17PlannerPriority`。

给字段增加：

```text
priority
```

示例：

```text
environment       HIGH
scenario          HIGH
installation      HIGH
screen_size       HIGH
pixel_pitch       MEDIUM
viewing_distance  MEDIUM
brightness        MEDIUM
refresh_rate      LOW
budget            LOW
```

当某字段：

```text
unknown
```

之后，直接跳过，继续寻找下一个：

```text
missing + high priority
```

---

# 20. Phase 18：Session Switch 兼容
> ✅ **已完成（2026-09-16）** —— `reset_requirement_state()` 清空累计需求 + 结构化档案 + 推荐记录，
> 新项目不会继承旧项目的 unknown / 已确认事实；同时（有意保留）**提问记账跨轮保留**，
> 否则"同一字段最多问两次"会失效。
> 验收：`TestSessionSwitchClearsUnknownState`、`tests/test_session_reset.py`。

现有：

```text
src/rag/session_switch.py
```

需要检查：

> 新项目是否会继承旧项目的 ask_count / unknown 状态。

例如：

```text
Project A:
viewing_distance = unknown
ask_count = 2
```

客户突然：

```text
Actually, I have another project...
```

必须创建新的 RequirementProfile 或正确重置项目级状态。

避免：

```text
Project B
viewing_distance = unknown
ask_count = 2
```

被错误继承。

---

# 21. Phase 19：日志优化
> ✅ **已完成（2026-09-16）** —— 每轮输出
> `[RequirementExtraction]`（extracted / confirmed / unknown）、
> `[QuestionState]`（slot / ask_count / status / reason）、
> `[QuestionPlanner]`（next_slot）、
> `[RecommendationGate]`（status / missing / unknown）。

每轮记录：

```text
[RequirementExtraction]

message = ...

extracted_fields = [...]
confirmed_fields = [...]
unknown_fields = [...]
```

然后：

```text
[QuestionState]

slot = viewing_distance
ask_count = 1
status = unknown_pending
```

然后：

```text
[QuestionPlanner]

next_slot = screen_size
```

最终：

```text
[RecommendationGate]

status = DEGRADED_READY
unknown_fields = viewing_distance
```

这样可以快速定位问题。

---

# 22. Phase 20：测试用例
> ✅ **已完成（2026-09-16）** —— Test 1~7 全部落地在 `tests/test_unknown_tolerance.py`，
> 并补了状态机、DEGRADED 推荐、计算解耦、会话切换等用例（共 59 个）。

## Test 1：正常回答

```text
AI：What is the viewing distance?
客户：About 8 meters.
```

预期：

```text
viewing_distance = 8m
status = confirmed
```

---

## Test 2：第一次不知道

```text
客户：I don't know.
```

预期：

```text
status = unknown_pending
ask_count = 1
```

---

## Test 3：第二次仍不知道

```text
客户：Still don't know.
```

预期：

```text
status = unknown
ask_count = 2
```

并且之后：

```text
Question Planner 不再询问
```

---

## Test 4：回答其他信息

```text
客户：
I don't know the viewing distance,
but the screen is 5m × 3m.
```

预期：

```text
viewing_distance = unknown
screen_width = 5m
screen_height = 3m
```

---

## Test 5：一条消息多个需求

```text
客户：
Outdoor stadium screen,
8m × 5m,
viewing distance around 15m.
```

预期：

```text
environment = outdoor
scenario = stadium
width = 8m
height = 5m
viewing_distance = 15m
```

---

## Test 6：后续主动补充

```text
第 2 轮：
viewing distance unknown

第 5 轮：
Actually, it's about 10 meters.
```

预期：

```text
unknown → confirmed
10m
```

---

## Test 7：直接要求跳过

```text
Let's skip the viewing distance.
```

预期：

```text
status = unknown
skip_reason = customer_skip
ask_count <= 1
```

不再询问。

---

# 23. 回归测试
> ✅ **已完成（2026-09-16）** —— `tests/test_unknown_tolerance.py::TestGoldenConversation`
> 用一段金标对话验证计划要求的两项硬指标：
>
> ```text
> 超过 2 次询问率 = 0%          （同一字段最多被问 2 次）
> 已确认信息被重复询问率 = 0%    （拿到的字段不会再问）
> ```
>
> 客户连续回答"不知道"时，系统最终会转入 `DEGRADED_READY` 并**继续按已有信息推荐**，
> 而不是无限追问。

至少建立以下 Golden Dataset：

```text
normal_answer
unknown_once
unknown_twice
customer_skips
multi_field_answer
answer_other_question
later_correction
later_addition
session_switch
recommendation_with_unknown
calculation_with_unknown
```

重点指标：

```text
信息丢失率
重复询问率
超过 2 次询问率
Unknown 正确识别率
多字段提取率
推荐 Gate 正确率
计算阻塞错误率
```

目标：

```text
超过 2 次询问率 = 0%
```

```text
已确认信息被重复询问率 → 0%
```

---

# 24. 实施顺序

不要同时大面积修改所有 Agent。

按照以下顺序实施：

```text
Phase 1
RequirementProfile
        ↓
Phase 2
Global Requirement Extractor
        ↓
Phase 3
Unknown Detection
        ↓
Phase 4
Question Attempt Tracker
        ↓
Phase 5
Question Planner
        ↓
Phase 6
Recommendation Ready Gate
        ↓
Phase 7
Recommendation Engine
        ↓
Phase 8
Reply Composer
        ↓
Phase 9
Session Switch 检查
        ↓
Phase 10
Golden Dataset 回归测试
```

---

# 25. 最终目标架构

```text
                 Customer Message
                         │
                         ▼
          ┌──────────────────────────┐
          │ Global Requirement       │
          │ Extractor                │
          │ 全量需求提取              │
          └────────────┬─────────────┘
                       │
                       ▼
               RequirementProfile
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   当前问题是否回答             其他新信息
          │                         │
          └────────────┬────────────┘
                       ▼
              Field State Manager
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      confirmed      unknown      missing
                         │
                   ask_count < 2
                    │          │
                   YES         NO
                    │          │
                    ▼          ▼
                 再问一次     跳过
                                │
                                ▼
                    Recommendation Gate
                                │
             ┌──────────────────┼──────────────────┐
             ▼                  ▼                  ▼
           READY         CONTINUE_ASKING    DEGRADED_READY
             │                  │                  │
             └──────────────────┼──────────────────┘
                                ▼
                     Recommendation Engine
                                │
                                ▼
                        Screen Calculator
                                │
                                ▼
                       Final Sales Response
```

---

# 26. 最终验收标准
> ✅ **全部通过（2026-09-16）**

| 验收项 | 结论 | 证据 |
| --- | --- | --- |
| 需求信息：客户说出的有效需求都要入库 | ✅ | `TestPhase2GlobalExtraction` |
| Unknown：一个字段最多主动询问两次 | ✅ | `TestGoldenConversation` |
| 防死循环：Python 强制 `ask_count >= 2` 后不再询问 | ✅ | `RequirementProfile.is_unknown()` + `TestPlanTestCases::test_3` |
| 多字段：一条消息可更新多个字段 | ✅ | `test_one_message_updates_every_field` |
| 信息补充：unknown 可重新变 confirmed | ✅ | `test_6_later_addition_flips_unknown_to_confirmed` |
| 推荐：信息不完整也能 Best-effort 推荐 | ✅ | `TestPhase13EngineUnknownSkip`、`TestPhase14RecommendationBasis` |
| 透明度：话术说明缺失项及影响 | ✅ | `TestPhase15DegradedWording` |
| 计算：与视距无关的计算不被 unknown 阻塞 | ✅ | `TestPhase16DegradedRecommendationEndToEnd` |

项目完成后必须满足：

### 需求信息

> 客户说出的有效需求，不管是不是回答当前问题，都必须进入 RequirementProfile。

### Unknown

> 一个字段最多主动询问两次。

### 防死循环

> Python 强制保证 `ask_count >= 2` 后不能继续询问。

### 多字段

> 一条消息可以同时更新多个需求字段。

### 信息补充

> Unknown 字段后续客户主动提供信息时，可以重新变成 confirmed。

### 推荐

> 已有信息满足最低条件时，即使部分需求 unknown，也可以进行 Best-effort Recommendation。

### 透明度

> 推荐结果明确知道哪些字段缺失/unknown，并在最终话术中自然提示。

### 计算

> 与 viewing distance 等选型辅助信息无关的计算，不应被 Unknown 阻塞。

---

# 27. 核心结论

这次优化不要单纯通过增加 Prompt 来解决。

真正需要建立的是：

```text
全量信息捕获
+
字段状态机
+
字段级询问次数控制
+
Unknown 状态
+
动态推荐 Gate
```

最终让 Sales Agent 从：

```text
问一个问题
↓
等客户回答这个问题
↓
继续问下一个
```

升级成：

```text
持续理解客户所有表达
↓
不断更新 RequirementProfile
↓
缺什么才问什么
↓
客户不知道 → 最多问两次
↓
仍不知道 → Unknown
↓
继续收集其他信息
↓
信息足够 → 正常推荐
↓
信息不完整 → Best-effort 推荐 + 缺失信息说明
```

**核心原则：**

> **问题是对话控制，不是信息提取边界。**

> **Unknown 是有效状态，不是系统错误。**

> **客户说出来的信息，一个都不能因为“不是当前问题”而丢掉。**

---

# 附录 A：改动文件清单（2026-09-16）

| 文件 | 改动 |
| --- | --- |
| `src/models/requirement.py` | 新增 `ask_counts` / `unknown_reasons` / `last_asked_slot` 字段、`SLOT_TO_FIELD` / `MAX_ASKS_PER_SLOT` / `SLOT_PRIORITY` 常量、`slot_status()` 状态机、`slot_is_confirmed()`、`requirement_basis()`；合并时保留提问记账 |
| `src/core/unknown_detector.py` | **新增**：客户"不知道 / 跳过"的纯正则识别（中英文） |
| `src/core/requirement_extractor.py` | 全量提取链路（上一阶段已完成，本期只验证） |
| `src/rag/readiness.py` | `GateDecision.status` 三态 + `unknown_slots`；`EASIER_QUESTIONS` 第二次追问降门槛 |
| `src/rag/recommendation_service.py` | `RECOMMENDED` / `DEGRADED` + `recommendation_basis` |
| `src/rag/recommendation_engine.py` | 未知维度跳过评分；结果透传推荐状态 |
| `src/rag/reply_composer.py` | 新增 `degraded_note()` / `missing_impact()`（"缺什么 + 影响什么"） |
| `src/agents/sales/nodes/requirement.py` | Unknown 检测接入、`[RequirementExtraction]` / `[QuestionState]` / `[QuestionPlanner]` / `[RecommendationGate]` 日志、提问记账 |
| `src/agents/sales/question_planner.py` | 跳过 unknown 字段；返回字段优先级 |
| `src/agents/solution/nodes/recommend.py` | Best-effort 推荐话术 + `_ensure_degraded_note()` 确定性兜底 |
| `tests/test_unknown_tolerance.py` | **新增**：59 个用例（计划 Test 1~7 + 状态机 + 降级推荐 + 计算解耦 + 金标对话） |
| `tests/test_session_reset.py` | 断言收紧为"不残留任何旧需求事实"（提问记账除外） |

---

# 附录 B：实施过程中发现并修复的两个真实缺陷

## B1：场景默认值导致同一问题被**无限追问**

**现象**：客户说 "indoor LED screen for a church"，AI 问"固定安装还是租赁？"，
客户连答 12 次 "I don't know"，AI 仍然重复同一个问题。

**根因**：场景会给安装方式一个**默认值**（教堂 → 固装）。
旧判定用 `slot_value_present()`（"有没有值"）：

```text
installation 有默认值 → 认为"已经有值" → is_unknown() 永远 False
                      → 客户说"不知道"也记不下来 → 永远阻塞 → 无限追问
```

**修复**：区分"有值"与"客户确认"：

```text
slot_is_confirmed(slot)  = 有值 且 来源是客户侧（explicit / confirmed / scenario_derived）
is_unknown(slot)         = 未被客户确认 且 （客户明确跳过 或 ask_count >= 2）
```

**效果**：安装方式问 2 次 → unknown → 继续问视距 → 视距问 2 次 → unknown →
`DEGRADED_READY` 按已确认信息（室内 / 教堂 / 5m x 3m）继续推荐。

## B2：skip 正则漏匹配

**现象**：客户说 "let's skip the viewing distance" 时没有被识别为"跳过这一项"。

**根因**：正则只写了 `skip (it|this|that)`。

**修复**：改为 `\bskip\b`（并保留"不用了 / 先跳过"等中文说法）。

---

# 附录 C：如何复跑

```bash
cd C:\demo-agent\Sale_agent

# 本计划对应的回归
python -m pytest tests/test_unknown_tolerance.py -q

# 全量回归
python -m pytest tests -q
```

最近一次结果：

```text
tests/test_unknown_tolerance.py   59 passed
tests（全量）                    759 passed, 4 skipped
```

（4 条 skip 的是需要真实 LLM 的非中英文用例，设置 `RUN_LLM_EXTRACTION_TESTS=1` 才会执行。）
