# LED RAG 智能销售系统：关键词依赖优化与统一需求理解改造计划

> 版本：v1.0  
> 日期：2026-09-15  
> 适用仓库：`WSLVOUD/sale_agent_demo`  
> 目标：降低系统对“关键词命中”的依赖，让客户使用任意自然语言表达需求时，都能稳定转换为统一的 `RequirementProfile`，同时保留确定性规则、数字解析和推荐 Gate 的可靠性。

---

# 执行进度（2026-09-15 更新）

| 阶段 | 内容 | 状态 |
|---|---|---|
| Phase 1 | 唯一 RequirementProfile / canonical fields / confirmed-inferred / confidence / conflict | ✅ 已完成 |
| Phase 2 | RequirementExtractor（LLM 语义 + 规则 + 数字解析 + **短语快速路径**） | ✅ 已完成 |
| Phase 3 | Sales requirement 接入统一 Extractor，删除重复 environment 关键词 | ✅ 已完成 |
| Phase 4 | Solution requirement 复用同一轮语义结果（不再重复调 LLM） | ✅ 已完成 |
| Phase 5 | `query_understanding` 降级为确定性解析层（purpose 短语表成为 fast path） | ✅ 已完成 |
| Phase 6 | Router 改用结构化需求判断（关键词表降为 fallback） | ✅ 已完成 |
| Phase 7 | Script Generator 只读 RequirementProfile，不再自己判断场景 | ✅ 已完成 |
| Phase 8 | Gate 改结构化字段判断 + 冲突阻断 | ✅ 已完成 |
| Phase 9 | 113 条 Golden Cases + 6 类测试（关键词缺失 / 场景歧义 / 参数解析 / 多语言 / 推荐回归） | ✅ 已完成 |
| Phase 10 | 删除重复关键词/场景分类 | ✅ 已完成（保留快速路径与兜底，见下方说明） |

**验证**：全量测试 **660 passed / 4 skipped**（跳过的 4 条是"非中英文场景"用例，
需要 LLM 语义层：`RUN_LLM_EXTRACTION_TESTS=1` 时才跑）。

**实施过程中顺带修掉的真实 bug**：

1. **语言识别把英文判成德语**：`_LANGUAGE_HINTS` 的德语提示里混进了英文单词
   `display`，导致任何含 "display" 的英文句子被判成 `de`（影响回复语言策略与
   多语言关键词表选择）。现在提示词只保留该语言特有词，并补了 ru。
2. **英文词形变化不识别**：`permanent` 匹配不到 `permanently`、`bank` 匹配不到
   `banking`、`wall mount` 匹配不到 `wall mounted`。英文关键词现在支持
   `s/es/d/ed/ing/ly` 词尾。
3. **中文视距"6 米远"解析不到**：裸数值兜底原本要求单位后是空白/标点，
   现在也接受中文（"6 米远" → 6.0m）。
4. **`open-air` 不在室外关键词里**：现在补上（"open-air advertising" → outdoor）。
5. **零售场景环境缺失**：`EnvironmentResolver` 原本把 `retail` 当"室内外都可能"，
   与 `query_understanding` 的"商场=室内"不一致 —— 已统一为 indoor
   （商场外立面属于 advertising，不受影响）。
6. **冲突检测过严**：原实现把"客户明确说 rental + 会议室场景"当成冲突并阻断推荐，
   但租一块屏开一天会是正常业务。现在只有"同一轮出现两个互相矛盾的明确信号"
   才算冲突，客户明确事实永远优先于场景默认值。

**关于"删除重复关键词"（Phase 10）的口径**：
按计划 6.1/6.2，关键词表**不是删除而是降级**：`query_understanding` 的关键词/正则
只负责确定性事实（室内外、固装租赁、LED/LCD/IFP、COB/HDR/防水、P2.5、5m、5000nit…）
与数字/单位解析；`Router` 的 `_SCENE_KEYWORDS` 变成 fallback（有结构化需求时不再看它）；
场景理解改由 `RequirementExtractor`（短语快速路径 + LLM 语义）承担。

---

## 1. 本次改造的核心结论

当前项目**不是纯关键词系统**。

现有代码已经具备：

- Solution 侧 LLM Requirement Extraction
- Sales 侧 requirement mining
- `query_understanding.py` 中的规则/关键词解析
- `parameter_inference.py` 技术参数推断
- Recommendation Ready Gate
- 确定性 Product Recommendation
- 数字、尺寸、视距、单位等正则解析

真正的问题是：

> **同一个“客户需求理解”被多个模块分别实现，导致关键词表重复、规则重复、语义标准不统一。**

目前至少存在以下几类重复理解逻辑：

```text
Sales requirement.py
    └── SCENE_CATEGORIES
    └── indoor/outdoor keywords
    └── display_type keywords

Solution requirement.py
    └── LLM requirement extraction
    └── search keyword extraction

rag/query_understanding.py
    └── INDOOR_KEYWORDS
    └── OUTDOOR_KEYWORDS
    └── PURPOSE_KEYWORDS
    └── DISPLAY_TYPE_KEYWORDS
    └── RENTAL/FIXED/COB/HDR/GOB/FLEXIBLE/BUDGET keywords

rag/router.py
    └── SCENE_KEYWORDS

sales/script_generator.py
    └── indoor/outdoor keywords

parameter_inference.py
    └── 再次引用 query_understanding 的关键词
```

因此本次优化不是“继续补关键词”，而是：

> **建立统一 Requirement Extractor，LLM 负责自然语言语义理解，程序规则负责确定性事实和数值解析，所有后续模块只消费统一的结构化需求。**

---

# 2. 当前代码问题定位

## 2.1 `src/rag/query_understanding.py`

这是本次改造的重点。

当前存在：

```python
_INDOOR_KEYWORDS
_OUTDOOR_KEYWORDS
_SEMI_OUTDOOR_KEYWORDS
_RENTAL_KEYWORDS
_FIXED_KEYWORDS
_DISPLAY_TYPE_KEYWORDS
_PURPOSE_KEYWORDS
_WATERPROOF_KEYWORDS
_COB_KEYWORDS
_HDR_KEYWORDS
_GOB_KEYWORDS
_FLEXIBLE_KEYWORDS
_BUDGET_LOW_KEYWORDS
_BUDGET_HIGH_KEYWORDS
_BUDGET_MID_KEYWORDS
```

同时 `_detect_purpose()` 通过遍历 `_PURPOSE_KEYWORDS` 来得到标准 purpose。

这意味着：

```text
客户表达
    ↓
必须命中词表
    ↓
purpose
```

例如：

```text
shopping mall → retail
shopping center → retail
commercial complex → retail
commercial plaza → retail
```

如果词表没有覆盖其中某一种表达，就可能无法得到标准 `retail`。

### 改造原则

不要删除关键词表。

而是把关键词从：

> “主要语义理解手段”

降低为：

> “确定性规则 + 快速路径 + 兜底机制”。

---

# 3. 新的总体架构

改造后：

```text
Customer Message
       │
       ▼
┌────────────────────────────┐
│ Unified Requirement        │
│ Extractor                  │
│                            │
│ ① LLM Semantic Extraction  │
│ ② Rule Extraction          │
│ ③ Numeric/Unit Parser      │
└──────────────┬─────────────┘
               │
               ▼
      RequirementProfile
               │
               ▼
┌────────────────────────────┐
│ Requirement Normalizer     │
│                            │
│ purpose → canonical token  │
│ environment → canonical    │
│ installation → canonical   │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ Requirement Validator      │
│                            │
│ confirmed / inferred       │
│ confidence                │
│ conflict detection         │
└──────────────┬─────────────┘
               │
        ┌──────┴───────┐
        │              │
      缺信息          信息足够
        │              │
        ▼              ▼
      Clarify      Recommendation Gate
                       │
                       ▼
               Product Filter
                       │
                       ▼
                    Scoring
                       │
                       ▼
                 Recommendation
```

核心原则：

```text
LLM = 理解语言
Regex/Parser = 解析数字与明确事实
Rule Engine = 验证和约束
RAG = 提供产品事实
Recommendation Engine = 最终选型
```

禁止：

```text
LLM → 直接决定产品型号
关键词 → 直接触发推荐
场景词 → 自动猜测客户未明确提供的工程参数
```

---

# 4. 第一阶段：建立统一 Requirement Schema

## 4.1 目标

先统一整个项目使用的需求字段，避免 Sales、Solution、RAG 各自维护不同字段。

建议最终标准结构：

```python
RequirementProfile = {
    "display_type": None,

    "environment": None,
    "installation": None,

    "purpose": None,

    "size": {
        "width_mm": None,
        "height_mm": None,
        "area_m2": None
    },

    "viewing_distance_m": None,

    "brightness": None,
    "resolution": None,
    "pixel_pitch": None,

    "rental": None,
    "flexible": None,
    "waterproof": None,
    "cob": None,
    "gob": None,
    "hdr": None,

    "budget_level": None,

    "confirmed_fields": [],
    "inferred_fields": [],

    "confidence": {},
    "conflicts": []
}
```

实际字段应以仓库已有 `RequirementProfile` 为准，不要重新制造第二套 Profile。

### 要求

先检查：

```text
src/models/
src/agents/sales/state.py
src/agents/solution/state.py
src/rag/
```

最终只保留一个 canonical requirement schema。

---

# 5. 第二阶段：建立 `RequirementExtractor`

建议新增：

```text
src/core/requirement_extractor.py
```

或者如果当前项目已经有统一模型目录，则放在：

```text
src/models/requirement_extractor.py
```

推荐优先选择：

```text
src/core/requirement_extractor.py
```

职责：

```text
客户消息
    ↓
extract()
    ↓
结构化 RequirementProfile
```

---

## 5.1 LLM Semantic Extraction

LLM 专门处理自然语言：

例如：

```text
We need a screen for a shopping center.
```

输出：

```json
{
  "purpose": "retail",
  "environment": null,
  "installation": null,
  "display_type": "LED"
}
```

注意：

**不能因为 shopping center 就擅自把 environment 设置为 indoor，除非业务规则明确允许。**

---

## 5.2 Purpose 标准化

建立标准 purpose enum：

```python
PURPOSES = {
    "retail",
    "advertising",
    "conference",
    "classroom",
    "stadium",
    "concert",
    "stage",
    "wedding",
    "church",
    "museum",
    "showroom",
    "airport",
    "bank",
    "hotel",
    "restaurant",
    "office",
    "hospital",
    "exhibition",
    "hall",
    "rental",
    "other"
}
```

LLM负责把自然语言映射到这些 canonical token。

例如：

```text
shopping mall
shopping center
commercial complex
department store
retail complex
```

统一：

```text
retail
```

再例如：

```text
sports complex
football venue
basketball arena
sports venue
```

统一：

```text
stadium
```

### 重要

不要把所有自然语言表达全部写进 Python。

Python 只保存：

```text
canonical categories
```

而不是：

```text
所有可能的客户表达
```

---

# 6. 第三阶段：保留确定性关键词，但缩小职责

现有关键词表不要一次删除。

## 6.1 必须保留

以下类型继续使用规则：

```text
indoor
outdoor
semi-outdoor

rental
fixed

LED
LCD
IFP

COB
GOB
HDR

waterproof
IP65
IP66

flexible
curved

P2.5
P3
P4
...

5m
10 meters
30ft

5000nit
6000 nits
```

原因：

这些属于高确定性字段。

---

## 6.2 Purpose 关键词降级

当前：

```python
_PURPOSE_KEYWORDS
```

不要再无限增加。

保留少量高频、高确定性表达作为：

```text
fast path
fallback
validation
```

而不是主要语义理解入口。

---

# 7. 第四阶段：统一 Environment 判断

这是目前最容易出现逻辑冲突的地方。

当前不同模块都可能自己判断：

```text
indoor
outdoor
```

必须统一成：

```python
resolve_environment()
```

优先级：

```text
Priority 1:
客户明确说 indoor/outdoor
        ↓
Priority 2:
LLM 从语义中提取，且 confidence 足够高
        ↓
Priority 3:
明确业务规则允许的场景默认值
        ↓
Priority 4:
无法确定 → None
```

---

## 7.1 明确禁止的行为

不能：

```text
concert → indoor
stage → indoor
wedding → indoor
rental → outdoor
```

因为这些场景可能同时存在室内和室外。

应保持：

```text
concert → ambiguous
stage → ambiguous
wedding → ambiguous
rental → ambiguous
```

然后进入 Clarify。

---

# 8. 第五阶段：统一 Installation 判断

同样不要在 Sales、Solution、Script Generator 各自判断。

统一：

```python
resolve_installation()
```

canonical values：

```text
fixed
rental
unknown
```

规则：

```text
客户明确 rental
    → rental

客户明确 permanent/fixed installation
    → fixed

没有明确说明
    → 不强制猜测
```

如果业务上有“某些产品默认固装”的规则，应放在 Recommendation/Business Rule 层，而不是自然语言解析层。

---

# 9. 第六阶段：数字和工程参数继续使用程序解析

这一部分不要改成完全依赖 LLM。

当前项目已经有大量成熟的：

```text
distance parser
size parser
unit conversion
axis measurement
pixel pitch
```

继续保留。

例如：

```text
5.28m × 3.2m
```

必须解析为：

```json
{
  "width_mm": 5280,
  "height_mm": 3200
}
```

而不是让 LLM 返回：

```text
5.28m × 3.2m
```

然后再由其他模块猜。

---

## 9.1 视距

继续保留：

```text
5m
5 meters
100 feet
30ft
```

等程序解析。

统一输出：

```text
viewing_distance_m
```

并完成单位换算。

---

# 10. 第七阶段：建立 confirmed / inferred 双来源

这是本项目非常重要的一步。

每个需求字段都应该区分：

```text
confirmed
inferred
```

例如：

客户：

> We need a screen for a shopping mall.

得到：

```text
purpose = retail
source = confirmed
```

如果系统通过规则推断：

```text
environment = indoor
source = inferred
```

则：

> inferred 信息不能直接打开 Recommendation Ready Gate，除非业务规则明确规定该字段允许自动推断。

---

## 推荐数据结构

```json
{
  "purpose": {
    "value": "retail",
    "source": "confirmed",
    "confidence": 0.97
  },
  "environment": {
    "value": "indoor",
    "source": "inferred",
    "confidence": 0.72
  }
}
```

这样后续系统可以明确知道：

```text
客户说了什么
系统推断了什么
```

---

# 11. 第八阶段：解决多个模块重复判断

## 11.1 `src/agents/sales/nodes/requirement.py`

当前有：

```python
SCENE_CATEGORIES
```

以及：

```python
_rule_based_inference()
```

改造：

- 保留数字解析
- 保留明确事实解析
- 删除重复的 purpose 分类逻辑
- 删除独立的 indoor/outdoor keyword 表
- 改为调用统一 Requirement Extractor

目标：

```python
requirements = requirement_extractor.extract(...)
```

---

# 12. `src/agents/solution/nodes/requirement.py`

当前已经有 LLM requirement extraction。

这是最接近新架构的模块。

改造：

```text
现有 LLM extraction
        ↓
统一 RequirementExtractor
```

不要让 Solution 再维护自己的字段解释规则。

同时保留：

```text
requirements_skip_understand
```

这种性能优化机制。

---

# 13. `src/rag/query_understanding.py`

这是本次最核心的重构对象。

### 第一阶段不要删除

```text
_PURPOSE_KEYWORDS
_INDOOR_KEYWORDS
_OUTDOOR_KEYWORDS
...
```

先迁移调用关系。

新增：

```python
canonicalize_requirement()
```

然后逐步让：

```text
_detect_purpose()
environment_from_purpose()
```

成为兼容层，而不是主入口。

最终目标：

```text
query_understanding
    ↓
只负责 query normalization / deterministic parsing
```

而不是承担整个客户需求理解。

---

# 14. `src/rag/router.py`

当前 `_SCENE_KEYWORDS` 用来判断：

```text
纯参数查询
还是
场景/Agent 查询
```

这个逻辑不应该继续维护第二套场景词表。

改成：

```text
Requirement Extractor
        ↓
has_semantic_requirement
```

例如：

```python
requirement.has_meaningful_context()
```

Router 根据结构化结果判断。

---

# 15. `src/agents/sales/nodes/script_generator.py`

当前还有：

```python
indoor_keywords
outdoor_keywords
```

不能继续让话术生成器自己推断环境。

改成：

```text
Script Generator
       ↓
只读取 RequirementProfile
       ↓
生成销售话术
```

Script Generator 不负责：

```text
客户需求理解
环境判断
产品选择
技术参数推断
```

---

# 16. `parameter_inference.py`

继续保留。

但职责必须明确：

```text
Requirement
    ↓
Engineering Parameter Inference
```

而不是：

```text
客户自然语言
    ↓
parameter_inference 自己重新识别客户场景
```

也就是说：

```text
query_understanding
→ 理解客户

parameter_inference
→ 根据已经确定的需求做工程推断
```

---

# 17. 第九阶段：Recommendation Gate 重新定义

Recommendation Gate 不应该检查：

```text
keyword 是否命中
```

而应该检查：

```text
RequirementProfile 是否满足推荐所需条件
```

例如：

```python
required = [
    "display_type",
    "environment",
    "purpose",
    "installation"
]
```

以及根据项目业务要求判断：

```text
size
viewing_distance
```

是否已经足够。

最终：

```text
Gate = structure completeness
```

而不是：

```text
Gate = keyword completeness
```

---

# 18. 第十阶段：建立冲突检测

LLM + Rule 同时存在时必须处理冲突。

例如：

客户：

> “It is an outdoor screen for our shopping mall.”

LLM：

```text
purpose = retail
environment = outdoor
```

规则：

```text
retail
```

不能自动覆盖客户明确说的：

```text
outdoor
```

统一优先级：

```text
客户明确事实
    >
确定性规则解析
    >
LLM语义推断
    >
业务默认值
```

如果出现真正冲突：

```text
environment = indoor
environment = outdoor
```

则：

```text
conflicts = ["environment"]
```

Gate 不允许直接推荐。

---

# 19. 第十一阶段：建立“关键词依赖测试集”

这是本次改造必须完成的验收项目。

建立：

```text
tests/
└── requirement_extraction/
    ├── test_environment.py
    ├── test_purpose.py
    ├── test_installation.py
    ├── test_size.py
    ├── test_distance.py
    ├── test_multilingual.py
    └── golden_cases.json
```

---

# 20. Golden Dataset

至少建立 100 条真实/模拟客户表达。

## 20.1 Purpose

例如：

```text
I need a display for a shopping mall.
We are opening a new commercial center.
The screen is for a retail complex.
We need signage for our department store.
```

全部应该：

```text
purpose = retail
```

---

## 20.2 Stadium

```text
We need a display for a football venue.
It will be used in a sports complex.
We need a screen for a basketball arena.
```

统一：

```text
purpose = stadium
```

---

## 20.3 Conference

```text
corporate boardroom
meeting facility
conference center
executive meeting room
```

统一：

```text
purpose = conference
```

---

## 20.4 Outdoor

测试：

```text
outside the building
on the facade
on the exterior wall
open-air advertising
out in the open
```

---

## 20.5 Rental

测试：

```text
temporary event
touring show
short-term installation
portable event display
concert rental
```

---

# 21. 多语言测试

至少：

```text
中文
English
German
French
Spanish
Russian
Japanese
```

例如：

```text
Wir brauchen einen Bildschirm für ein Einkaufszentrum.
```

应正确得到：

```text
purpose = retail
```

而不是依赖：

```text
"Einkaufszentrum"
```

必须提前写进关键词表。

---

# 22. 第十二阶段：性能控制

不能因为引入 LLM 就让系统更慢。

推荐采用：

```text
明确规则命中
    ↓
先走 deterministic fast path

无法确定
    ↓
调用 LLM Semantic Extractor
```

例如：

```text
"Outdoor LED P3 rental screen"
```

直接解析：

```text
display_type = LED
environment = outdoor
pixel_pitch = P3
installation = rental
```

无需 LLM。

但：

```text
"We need something for our new commercial complex..."
```

无法通过规则完整理解：

```text
→ 调用 LLM
```

这样可以兼顾：

```text
准确率
+
响应速度
```

---

# 23. 第十三阶段：缓存语义结果

对于同一轮对话，不允许多个 Agent 重复调用 LLM 理解相同内容。

例如：

```text
Sales Agent
    ↓
LLM extraction

Solution Agent
    ↓
又一次 LLM extraction

Script Generator
    ↓
又自己推断
```

应该：

```text
Customer Message
       ↓
Requirement Extractor
       ↓
RequirementProfile
       ├── Sales
       ├── Solution
       ├── Gate
       ├── Recommendation
       └── Script Generator
```

同一轮尽量只生成一次结构化需求。

---

# 24. 第十四阶段：迁移顺序

不要一次性删除旧代码。

推荐：

## Phase 1 — 建模

- [x] 确认唯一 `RequirementProfile`
- [x] 明确 canonical fields
- [x] 明确 confirmed/inferred（实际是 explicit/confirmed/scenario_derived/default/inferred）
- [x] 明确 confidence（`RequirementProfile.confidence`）
- [x] 明确 conflict（`RequirementProfile.conflicts` + Gate 阻断）

## Phase 2 — 建 Extractor

- [x] 新建 `RequirementExtractor`
- [x] 接入 LLM semantic extraction（带"客户原话证据"校验，防幻觉）
- [x] 接入现有数字/尺寸/视距解析
- [x] 接入现有 keyword fast path + canonical phrase 快速路径

## Phase 3 — 接 Sales

- [x] Sales requirement 改调用统一 Extractor
- [x] 删除重复 environment keywords（改为由 Extractor 统一解析）
- [x] 保留已有数字解析（`extract_slots` 仍负责尺寸/视距/点间距）

## Phase 4 — 接 Solution

- [x] Solution requirement 改调用统一 Extractor
- [x] 保留 skip-understand 优化
- [x] 避免重复 LLM extraction（同一轮语义结果缓存复用）

## Phase 5 — 接 RAG

- [x] `query_understanding.py` 改为确定性解析层（不再承担整个需求理解）
- [x] `_detect_purpose()` 降级为 fallback（canonical purpose 由 PurposeNormalizer 统一）
- [x] 统一 canonical purpose（22 个 token）
- [x] 保留技术关键词 parser

## Phase 6 — 接 Router

- [x] 删除 `_SCENE_KEYWORDS` 的主判断职责（有结构化需求时优先）
- [x] 改用 RequirementProfile（`has_structured_requirement()`）

## Phase 7 — 接 Script Generator

- [x] 删除独立 environment 推断
- [x] 只读取 RequirementProfile

## Phase 8 — Gate

- [x] 改为检查结构化字段
- [x] 禁止 inferred 字段无条件打开 Gate
- [x] 增加 conflict blocking

## Phase 9 — 测试

- [x] 100+ Golden Cases（113 条，`tests/requirement_extraction/golden_cases.json`）
- [x] 7 语言测试（中/英离线；de/fr/es/ru/ja 走 LLM 语义层，默认跳过）
- [x] 关键词缺失测试（≥30% purpose 用例不含关键词表原始词）
- [x] 场景歧义测试（concert/stage/wedding/rental 必须保持"继续问"）
- [x] 参数解析测试（视距 12 例 / 尺寸 10 例，含英制与中文）
- [x] 推荐回归测试（沿用既有 660 条全量回归）

## Phase 10 — 删除重复代码

只有测试全部通过后：

- [x] 删除重复关键词表（Sales 侧 environment 关键词已删，改由 Extractor 统一）
- [x] 删除重复场景分类（场景切换清理改为在 Profile 上做）
- [x] 删除 Script Generator 自己的环境推断
- [x] 删除 Router 独立场景词判断（降为 fallback）
- [x] 保留必要 deterministic parser（数字/单位/技术关键词）

---

# 25. 最终目录建议

建议最终形成：

```text
src/
├── core/
│   ├── llm.py
│   ├── requirement_extractor.py
│   ├── requirement_normalizer.py
│   ├── requirement_validator.py
│   └── rule_engine.py
│
├── models/
│   └── requirement.py
│
├── agents/
│   ├── sales/
│   │   └── nodes/
│   │       ├── requirement.py
│   │       └── script_generator.py
│   │
│   └── solution/
│       └── nodes/
│           └── requirement.py
│
├── rag/
│   ├── query_understanding.py
│   ├── parameter_inference.py
│   ├── recommendation_engine.py
│   └── router.py
│
└── tests/
    └── requirement_extraction/
        ├── golden_cases.json
        ├── test_environment.py
        ├── test_purpose.py
        ├── test_installation.py
        ├── test_size.py
        ├── test_distance.py
        └── test_multilingual.py
```

---

# 26. 改造后的职责边界

| 模块 | 负责 | 不负责 |
|---|---|---|
| RequirementExtractor | 理解客户自然语言 | 选产品 |
| Numeric Parser | 尺寸/距离/单位 | 场景理解 |
| RequirementNormalizer | 标准化 token | 产品推荐 |
| RequirementValidator | 冲突/完整性 | 生成销售话术 |
| ParameterInference | 工程参数推断 | 自己重新理解客户 |
| RAG | 产品事实召回 | 最终选型 |
| Recommendation Engine | 筛选/评分/选型 | 理解自然语言 |
| Recommendation Gate | 是否达到推荐条件 | 关键词匹配 |
| Script Generator | 生成销售话术 | 自己判断场景 |

---

# 27. 关键验收指标

本次改造不要只看“能不能运行”。

必须量化：

## Requirement Extraction

目标：

```text
Purpose accuracy        ≥ 95%
Environment accuracy    ≥ 98%
Installation accuracy   ≥ 98%
Display type accuracy   ≥ 98%
Size extraction         ≥ 98%
Distance extraction     ≥ 98%
```

---

## Keyword Independence

测试集中：

```text
至少 30% purpose case
```

不得包含当前关键词表中的原始关键词。

例如：

```text
shopping center
commercial complex
sports complex
corporate boardroom
brand experience center
temporary event installation
```

要求仍然正确归一化。

---

## Gate

重点验证：

```text
只知道：
indoor + purpose

不能直接推荐
```

以及：

```text
concert + stage
```

如果没有明确 indoor/outdoor：

```text
必须继续询问
```

---

## Recommendation

相同 RequirementProfile：

```text
无论客户原话怎么表达
```

最终产品排序必须一致。

例如：

```text
"We need a screen for a shopping mall."
```

和：

```text
"We are installing a display in a commercial complex."
```

应该得到：

```text
同一个 canonical requirement
↓
同一套产品筛选
↓
同一排序结果
```

---

# 28. 最重要的几个禁止事项

### 禁止 1

不要继续无限扩充：

```python
_PURPOSE_KEYWORDS
```

---

### 禁止 2

不要让每个 Agent 都自己判断：

```text
indoor/outdoor
purpose
installation
display_type
```

---

### 禁止 3

不要让 LLM 直接决定：

```text
推荐哪个型号
```

---

### 禁止 4

不要用场景直接推导客户没有说过的工程参数。

例如：

```text
conference
→ viewing distance = 5m
```

不允许。

---

### 禁止 5

不要让 `script_generator.py` 再重新理解客户。

它只消费：

```text
RequirementProfile
+
Recommendation Result
```

---

# 29. 最终目标

最终系统应该从：

```text
客户语言
↓
关键词命中
↓
场景
↓
推荐
```

升级为：

```text
客户语言
↓
Unified Requirement Extractor
↓
自然语言语义理解
+
确定性参数解析
↓
Canonical RequirementProfile
↓
Validator
↓
Recommendation Ready Gate
↓
Product Filter
↓
Scoring
↓
最终产品
```

因此：

> **不是让关键词表变得“更大”，而是让关键词表变得“更不重要”。**

最终关键词只承担：

```text
确定性规则
快速路径
数字/技术参数识别
安全兜底
```

而客户的自然语言场景理解交给统一的 Semantic Requirement Extractor。

---

# 30. 推荐实施优先级

### P0：必须做

1. 统一 RequirementProfile
2. 建立 Unified RequirementExtractor
3. Sales / Solution 共用 Extractor
4. Environment / Purpose / Installation 统一
5. Recommendation Gate 改成结构化字段判断
6. 禁止重复 LLM requirement extraction

### P1：强烈建议

7. confirmed / inferred
8. confidence
9. conflict detection
10. Router 使用 RequirementProfile
11. Script Generator 删除自己的场景判断

### P2：优化

12. keyword fast path
13. semantic extraction fallback
14. LLM result cache
15. multilingual Golden Dataset
16. 自动化 regression

---

# 31. 本次改造后的核心原则

最终只记住下面这句话：

> **关键词负责“确定”，LLM负责“理解”，Parser负责“数字”，Rule Engine负责“约束”，Recommendation Engine负责“选型”。**

这套职责划分最适合当前 `sale_agent_demo`，不需要推翻现有多 Agent + RAG + Recommendation Gate 架构，只需要把现在分散在多个模块中的“需求理解逻辑”收敛成一个统一入口。

---

# 附录：本轮改动文件清单（2026-09-15）

| 文件 | 改动 |
|---|---|
| `src/core/requirement_extractor.py` | 接入 Sales 语义结果（`semantic_override`）、同轮语义缓存、canonical phrase 快速路径、客户原话证据校验、冲突写回 Profile、规则字段来源修正为 explicit |
| `src/core/purpose_normalizer.py` | 补齐计划 Phase 20/27 列出的同义表达（shopping center / commercial complex / football venue / brand experience center / short-term installation 等） |
| `src/core/environment_installation_resolver.py` | retail 统一为 indoor；冲突检测改为"同一轮两个明确信号互相矛盾"才算冲突 |
| `src/models/requirement.py` | 新增 `conflicts` 字段；来源优先级修正为 explicit > scenario_derived > default |
| `src/rag/readiness.py` | Gate 遇冲突直接阻断推荐 |
| `src/rag/query_understanding.py` | 语言提示词去混入英文（修 de/en 误判）、英文词形变化（s/es/d/ed/ing/ly）、`open-air`、中文视距（"6 米远"）、university/lecture hall/language lab 归 classroom |
| `src/rag/router.py` | 新增 `has_structured_requirement()`，结构化需求优先于关键词表 |
| `src/agents/sales/nodes/requirement.py` | 改调统一 Extractor（同轮 LLM 结果作语义输入）；删除历史槽位重复解析与 legacy 场景切换块；场景切换清理改在 Profile 上做 |
| `src/agents/solution/nodes/requirement.py` | 复用统一 Extractor 的同轮语义结果，避免重复 LLM 调用 |
| `tests/requirement_extraction/` | **新增**：golden_cases.json（113 条）+ conftest + 5 个测试模块 |
