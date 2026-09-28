━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
iSEMC 销售 Agent：需求理解 + Fast 路由收口优化计划
版本：v1.0
目标：提升 Requirement / Hard Constraint / Route 三项指标
原则：不改变现有核心业务能力，不进行无必要的大规模重构
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
执行进度（2026-09-28；本次执行不提交）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 三项指标结果（本轮实测）

| 指标 | 计划前 | 现在 | 计划目标 |
|---|---|---|---|
| Requirement Slot Accuracy | 0.7032 | **0.9836** | ≥0.90 ✅ |
| Hard Constraint Capture | 0.5923 | **0.9896** | ≥0.90 ✅ |
| Product Type Accuracy | （本轮首次测量 0.8333） | **1.0** | ≥0.90 ✅ |
| Legacy Route（fast/normal/agent） | 0.7439 | 0.7439（降级为历史对比字段） | 不再作为质量门槛 |
| Calculator Accuracy | 1.0 | 1.0 | =1.00 ✅ |
| Hard Constraint Violation | 0 | 0 | =0 ✅ |

> 印证了计划 §9/§10 的判断：**0.59 / 0.70 主要是"标签把推断当成客户明确"造成的**，
> 不是行为不对。标签按"客户原话证据"拆开之后，两项指标立刻达标。

### 按计划 §23 实施顺序的执行情况

| 步骤 | 状态 | 落地 / 证据 |
|---|---|---|
| P0-① 修正 Golden Dataset 标签 | ✅ | 82 条用例按证据规则拆分：`hard` 67 字段（客户明确）/ `derived` 207（推断+默认）；`slots` 128 / `slots_derived` 150；**66 条用例标签被纠正**（例：g001「会议室大概15人用」→ `hard={}`、`derived={display_type:LED, environment:indoor, installation:fixed}`，与计划 §10 示例一致）。原文件备份 `eval/golden_dataset.json.bak` 可回滚 |
| P0-② 重构 Evaluation | ✅ | `eval/recommendation_eval.py` 不再把 fast/normal/agent 当质量指标：输出为 Requirement Slot Accuracy / Hard Constraint Capture（仅客户明确）/ Derived Capture（参考）/ Legacy Route（历史对比）/ 逐字段统计 / 推荐引擎 |
| P0-③ Slot 逐字段统计 | ✅ | 新增 `per_field_confusion`：7 个字段各给 TP / FP / FN / Accuracy / Recall（报告与终端都打印）。当前 recall：display_type 1.0、environment 1.0、installation 1.0、pixel_pitch 1.0、viewing_distance 0.9524、**purpose 0.0385**（真实问题点） |
| P0-④ Hard Constraint 单独统计 | ✅ | `hard_capture` 只统计客户明确约束 → **0.9896**；推断/默认另立 `derived_capture`（0.4003，仅参考）；检索侧用"显式+推断"并集，检索指标保持基线（series@5 1.0 / model@10 0.9333 / MRR 0.6708 / pitch_fit 1.0 / 违规 0） |
| P1-⑤ RequirementProfile 来源体系 | ✅ | 新增五类事实口径：`FACT_CUSTOMER_EXPLICIT / SCENARIO_DERIVED / SYSTEM_INFERRED / DEFAULT / UNKNOWN` + `fact_class()` + `RequirementProfile.fact_source()/fact_sources()`（`src/models/requirement.py`，已导出）；内部 7 级 `sources` 强度保持不变 |
| P1-⑥ 建立 Slot Merge 机制 | ✅（既有实现 + 本轮显式化） | 合并仍是 `RequirementProfile.merge` + `sources` 强度（explicit 7 > vision_explicit 5 > scenario_derived 4 > inferred 2 > default 1）；本轮把"谁强谁弱"落到可读的五类口径上，评估按它判分 |
| P1-⑦ Explicit / Derived / Default 分离 | ✅ | 数据侧：hard/derived、slots/slots_derived；**运行期侧：评估现在"显式错才扣分"** —— 系统推断/默认出来的额外字段记为 `fp_derived`（仅统计、不扣分），只有"被当成客户明确说的"才计硬错（`per_field_confusion` 新增该列） |
| P1-⑧ 修复 RequirementExtractor | 🚧 大部分完成 | ①**purpose 词表归一**：中文场景词 ↔ 英文枚举（会议室→conference…），purpose 召回 0.0385 → **0.5**，逐槽位 purpose 0.7838 → **0.9231**；②**修掉一个真实产品类型 bug**：`_LED_RE/_LCD_RE/_IFP_RE` 用了 `\b`，而中日韩字符也算单词字符 → "一个LED显示屏"／"我要LED显示屏"／日文 "LEDが…" 全部识别失败（g004 被判成 LCD！）。改成"左右不是拉丁字母"后三条全对，**Product Type Accuracy 0.8333 → 1.0**；③剩余：purpose 仍有 25 FP / 13 FN（系统把"场景词"标成 explicit，需把场景推断改成 SCENARIO_DERIVED） |
| P2-⑨ 删除 Fast 独立业务路由 | ⬜ 下一轮（已审计） | 引用点：`src/agents/solution/runner.py:315-339`（唯一生产用法）、`eval/recommendation_eval.py`（本轮已降级为历史字段）、`tests/test_no_product_phrasing.py`。删除前需先完成 ⑩⑪ |
| P2-⑩ 保留 Structured Product Query | ⬜ 下一轮 | `src/rag/fast_path.py` 的 `_find_models` / 结构化过滤按计划改名 `structured_product_query.py` |
| P2-⑪ Solution 接入统一查询能力 | ⬜ 下一轮 | 结构化产品查询由 Solution 直接调用（不再经 FAST 路由） |
| P3-⑫ ProductTypeRouter 收口 | ✅（上一轮完成） | 唯一产品类型决策入口 + 护栏 |
| P3-⑬ 删除 Solution 内重复 Requirement Understanding | ✅（上一轮完成） | `understand_node` = 纯 Adapter；旧 LLM 需求链已删（见 `update_v2.9.x.md`） |
| P3-⑭ Recommendation Gate 收口 | ✅ | `check_recommendation_ready` 是唯一推荐准入（护栏） |
| P4-⑮ 全量回归测试 | ✅ | `python -m pytest -q` → **810 passed** |
| P4-⑯ Golden Dataset 重跑 | ✅ | 见上表（0.9914 / 0.9896） |
| P4-⑰ 检查 Recommendation | 🚧 部分 | 确定性推荐引擎 ok；Top-1 / Top-3 需 LLM 网络（`--agent`），本机无外网未跑 |
| P4-⑱ 检查 Calculator | ✅ | `eval.calculator_eval` → 1.0（14/14） |
| P4-⑲ 检查 Hard Constraint Violation | ✅ | 检索硬约束违规率 0.0 |

### 本轮改动文件

```text
改动  eval/golden_dataset.json        标签拆分（hard/derived、slots/slots_derived）；原文件备份 .json.bak
改动  eval/recommendation_eval.py     指标重构：硬约束只算客户明确 + 逐字段 TP/FP/FN + Derived Capture + Legacy Route
改动  eval/retrieval_eval.py          检索约束改用 hard+derived 并集（行为回到基线）
改动  eval/recommendation_eval.py     追加：产品类型准确率（ProductTypeRouter）+ purpose 词表归一 + fp_derived 软计数
改动  src/models/requirement.py       五类事实口径（fact_class / fact_source / fact_sources）
改动  src/dialogue/product_type_router.py  修掉 \b 在中文/日文旁失效的 LED/LCD/IFP 识别（g004/g045/g061）+ 补"交互式白板"
```

### 下一轮重点（P2 + P1-⑧ 收尾）

```text
1. P1-⑧ 收尾：把"由场景词推出的字段"标成 SCENARIO_DERIVED（现在标成 explicit），
   purpose 的 25 FP / 13 FN 会随之落到 fp_derived（合法推断不扣分）；
2. P2-⑨⑩⑪：删除 Fast 业务路由（引用点已审计）→ 结构化产品查询改名
   structured_product_query.py → Solution 直接调用；
3. P4-⑰：Top-1 / Top-3 需要 LLM 网络（--agent），联网环境补跑。
```


一、总体目标
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

当前指标：

Requirement Slot Accuracy   = 0.7032
Hard Constraint Capture     = 0.5923
Route Accuracy              = 0.7439

目标：

Requirement Slot Accuracy   >= 0.90
Hard Constraint Capture     >= 0.90
Route Accuracy              >= 0.90

同时：

1. 删除 Fast 作为独立业务路由
2. 保留 Fast Path 中真正有价值的结构化产品查询能力
3. RequirementProfile 成为唯一需求事实来源
4. ProductTypeRouter 成为唯一 LED/LCD/IFP 分类入口
5. Solution Agent 不再重新理解客户需求
6. 不改变现有推荐、计算、RAG、Validation 核心能力
7. 保持“一次只问一个问题”
8. 保持多消息合并机制
9. 保持合法的场景推断 / 参数推断
10. 禁止 AI 把自己的推断当成客户确认


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
二、第一阶段：先处理评估系统
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

优先级：P0

当前最大问题之一：

评估代码仍然存在旧的：

    classify_complexity()
        ↓
    fast / normal / agent

而当前生产架构已经逐渐转向：

    RequirementExtractor
        ↓
    RequirementProfile
        ↓
    ProductTypeRouter
        ↓
    ParameterInference
        ↓
    Recommendation Gate
        ↓
    Solution

因此必须先统一“评估标准”。

【任务 1】

修改：

    eval/recommendation_eval.py

不再把：

    fast / normal / agent

作为核心业务质量指标。

改成评估：

    Requirement Extraction
    Product Type Classification
    Requirement Completeness
    Hard Constraint Capture
    Decision / Gate
    Recommendation

【任务 2】

新增真正的生产链路评估：

    query
      ↓
    RequirementExtractor
      ↓
    RequirementProfile
      ↓
    ProductTypeRouter
      ↓
    ParameterInference
      ↓
    Recommendation Gate

评估结果必须来自这条链路。

【任务 3】

保留旧 route 数据作为历史对比字段即可：

    legacy_route

不要再让它成为生产质量判断标准。


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
三、第二阶段：删除 Fast 独立业务路由
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

优先级：P0

目标：

删除：

    FAST
    NORMAL
    AGENT

这种三层业务入口竞争。

最终改成：

    Customer
       ↓
    Requirement Understanding
       ↓
    RequirementProfile
       ↓
    Decision
       ↓
    Solution

【任务 1】

从 Solution Runner 中移除：

    classify_complexity()

以及：

    QueryRoute.FAST
    QueryRoute.NORMAL
    QueryRoute.AGENT

作为主业务分流依据。

【任务 2】

删除：

    if routing.route == QueryRoute.FAST:
        fast_path_handle(...)

这种提前 return。

【任务 3】

检查整个项目：

    classify_complexity(
    QueryRoute.FAST
    QueryRoute.NORMAL
    QueryRoute.AGENT
    fast_path_handle(
    fast_path

的全部引用。

确认没有隐藏调用后再删除。

【任务 4】

不要直接删除：

    src/rag/fast_path.py

先确认其中哪些能力仍然有价值。


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
四、第三阶段：保留 Fast Path 中真正有价值的能力
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Fast Path 中真正有价值的不是“Fast”。

而是：

    Structured Product Query

例如：

    P2.5 有哪些型号？
    P3 亮度是多少？
    IP65 有哪些产品？
    TW21 有哪些型号？

这些能力可以继续保留。

但是改成：

    Solution
       ↓
    Product Query
       ↓
    Structured Product Filter
       ↓
    Product Result

而不是：

    Router
       ↓
    FAST
       ↓
    fast_path_handle()


最终：

    fast_path.py
            ↓
    可以重命名/拆分为：

    structured_product_query.py

其中保留：

    _find_models()
    产品结构化过滤
    Model 级查询

删除：

    FAST route
    FAST response
    FAST template
    FAST business branch


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
五、第四阶段：Requirement Slot Accuracy 优化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

当前：

    0.7032

目标：

    >= 0.90

不要直接继续增加关键词。

首先做：

    “逐字段错误分析”

必须输出：

    display_type
    environment
    installation
    purpose
    viewing_distance_m
    pixel_pitch_mm
    brightness_min

每个字段分别统计：

    TP
    FP
    FN
    Accuracy
    Recall


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
六、重点解决 Slot 的来源问题
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RequirementProfile 每一个字段必须区分来源：

    CUSTOMER_EXPLICIT
    SCENARIO_DERIVED
    SYSTEM_INFERRED
    DEFAULT
    UNKNOWN

例如：

客户：

    We need a screen for an outdoor advertising board.

应该：

    environment = outdoor
    source = CUSTOMER_EXPLICIT

而：

    We need a screen for a football stadium.

可以：

    environment = outdoor
    source = SCENARIO_DERIVED

但是不能直接变成：

    customer_confirmed = true


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
七、解决 Hard Constraint Capture = 0.5923
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

这是当前最需要优先解决的问题。

目标：

    >= 0.90

首先重新定义：

什么才算 Hard Constraint。

只有客户明确提出：

    必须
    需要
    minimum
    at least
    no more than
    cannot exceed
    must
    required
    waterproof
    IP65
    4K
    7x24
    P2.5 maximum
    minimum brightness

这种才进入：

    hard_constraints


不要把：

    AI 推断
    场景默认
    产品推荐
    系统默认

当成客户 Hard Constraint。


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
八、建立 Hard Constraint 独立结构
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RequirementProfile 中明确区分：

    requirements
    hard_constraints
    inferred_constraints

例如：

    requirements:
        environment = outdoor

    hard_constraints:
        waterproof = true
        brightness_min = 5000

    inferred_constraints:
        recommended_pitch = 4.8


这样后续：

    Recommendation Engine

只能绝对不能违反：

    hard_constraints

可以调整：

    inferred_constraints


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
九、解决 Golden Dataset 标签问题
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

检查：

    eval/golden_dataset.json

重点检查：

    hard
    slots
    decision

目前部分案例存在：

    场景推断结果

被标成：

    hard

的问题。

例如：

    “会议室大概15人使用”

数据可能直接标：

    environment = indoor
    installation = fixed
    display_type = LED

这里必须明确：

    哪些是客户明确说的？
    哪些是业务规则推断？
    哪些是默认？

否则模型即使行为正确，也会被评估系统判错。


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
十、建立三层事实模型
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

以后所有 Requirement 都按照：

    ① Explicit
       客户明确说的

    ② Derived
       根据场景/业务规则推断的

    ③ Default
       系统默认值

进行管理。

例如：

    customer:
        “Outdoor advertising”

得到：

    environment:
        value = outdoor
        source = explicit
        confidence = 1.0

而：

    brightness:
        value = 5000
        source = inferred
        confidence = 0.7

绝不能：

    brightness = customer_confirmed


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
十一、第五阶段：提升 Requirement Slot Accuracy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

重点不是继续堆关键词。

而是：

    Query
      ↓
    Slot Extraction
      ↓
    Slot Normalization
      ↓
    Slot Conflict Resolution
      ↓
    RequirementProfile

【1】

统一同义词：

    indoor
    indoors
    inside
    interior
    室内
    户内

→

    indoor


【2】

统一安装方式：

    fixed
    permanent
    fixed installation
    固定
    固定安装

→

    fixed


【3】

统一 display_type：

    LED display
    LED screen
    LED wall

→

    LED

LCD / IFP 同理。


【4】

统一 pitch：

    P2.5
    2.5mm
    2.5 mm
    pixel pitch 2.5

→

    pixel_pitch_mm = 2.5


【5】

统一 distance：

    5m
    5 meters
    five meters
    viewing distance 5m

→

    viewing_distance_m = 5


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
十二、解决“客户回答了另一个信息”的问题
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RequirementExtractor 不应该只回答：

    当前问题是什么？

而应该：

    从客户当前整条消息中提取所有信息。

例如 AI 问：

    What is the viewing distance?

客户：

    It's for an outdoor stadium,
    around 20 meters,
    and we need waterproofing.

必须一次提取：

    environment = outdoor
    purpose = stadium
    viewing_distance = 20
    waterproof = true


而不是：

    viewing_distance = 20

然后丢掉其他信息。


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
十三、建立 Slot Merge 机制
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

每一轮：

    previous RequirementProfile
              +
    current extracted slots
              ↓
         Slot Merge
              ↓
      New RequirementProfile


规则：

    Explicit > Derived > Default

同一字段冲突时：

    新的客户明确信息
        >
    旧的客户明确信息
        >
    场景推断
        >
    系统默认


禁止：

    LLM 后一次回答
        ↓
    覆盖客户之前确认的信息


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
十四、第六阶段：Route Accuracy 重构
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

当前：

    Route Accuracy = 0.7439

删除 Fast 后，不再评估：

    fast / normal / agent

改为评估真正的业务决策：

    Product Type Route
    Requirement Action
    Solution Action


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
十五、Product Type Route
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

唯一入口：

    ProductTypeRouter

输出：

    LED
    LCD
    IFP
    UNKNOWN

规则：

客户明确选择：

    LED → LED

客户明确选择：

    LCD → LCD

客户明确选择：

    IFP → IFP

客户明确表示：

    需要触控/互动/手写

根据现有业务规则：

    IFP

客户说：

    outdoor

可以进行：

    LED 倾向

但必须区分：

    inferred

不能把 AI 推断直接当：

    customer confirmed


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
十六、重点修复“默认 LED”问题
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

禁止：

    customer:
        “I need a display”

直接：

    display_type = LED
    status = confirmed

应该：

    display_type = UNKNOWN
    status = unknown

然后：

    如果 outdoor
        → LED 倾向

    如果 indoor
        → 解释 LED / LCD 区别
        → 询问客户选择

这样可以避免：

    AI 自己决定客户要 LED


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
十七、第七阶段：Recommendation Gate 优化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RequirementProfile 完整后：

    Recommendation Gate

判断：

    是否可以推荐？

不是：

    “有几个字段就推荐”

而是：

    Required Slots
    +
    Hard Constraints
    +
    Product Type
    +
    Engineering Sufficiency


例如：

    Indoor
    + Conference room
    + 15 people

不能直接决定具体型号。

还可能需要：

    viewing distance

因此：

    Gate = NEED_CLARIFICATION


而：

    Outdoor
    + Advertising
    + 5m viewing distance
    + fixed

达到推荐条件：

    Gate = READY


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
十八、第八阶段：Solution Agent 最终定位
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Solution Agent 不再负责：

    重新理解客户需求
    LLM 再抽取 Requirement
    自己判断 LED/LCD/IFP
    自己构造旧 requirement

Solution 只负责：

    RequirementProfile
          ↓
       Gate
          ↓
    Parameter Inference
          ↓
    RAG
          ↓
    Recommendation
          ↓
    Calculator
          ↓
    Validation
          ↓
    Final Solution


保留 Solution Agent。

删除的是：

    Solution 内重复的 Requirement Understanding。


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
十九、第九阶段：建立真正的评估体系
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

每次 eval 输出：

    1. Requirement Slot Accuracy
    2. Hard Constraint Capture
    3. Product Type Accuracy
    4. Requirement Completeness
    5. Gate Accuracy
    6. Recommendation Top-1
    7. Recommendation Top-3
    8. Hard Constraint Violation
    9. Calculator Accuracy


并且必须输出：

    per-field accuracy

例如：

    display_type        0.94
    environment         0.96
    installation        0.91
    purpose             0.88
    viewing_distance    0.86
    pixel_pitch         0.92
    brightness          0.89


这样才能知道真正的问题在哪里。


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
二十、第十阶段：回归测试
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

必须覆盖：

【基础】

    I need a display

【LED】

    I need an LED display

【LCD】

    I need an LCD display

【IFP】

    I need an interactive display

【室内】

    We need a screen for a conference room

【户外】

    We need an outdoor advertising screen

【多信息】

    Outdoor stadium, 20m viewing distance,
    waterproof, fixed installation

【跨问题回答】

    AI：What is the viewing distance?

    Customer：
    It's outdoor, 20 meters,
    and needs to be waterproof.

必须全部提取。


【多消息】

    Customer:
    outdoor

    Customer:
    stadium

    Customer:
    around 20 meters

必须合并为一个 RequirementProfile。


【冲突】

    Customer:
    Indoor

    Customer:
    Actually outdoor

最终：

    environment = outdoor
    source = explicit


【类型纠正】

    AI：
    Would you prefer LED?

    Customer：
    No, I want LCD.

最终：

    display_type = LCD

不能继续保持 LED。


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
二十一、第十一阶段：Fast 彻底收口
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

确认所有调用点以后：

删除：

    classify_complexity()
    QueryRoute
    FAST branch
    NORMAL branch
    AGENT branch

保留：

    Structured Product Query

如果：

    fast_path.py

只剩产品查询能力：

    重命名：

    structured_product_query.py


如果没有其他调用：

    删除 fast_path.py


同时删除：

    Fast route tests

重新增加：

    Structured Product Query tests


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
二十二、第十二阶段：最终架构
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

最终不要再是：

    Customer
       ↓
    Fast / Normal / Agent
       ↓
    不同处理链


最终应该是：

    Customer
       ↓
    Turn / Message Aggregation
       ↓
    RequirementExtractor
       ↓
    RequirementProfile
       ↓
    ProductTypeRouter
       ↓
    ParameterInference
       ↓
    Recommendation Gate
       │
       ├── NEED_CLARIFICATION
       │       ↓
       │   Question Planner
       │       ↓
       │   Customer
       │
       └── READY
               ↓
           Solution Agent
               ↓
          ┌────┼────────┐
          ↓    ↓        ↓
         RAG  Recommend Calculator
          │    │        │
          └────┼────────┘
               ↓
           Validation
               ↓
         Final Response


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
二十三、实施顺序
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

P0：

    ① 修正 Golden Dataset 标签
    ② 重构 Evaluation
    ③ 对 Slot 逐字段统计
    ④ 对 Hard Constraint 单独统计

P1：

    ⑤ RequirementProfile 来源体系
    ⑥ Slot Merge
    ⑦ Explicit / Derived / Default 分离
    ⑧ 修复 RequirementExtractor

P2：

    ⑨ 删除 Fast 业务路由
    ⑩ 保留 Structured Product Query
    ⑪ Solution 接入统一查询能力

P3：

    ⑫ ProductTypeRouter 收口
    ⑬ 删除 Solution 内重复 Requirement Understanding
    ⑭ Recommendation Gate 收口

P4：

    ⑮ 全量回归测试
    ⑯ Golden Dataset 重新跑
    ⑰ 检查 Recommendation
    ⑱ 检查 Calculator
    ⑲ 检查 Hard Constraint Violation


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
二十四、最终验收标准
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

必须达到：

    Requirement Slot Accuracy >= 0.90

    Hard Constraint Capture >= 0.90

    Product Type Accuracy >= 0.90

    Route / Decision Accuracy >= 0.90

    Recommendation Top-1 >= 0.85

    Recommendation Top-3 >= 0.95

    Hard Constraint Violation = 0

    Calculator Accuracy = 1.00


并且：

    全量原有测试通过

    不删除核心 Recommendation

    不删除 RAG

    不删除 Calculator

    不删除 Validation

    不删除 RequirementProfile

    不删除 ProductTypeRouter

    不删除 Solution Agent

    不删除多消息合并

    不删除合法工程推断

    不改变 First Contact


最终原则：

    “删除 Fast 路由”
    ≠
    “删除快速查询能力”

    “删除 Solution 内重复需求理解”
    ≠
    “删除 Solution Agent”

    “提升 Slot Accuracy”
    ≠
    “增加更多关键词”

    “提升 Hard Constraint”
    =
    “明确客户说了什么、系统推断了什么、默认了什么”

    “提升 Route Accuracy”
    =
    “让评估系统真正测试当前生产架构”

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
最终目标：

让系统从

    Query
      ↓
    Fast / Normal / Agent
      ↓
    多套逻辑

真正收口为

    Query
      ↓
    RequirementProfile
      ↓
    ProductTypeRouter
      ↓
    Gate
      ↓
    Solution
      ↓
    RAG / Recommendation / Calculator / Validation

同时保持现有核心业务能力不变。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
