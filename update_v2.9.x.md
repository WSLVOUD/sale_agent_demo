============================================================
iSEMC 项目最新架构收口计划
版本：v2.9.x → 最终职责收口版
目标：解决 Solution Agent 内部残留的旧需求理解链
原则：不删除 Solution、不改变核心业务功能、不重写成熟推荐链
============================================================

============================================================
执行进度（2026-09-28；本次执行不提交）
============================================================

| 阶段 | 状态 | 落地 / 证据 |
|---|---|---|
| 第一阶段 冻结核心功能 | ✅ | 冻结清单（LED/LCD/IFP 链、RAG、Gate、Calculator、First Contact、Session、聚合、Turn Lock）与 `docs/refactor/baseline.md` 一致；本轮改动后 **810 passed**、评测与基线逐位一致 |
| 第二阶段 RequirementProfile 唯一真值 | ✅ | 审计 14 处 `state["requirement"]` 用法并分类；业务决策模块（Gate / ProductTypeRouter / query_understanding / classify）**不再读** legacy requirement（新护栏）；`parameter_inference` 改为**档案优先**：`profile_to_legacy(profile)` 覆盖旧字典同名字段，旧字典只补缺 |
| 第三阶段 处理 understand_node | ✅ | 改成**纯 Adapter**：读 `requirement_profile` → `check_recommendation_ready` → 返回 `info_sufficient/missing_info`；没有档案 → `REQUIREMENT_NOT_READY`（不再自己造需求）。文件 663 → **275 行** |
| 第四阶段 删除 Solution 内部 LLM 需求提取 | ✅ | 删除 `_build_requirement_prompt` / `_parse_requirement_response` / `get_llm` 调用 / JSON 解析链；无用 import 一并清理 |
| 第五阶段 删除 Solution 内部 Display Type 判断 | ✅ | 删除 `infer_display_type()`（"IFP 否则 LED"）；Solution 全目录禁止出现 `infer_display_type` / `route_display_type` / `_IFP_RE` / `_LED_RE`（护栏） |
| 第六阶段 保留并规范 IFP 自动识别 | ✅ | IFP 意图仍由 `utils/ifp_intent` + `ProductTypeRouter._IFP_RE` 判定：护栏断言"handwriting / interactive whiteboard → LCD + subtype=IFP"；Solution 侧二次 IFP 锁定已删除 |
| 第七阶段 整理 clarify_node | ✅ | 只保留：档案 → Gate → 一个问题；删除面积/亮度/分辨率/display_type/会议室 IFP 等旧启发式 |
| 第八阶段 保留合法工程推断、统一出口 | ✅ | 工程推断统一在 `rag.parameter_inference`（`infer_parameters_node` 直接委托）；档案是唯一真值 |
| 第九阶段 删除 Solution 中的重复工程推断 | ✅ | 删除 clarify 里的 面积→视距/尺寸、indoor/outdoor→亮度、点间距→分辨率 重复块；权威实现留在 `parameter_inference` 与 `engineering/*` |
| 第十阶段 Recommendation Gate 最终收口 | ✅ | `recommendation_gate_node` 只用 `check_recommendation_ready(profile)`；推荐入口 `RecommendationService` 必过 Gate（护栏） |
| 第十一阶段 LCD / LED / IFP 路由边界 | ✅ | ProductTypeRouter 唯一决策入口（护栏）；新增"LCD 不会被场景悄悄改成 LED"；indoor 未指定类型不强制 LED（既有收口层闸门） |
| 第十二阶段 状态结构最终清理 | ✅ | 核心状态与兼容字段在 README「Solution 职责边界」+ `dependency_final.md` 记录；legacy 字典读取者只允许 4 个兼容消费者（护栏钉死） |
| 第十三阶段 测试增加架构约束 | ✅ | 新增 `tests/architecture/test_architecture_v29x_contracts.py`（10 条，覆盖计划第 1~9 条；10~15 条由既有用例与评测覆盖，文件头有对照表）。架构护栏合计 **70 条** |
| 第十四阶段 测试历史文件整理 | 🚧 分步 | 分类与方案见 `docs/refactor/test_inventory.md`；`tests/architecture/` 已迁移；其余功能域分批搬（先统一 `__file__` 路径样板） |
| 第十五阶段 README 最终同步 | ✅ | 修正已不存在的内容（EnhancedMemoryStore / SQLite 规划 / `memory/enhanced.py` / 591 条测试数）；新增「Solution Agent 职责边界」表；测试数更新为 810 |
| 第十六阶段 最终文件职责检查 | ✅ | 见下方职责表；全量 **810 passed**；`eval.calculator_eval` 1.0、`eval.recommendation_eval` 0.7032/0.5923/0.7439 与基线一致 |

### 最终文件职责（第十六阶段核对结果）

```text
RequirementExtractor      → 只理解需求（Sales 侧唯一调用方）
RequirementProfile        → 需求唯一真值（只读）
ProductTypeRouter         → 唯一产品类型决策（LED / LCD(IFP)）
ParameterInference        → 唯一工程参数推断出口（档案优先）
RecommendationGate        → 唯一推荐准入
Solution Agent            → 执行方案（understand=Adapter / gate / clarify / retrieve / recommend / reflect）
RAG                       → 产品事实
Recommendation Engine     → 产品选择
Calculator                → 工程计算
Validation                → 最终验证
Response                  → 对外表达
```

### 本轮改动文件

```text
改动  src/agents/solution/nodes/requirement.py   删旧需求链；understand 改 Adapter；clarify 精简（663 → 275 行）
改动  src/agents/solution/nodes/retrieval.py     不再用 legacy requirement 补 purpose（不写档案）
改动  src/rag/parameter_inference.py             档案优先（同名字段以档案为准）
新增  tests/architecture/test_architecture_v29x_contracts.py（10 条）
改动  tests/architecture/test_architecture_agents_and_recommendation.py（3 条旧断言按新架构改写）
改动  README.md                                  去掉已不存在的内容 + 新增 Solution 职责边界
```

### 明确暂缓

```text
1. 第十四阶段的"历史测试改名/搬迁"：先统一老测试的 __file__ 路径样板再整批搬
2. IFP 仍是 LCD 的子类型（第一层不出 IFP）—— v2.9.x 已确认口径，
   本计划的"IFP 锁定"落在 ProductTypeRouter 的子类型判定上
```


一、当前最新仓库的核心问题
------------------------------------------------------------

目前项目已经完成大部分架构瘦身，但 Solution Agent 内仍然存在
一套历史遗留的「需求理解/需求推断」逻辑。

主要集中在：

src/agents/solution/nodes/requirement.py

当前仍存在：

1. requirement 旧需求字典
2. requirement_profile 新需求模型
3. understand_node()
4. _build_requirement_prompt()
5. _parse_requirement_response()
6. Solution 内部再次调用 LLM 提取需求
7. infer_display_type()
8. clarify_node() 内大量旧启发式判断
9. Solution 内部重新判断 LED / IFP
10. Solution 内部根据面积推算视距/尺寸
11. Solution 内部根据视距/点间距推断分辨率
12. Solution 内部根据 indoor/outdoor 推断亮度

这会导致：

Customer
   ↓
RequirementExtractor
   ↓
RequirementProfile
   ↓
Solution

本来应该是一条链，

但现在实际上仍然存在：

Customer
   ↓
RequirementExtractor
   ↓
RequirementProfile
   ↓
Solution
   ↓
旧 requirement
   ↓
再次理解
   ↓
再次推断
   ↓
再次判断 Display Type

最终造成多个模块都有能力修改需求。

这也是后续继续出现：
- LCD / LED 分类冲突
- IFP 被重新判断
- 问题重复
- 需求被覆盖
- 推荐过早触发
- Solution 与 Sales Agent 结果不一致

的重要架构风险。


============================================================
二、最终目标架构
============================================================

必须最终收敛成：

                    Customer
                       │
                       ▼
              RequirementExtractor
                       │
                       ▼
               RequirementProfile
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
 ProductTypeRouter   Gate      ParameterInference
          │            │            │
          │            │            │
          └────────────┼────────────┘
                       ▼
                 Solution Agent
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
         RAG     Recommendation   Calculator
                       │
                       ▼
                  Validation
                       │
                       ▼
                  Final Response


职责必须固定：

RequirementExtractor
→ 负责理解客户说了什么

RequirementProfile
→ 唯一的需求真值

ProductTypeRouter
→ 唯一负责 LED / LCD / IFP 分类

ParameterInference
→ 负责允许的工程参数推断

Recommendation Gate
→ 判断当前是否允许推荐

Solution Agent
→ 执行方案，而不是重新理解需求

RAG
→ 找产品事实

Recommendation
→ 选产品

Calculator
→ 做工程计算

Validation
→ 最终检查

Response
→ 组织自然语言


============================================================
三、第一阶段：冻结核心功能
============================================================

这一阶段不要修改业务规则。

必须明确冻结：

1. LED 推荐链
2. LCD 推荐链入口
3. IFP 推荐链入口
4. RAG 检索
5. Hybrid Search
6. Recommendation Engine
7. Hard Filter
8. Pixel Pitch 推断
9. Screen Calculator
10. Resolution Validation
11. Recommendation Ready Gate
12. Calculation Ready Gate
13. First Contact
14. Session Memory
15. 多消息聚合
16. Turn Lock / Dedup
17. 图片需求提取
18. 环境自动识别
19. pitch ↔ viewing distance 的既有合法推断


特别注意：

本次不是重写业务。

只是把「谁负责做什么」重新固定下来。


============================================================
四、第二阶段：RequirementProfile 成为唯一需求真值
============================================================

目标：

整个系统真正做到：

RequirementProfile = 唯一业务需求来源


第一步：

检查所有代码中：

state["requirement"]

的用途。


分成三类：

A. 业务决策用途
→ 必须删除

B. 兼容旧代码用途
→ 暂时保留只读

C. 输出/日志用途
→ 可以保留


最终要求：

任何 Gate / Router / Recommendation / Calculator
不能依赖：

state["requirement"]


必须依赖：

state["requirement_profile"]


最终状态：

RequirementProfile
    ↓
所有业务决策


而不是：

RequirementProfile
    ↓
legacy requirement
    ↓
业务决策


============================================================
五、第三阶段：处理 Solution 的 understand_node
============================================================

这是本次最重要的修改。

当前：

understand_node()

仍然具有：

1. 自己判断需求是否足够
2. 自己调用 RequirementExtractor
3. 自己 fallback 到 LLM
4. 自己解析 JSON
5. 自己 merge requirement
6. 自己生成 missing_info


这些职责必须从 Solution 中退出。


最终：

Solution 不再负责：

「客户到底需要什么？」


Solution 只负责：

「已经知道客户需要什么，现在怎么给方案？」


因此：

understand_node()

不能再成为独立的需求理解 Agent。


推荐处理方式：

方案 A：

直接移除 Solution 内的需求理解节点。


或者：

方案 B：

暂时保留 understand_node 名字，
但把它改造成纯 Adapter：

只做：

state
 ↓
读取 requirement_profile
 ↓
检查状态
 ↓
返回 state


禁止：

LLM
RequirementExtractor
重新解析客户消息
修改需求
重新判断 display_type


最终推荐：

优先方案 A。


============================================================
六、第四阶段：删除 Solution 内部 LLM 需求提取
============================================================

以下函数属于旧需求理解链：

_build_requirement_prompt()
_parse_requirement_response()


必须删除。

同时删除：

get_llm()

在这个 requirement.py 中用于需求提取的调用。


尤其要删除：

llm.invoke(prompt)


这一条链。

原因：

现在已经存在统一：

RequirementExtractor


如果 Solution 再调用一次 LLM：

第一次：

Customer
 ↓
RequirementExtractor
 ↓
Profile


第二次：

Solution
 ↓
LLM
 ↓
Requirement


就会产生两个需求真相。


最终要求：

一个客户输入
→ 一次需求理解
→ 一个 RequirementProfile


============================================================
七、第五阶段：删除 Solution 内部的 Display Type 判断
============================================================

当前存在：

infer_display_type()


其中有：

IFP → IFP
其它 → LED

这部分必须移出 Solution。


原因非常重要：

这会直接绕过新的：

ProductTypeRouter


最终必须保证：

                    ProductTypeRouter
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
             LED          LCD          IFP


只有 ProductTypeRouter
可以决定：

display_type


Solution 不允许再次决定。


尤其不能再存在：

「不是 IFP 就默认 LED」

这种逻辑。


否则客户：

"I need an indoor display"

可能在某个旧分支直接：

LED


而新的分类流程本来应该：

Indoor
 ↓
需要确认 LED / LCD
 ↓
客户选择
 ↓
进入对应链路


============================================================
八、第六阶段：保留并规范 IFP 自动识别
============================================================

这里不能简单删除 IFP 逻辑。

你的业务要求仍然保留：

客户明确表达：

- handwriting
- touch
- interactive
- whiteboard
- interactive classroom
- interactive meeting

等明确 IFP 意图时：

→ ProductTypeRouter 锁定 IFP


也就是说：

删除：

Solution.infer_display_type()


保留：

IFP Intent Detection


但是调用关系改成：

Customer
 ↓
RequirementExtractor
 ↓
IFP Intent
 ↓
ProductTypeRouter
 ↓
IFP


而不是：

Solution
 ↓
has_ifp_intent()
 ↓
IFP


============================================================
九、第七阶段：整理 clarify_node
============================================================

clarify_node 当前仍然承担太多历史逻辑。


必须进行拆分。


最终 clarify_node 只做：

1. 读取 RequirementProfile
2. 读取 Recommendation Gate
3. 获取 next_question
4. 输出一个问题
5. 结束当前轮


最终结构：

RequirementProfile
        ↓
RecommendationGate
        ↓
ready?
   ┌────┴────┐
   │         │
  YES        NO
   │         │
   ▼         ▼
retrieve   clarify
             │
             ▼
       ask one question


clarify_node 不再自己决定：

- LED / LCD / IFP
- 亮度
- 分辨率
- 视距
- 尺寸
- 面积
- 点间距


这些必须由上游统一模型 / Gate / ParameterInference 完成。


============================================================
十、第八阶段：保留合法工程推断，但统一出口
============================================================

这里特别注意：

不是把所有自动推断全部删除。


你之前明确要求保留：

1. 根据明显场景推断 indoor/outdoor
2. pitch ↔ viewing distance 的合理工程推断
3. 参数之间的工程关系


这些功能必须保留。


但是不能继续散落在：

Solution.requirement.py


最终统一：

Customer Input
      ↓
RequirementExtractor
      ↓
RequirementProfile
      ↓
ParameterInference
      ↓
RequirementProfile 更新
      ↓
Gate
      ↓
Solution


也就是说：

「推断功能保留」

但：

「推断代码不能散落」。


============================================================
十一、第九阶段：删除 Solution 中的重复工程推断
============================================================

重点检查并迁移：

1. 面积 → 视距
2. 面积 → 屏幕尺寸
3. 视距 → 点间距
4. 点间距 → 分辨率
5. indoor → 亮度
6. outdoor → 亮度


尤其当前 clarify_node 中：

area_sqm
→ inferred_dist
→ inferred_size


这一整套旧逻辑需要重新确认。


不能简单删除业务能力。


应该：

如果属于正式 ParameterInference 规则：

→ 保留在 ParameterInference


如果属于历史临时启发式：

→ 删除


最终要求：

一个工程参数
只能有一个权威计算/推断入口。


例如：

Viewing Distance
→ 只能由 ParameterInference / 客户明确输入产生


Screen Size
→ 只能由统一 Screen Calculation / ParameterInference 产生


Pixel Pitch
→ 只能由统一 ParameterInference 产生


不能：

Solution 再算一遍。


============================================================
十二、第十阶段：Recommendation Gate 最终收口
============================================================

Recommendation Gate 必须成为唯一推荐入口。


要求：

Solution 不能自行判断：

「我觉得需求够了，可以推荐」


必须：

RequirementProfile
       ↓
check_recommendation_ready()
       ↓
Decision


如果：

ready = false

→ 只能进入 clarify


如果：

ready = true

→ 才能：

retrieve
 ↓
hard filter
 ↓
recommendation
 ↓
calculator
 ↓
validation


最终禁止：

Solution 内部自己判断推荐条件。


============================================================
十三、第十一阶段：LCD / LED / IFP 路由最终边界
============================================================

最终分类规则必须统一到：

ProductTypeRouter


推荐流程：

客户首次表达：

「I need a display」

        ↓

没有 display type
        ↓
不要直接强制 LED
        ↓
询问 LED / LCD


如果客户：

「I need an LED display」

        ↓
LED
        ↓
LED 成熟需求链


如果客户：

「I need an LCD display」

        ↓
LCD
        ↓
LCD 需求链入口


如果客户：

「I need an interactive display」
「I need handwriting」
「I need a digital whiteboard」

        ↓
IFP
        ↓
IFP 需求链入口


如果客户：

「outdoor display」

        ↓
LED


如果客户：

「indoor display」

        ↓
不要自动强制 LED
        ↓
根据 LED/LCD 边界继续确认


这个边界不能再被 Solution 覆盖。


============================================================
十四、第十二阶段：状态结构最终清理
============================================================

最终核心状态建议固定为：

messages
current_message
session_id

requirement_profile

product_type
product_type_reason

recommendation_gate
calculation_gate

parameter_inference

retrieval_context
recommendation_result
calculation_result
validation_result

pending_question
next_action


历史兼容字段：

requirement
missing_info

只能作为：

Legacy Adapter


不能作为：

业务真值。


最终逐步删除。


============================================================
十五、第十三阶段：测试必须增加架构约束
============================================================

这一次不能只测试业务结果。

必须增加：

Architecture Tests


至少测试：

1. Solution 不允许调用 LLM 做需求提取

2. Solution 不允许重新调用 RequirementExtractor

3. Solution 不允许修改 RequirementProfile 的核心需求字段

4. Solution 不允许自行决定 display_type

5. ProductTypeRouter 是唯一 display_type 决策入口

6. Recommendation 只能由 Gate 放行

7. requirement 不再参与新的业务决策

8. LCD 不会被自动变成 LED

9. IFP 明确意图仍然能够锁定 IFP

10. Outdoor 仍然能够进入 LED

11. Indoor + 未指定类型不会直接强制 LED

12. 一轮客户消息只能产生一个问题

13. 多条客户消息聚合后只执行一次需求理解

14. 原有 LED 推荐测试全部保持通过

15. Calculator / Validation 测试全部保持通过


============================================================
十六、第十四阶段：测试历史文件整理
============================================================

目前测试目录仍有较多：

test_v24_*
test_v27_*
test_...


不要直接全部删除。


处理方式：

第一步：

找出仍然覆盖当前代码的测试。


第二步：

把有效测试迁移成当前架构名称。


例如：

test_v27_xxx.py

→

test_requirement_profile_xxx.py


第三步：

已经验证历史 Bug、
且当前架构已经不存在的测试：

→ 删除


最终测试目录应该表达：

「当前系统是什么」

而不是：

「过去系统经历过什么」。


============================================================
十七、第十五阶段：README 最终同步
============================================================

当前 README 仍存在历史架构描述。

重点清理：

1. 旧 Memory 架构
2. SQLite planned
3. 历史 Phase 说明
4. 已删除模块
5. 旧 Requirement 架构
6. 旧 Product Router 架构
7. 旧测试数量
8. 已经不存在的文件


README 最终只描述：

当前真实架构。


建议最终结构：

1. Project Overview
2. Architecture
3. Customer Flow
4. RequirementProfile
5. ProductTypeRouter
6. Solution Agent
7. RAG
8. Recommendation
9. Calculator
10. Validation
11. Vision
12. API
13. Testing
14. Run


不要继续把整个项目历史写进 README。


============================================================
十八、第十六阶段：最终文件职责检查
============================================================

完成后逐文件检查：

RequirementExtractor
→ 只能理解需求


RequirementProfile
→ 需求唯一真值


ProductTypeRouter
→ 唯一产品类型决策


ParameterInference
→ 唯一工程参数推断


RecommendationGate
→ 唯一推荐准入


Solution Agent
→ 执行方案


RAG
→ 产品事实


Recommendation Engine
→ 产品选择


Calculator
→ 工程计算


Validation
→ 最终验证


Response
→ 对外表达


任何一个模块发现：

「我顺便也做一下另外一个模块的工作」

就继续收口。


============================================================
十九、最终禁止出现的架构
============================================================

禁止：

RequirementExtractor
        ↓
RequirementProfile
        ↓
Solution
        ↓
LLM
        ↓
重新生成 requirement


禁止：

ProductTypeRouter
        ↓
LED/LCD/IFP


同时：

Solution
 ↓
重新判断 LED/IFP


禁止：

ParameterInference
        ↓
参数推断


同时：

Solution
 ↓
自己重新推断参数


禁止：

RecommendationGate
        ↓
ready=false


同时：

Solution
 ↓
「我觉得可以推荐」
 ↓
Recommendation


禁止：

RequirementProfile
 ↓
legacy requirement
 ↓
业务决策


============================================================
二十、最终验收标准
============================================================

本次整理完成后，必须满足：

[ ] Solution Agent 没有被删除

[ ] Solution Agent 核心功能全部保留

[ ] RAG 保留

[ ] Recommendation 保留

[ ] Calculator 保留

[ ] Validation 保留

[ ] Gate 保留

[ ] LED 推荐链不受影响

[ ] LCD 入口不受影响

[ ] IFP 入口不受影响

[ ] 图片识别入口不受影响

[ ] 多消息聚合不受影响

[ ] Session Lock 不受影响

[ ] Dedup 不受影响

[ ] RequirementProfile 成为唯一需求真值

[ ] Solution 不再重新理解客户需求

[ ] Solution 不再调用 LLM 提取需求

[ ] Solution 不再拥有独立 display_type 判断

[ ] ProductTypeRouter 成为唯一产品类型决策入口

[ ] ParameterInference 成为工程推断唯一入口

[ ] Recommendation Gate 成为推荐唯一入口

[ ] 不再存在旧 requirement 的业务决策依赖

[ ] 不再出现两个模块同时推断同一个参数

[ ] 不再出现 LCD/LED 路由互相覆盖

[ ] 不再出现 IFP 被 Solution 二次判断

[ ] 一轮最多一个问题

[ ] README 与真实代码一致

[ ] 历史测试完成迁移/清理

[ ] 全量测试通过


============================================================
最终架构
============================================================

                  ┌──────────────────┐
                  │     Customer     │
                  └────────┬─────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ RequirementExtractor│
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  RequirementProfile │
                │   唯一需求真值       │
                └──────┬──────┬───────┘
                       │      │
             ┌─────────┘      └─────────┐
             ▼                          ▼
   ┌──────────────────┐       ┌──────────────────┐
   │ ProductTypeRouter│       │ParameterInference│
   └────────┬─────────┘       └────────┬─────────┘
            │                          │
            └────────────┬─────────────┘
                         ▼
                ┌─────────────────┐
                │ Recommendation  │
                │      Gate       │
                └────────┬────────┘
                         │
                  ready? │
                    ┌────┴────┐
                    │         │
                   NO        YES
                    │         │
                    ▼         ▼
                 Clarify   Solution
                              │
                 ┌────────────┼────────────┐
                 ▼            ▼            ▼
                RAG      Recommendation Calculator
                              │
                              ▼
                         Validation
                              │
                              ▼
                       Final Response


核心原则只有一句：

「Solution 不删除，Solution 里的重复需求理解和重复决策删除。」

这样既不会破坏你现在已经做好的核心功能，又能把当前项目真正从“多个模块都能理解需求”收敛成“一处理解、统一需求、Solution 执行方案”的结构。
============================================================
