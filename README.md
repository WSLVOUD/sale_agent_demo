# LED RAG 智能销售系统

> 基于大模型（DeepSeek）的 LED/LCD/IFP 全品类显示产品智能销售助手，采用多 Agent 协作 + 混合检索（RAG）架构，为销售团队提供实时产品推荐和技术咨询能力。

> **当前版本：v2.9.6（2026-09-24）** ｜ 全量测试：`python -m pytest -q` → **760 条，约 2 分 40 秒**
>
> 当前行为口径集中在下面「当前行为口径」一节；历史版本的逐条变更见文末「变更明细」。
>
> **架构整理（2026-09-24）**：代码结构瘦身与职责收敛见 [`docs/refactor/`](docs/refactor/)，
> 结论见下方「架构整理」一节。文中历史章节的数字（如 591 条）是当时的快照，保留不动。

---

## 架构整理（2026-09-24）

目标：**不改业务行为，只收敛职责**。基线与验收数字都在 [`docs/refactor/`](docs/refactor/)：

| 文档 | 内容 |
|---|---|
| `baseline.md` | 环境 / 启动 / 12 个接口 / 760 条测试 / 评测基线（计算 1.0、槽位 0.7032、路由 0.7439…） |
| `dependency_map.md` | 真实 import 索引 + 分层调用链 + 多入口风险点 |
| `module_inventory.md` | 160 个模块分类（CORE / ADAPTER / COMPATIBILITY / LEGACY / TEST_ONLY） |
| `behavior_baseline.md` | S1~S7 关键对话行为 + 量化基线 + 行为锚点测试 |
| `test_inventory.md` | 测试按功能分类 + 迁移方案 |

本轮实际做的事：

```text
1. 产品类型词汇只保留一处定义（turn_kind），product_type_router 改为引用
2. "一轮最多一个问题"的预算只保留一处定义（action.MAX_QUESTIONS_PER_TURN），
   final_guard / final_response / response_context 全部改为引用
3. 删除两个零引用的旧需求适配层：
   src/core/sales_requirement_adapter.py、src/core/solution_requirement_adapter.py
4. 新增 5 个架构护栏测试文件（34 条）：唯一需求模型 / 唯一产品类型入口 /
   唯一对话决策入口 / 层间边界（RAG 不碰对话、计算层独立、Vision 只抽需求）/
   Sales 与 Solution 共用同一套需求系统 + 推荐单一入口
```

验收（同一环境、与 `baseline.md` 逐项对比）：

```text
python -m pytest -q                          → 760 passed（基线 726，只增不减）
python -m eval.calculator_eval               → 1.0（14/14，与基线一致）
python -m eval.recommendation_eval           → 0.7032 / 0.5923 / 0.7439（与基线逐位一致）
python -m eval.retrieval_eval --limit 12     → model_recall@10 0.9333、MRR 0.6708、违规率 0.0（一致）
```

---

## 项目状态

> 📚 **文档入口**
>
> | 目录 | 内容 |
> |---|---|
> | [`docs/refactor/`](docs/refactor/) | 架构基线与瘦身记录（baseline / 依赖地图 / 模块清单 / 行为基线 / 测试分类 / 最终依赖扫描） |
> | [`docs/history/`](docs/history/) | 历史变更明细（v2.0~v2.9.x 的开发记录，已从 README 迁出） |
> | [`docs/plans/`](docs/plans/) | 各阶段优化计划（含架构瘦身 2.0 计划与未来优化清单） |
> | [`docs/testing/`](docs/testing/) | 测试说明与入口 |
> | [`eval/README.md`](eval/README.md) | 评测体系（Golden Dataset / 检索 / 推荐 / 计算） |

| 模块 | 状态 |
|------|------|
| 销售 Agent（Sales Agent） | ✅ 已完成 |
| 方案 Agent（Solution Agent） | ✅ 已完成 |
| 混合检索（RAG：Dense + Sparse + BM25 + RRF） | ✅ 已完成 |
| 质量评估反射（Reflection Quality Gate） | ✅ 已完成 |
| 三层路由（Fast / Normal / Agent Path） | ✅ 已完成 |
| 增强记忆（EnhancedMemoryStore + CustomerProfile + Summary） | ✅ 已完成 |
| 可观测性（PerfTracker + 可选 Langfuse） | ✅ 已完成 |
| 评估体系（Golden Dataset + 自动评测） | ✅ 已完成 |
| Model 级产品数据与语料（87 Model / 35 Series：56 LED + 21 LCD + 10 IFP） | ✅ 已完成 |
| 推荐就绪 Gate（Recommendation Ready Gate） | ✅ 已完成 |
| 计算就绪 Gate（Calculation Ready Gate） | ✅ 已完成 |
| 确定性推荐引擎（Recommendation Engine） | ✅ 已完成 |
| 确定性校验（替代 LLM Reflection 重新决策） | ✅ 已完成 |
| 箱体 / 模组工程计算（Screen Calculator） | ✅ 已完成 |
| 多语言 Query Understanding + Query Rewrite | ✅ 已完成 |
| **首次客户固定工作流（First Contact）** | ✅ 已完成 |
| **全量需求捕获 + Unknown 容错（每字段最多问两次）** | ✅ 已完成 |
| **智谱视觉需求提取（客户发图片识别需求）** | ✅ 已完成 |
| **需求采集扩展问答（使用场景 / 价位取向）** | ✅ 已完成 |
| **一个项目多条屏体需求（多块屏 / 多品类）** | ✅ 已完成 |
| **联系方式收集已按客户口径删除**（改为人工获取） | ✅ 已移除 |
| **客户决策状态 v2.1（授权 AI / 不知道 / 拒绝 / 延后）** | ✅ 已完成 |
| **观看距离确定性推导 + 点间距物理窗口 v2.2** | ✅ 已完成 |
| **追问节奏：客户说"不知道"先换下一问，最后一轮再问** | ✅ v2.2.5 |
| **服务口径一致性（安装 / 说明书 / 质保不自相矛盾）** | ✅ v2.2.5 |
| **Memory 持久化（SQLite）** | 🚧 规划中 |

### 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env          # 填 DEEPSEEK_API_KEY（可选：ZHIPUAI_API_KEY 用于识图）
python init_vectorstore.py    # 首次运行：建向量库
uvicorn src.api:app --port 8000
```

打开 `http://localhost:8000` 即可对话；接口与配置详见文末「API 端点」「核心配置项」。

### 测试（一棵树：全量 591 条，约 1.5 分钟）

```bash
python -m pytest -q                          # 全量 591 条（约 1.5 分钟）
python -m pytest tests/test_vision_pipeline.py -q   # 单跑某个文件
```

公共设施（环境变量 + fixtures）在仓库根目录的 `conftest.py`。

覆盖：对话决策（Policy / 闲聊承接 / 重复提问 / 单一回复 / 一轮一条回复）、
输入层 Turn 引擎（去重 / 聚合 / 会话锁 / 幂等 / 生成期间补发并轮）、API、
话术与口径（报价、服务口径、交期、自然度）、会话重置、真实日志 Replay、
对话决策 Golden Dataset、自由问答语境（others）、前端聚合（app.js）、
工程计算（箱体 / 模组）与视觉链路 smoke。

> 瘦身口径（客户 2026-09-22）：**客户平时跑的"全量"就是核心回归那 585 条**；
> 同一张用例表里的多个说法（例如 10 条"重新来 / 换产品"的中文说法）已用
> `tests/_cases.py::assert_all_cases` 压在一条测试里 —— 断言一条不少，收集条数下降，
> 失败时一次性列出所有失败 case。
>
> **被精简掉的扩展语料**（需求抽取 Golden Cases、RAG 检索、推荐引擎 / Gate、
> 分辨率 / 可行性、措辞大全等，约 1100 条）**代码还在 git 历史里**，需要时取回：
>
> ```bash
> git log --oneline -- tests/test_retrieval.py          # 找最后一次有它的提交
> git checkout <commit> -- tests/test_retrieval.py      # 取回单个文件
> git checkout <commit> -- tests/requirement_extraction  # 取回整个目录
> ```
>
> 历史章节里写的 `tests/xxx.py` 若不在当前目录树里，就是这一类。

测试环境里 LLM 一律**快速失败**（`LLM_MAX_RETRIES=0`、`LLM_TIMEOUT_SECS=3`）：
失败路径仍然降级到结构化兜底，但不再白等重试的 1s + 2s。

---

## 当前行为口径（v2.3.1）

### 0a. 一轮一个动作 / 一条回复 + 对话状态（v2.6）

```text
硬约束（架构层，不靠 Prompt 补丁）
    ONE TURN → ONE ACTION → ONE RESPONSE
    · 一个客户 turn 最多一条客户可见回复（追加气泡并入同一条，不再另起一条）
    · 一条回复最多一个"需要客户回答"的问题
    · 一轮只有一个最终 DialogueAction（候选可以有多个，其余进 discarded_actions）

双状态
    RequirementProfile    "客户有什么需求"     environment=indoor / size=3×5 / P3 …
    ConversationState     "客户现在在做什么"   last_question_slot / current_answer_slot …
    · AI 每次提问都记 last_ai_question + last_question_slot
    · 客户回答优先匹配上一轮问的那一项（P3 → 答的是 pixel_pitch）
    · 答非所问不丢信息：问 indoor/outdoor、客户答 3×5 → 尺寸照记，环境仍未知
    · 已答 / 已有结论的字段不再问（AskedQuestionRegistry 状态机）

环境（室内外）例外：环境没定就是"唯一最高优先级"，即使客户答的是别的
（"3×5"）也仍然只问这一项；问满两次拿不到才交给 Gate 的 DEFERRED / BLOCKED。
（这条取代 v2.4 的"答非所问先换下一问"，按 v2.6 §10/§12/§13。）
```

实现位置：`src/dialogue/final_response.py`（唯一出口）、
`src/dialogue/conversation_state.py`（对话状态 + 答问匹配）、
`src/dialogue/question_registry.py`（问题状态机）、
`src/dialogue/action.py`（候选 → 唯一 Action）、
`src/orchestrator.py`（`_finalize_turn_response` 收口 + `[Turn]` / `[DecisionAudit]` 日志）。

### 0a+. 不要连续提问：是闲聊才承接，聊需求就追问（2026-09-22 最终口径）

```text
判定看"客户这句话跟需求有没有关系"（AI 在语境里判断，不看关键词）：
    与需求有关（给了参数 / 答了别的一项 / 主动聊需求 / 问业务问题）
        → 直接"接住 + 追问缺的那一项"，**不做只承接**
    与需求无关（闲聊 / 寒暄 / 题外话）
        → 第 1 条：只承接（正文里不许留问句）
        → 第 2 条：**接住这句话 + 提问写在同一条消息里**
          （承接上限 1 条，`LED_RAG_MAX_ACK_STREAK`）
    客户回到需求话题 → 承接计数清零
    同一轮里客户连发多条消息（前端聚合）只算一次
```

> 客户口径原文："客户再说的是需求有关的话题，AI 就不要再寒暄承接了……如果客户
> 聊得是无关的才承接闲聊，然后再客户说第二条闲聊消息的时候，承接客户的这个句话，
> 在同一条消息询问 AI 需求……不要设置什么关键词去触发，让 AI 在语境里面去感知"。
> （上一版口径是"客户没回答上一问 → 先承接 1 条"；实测 bug：客户答 `maybe 5m`
> 是**观看距离**这种需求信息，却只收到一句寒暄，既不追问也不推进。）
> 判定：`src/agents/sales/nodes/requirement.py::is_offtopic_message`
> （LLM 语义层 `semantic_payload` 优先，规则解析兜底）；
> 决策：`src/dialogue/continuation_budget.py::decide_continuation`；
> 收口：`src/orchestrator.py::_finalize_turn_response`（承接轮由
> `_continuation_only_text` 去掉问句）。
> 回归测试：`tests/dialogue/test_ack_streak.py`、`tests/dialogue/test_offtopic_intent.py`。

### 0a++. 客户连发的消息 = 一个 Turn（只回一条）（2026-09-22）

```text
客户: i need a led display
客户: 3*5                       ← 上一轮还在生成时补发
🤖 An LED display gives us plenty to work with … indoors or outdoors?
🤖 A 3 by 5 meter screen … fixed or rental?     ← ❌ 又回了一轮（客户看到两个问题）
```

规则：客户在"上一轮还在生成"期间补发的消息，必须并进**同一个 turn**，只回**一条**
（AI 把两条消息当成一句话回答）。

| 位置 | 做法 |
|------|------|
| 前端 `static/app.js` | 补发的消息**立刻送出去**（以前 `setTimeout(..., 800)` 一直等到上一轮回复才发 → 后端只能当成两个 turn）；并发请求用 `inflightCount` 计数，"正在输入"等全部结束再收起；被并入的那条响应带 `duplicate=true`，前端不渲染第二个气泡 |
| 后端 `src/input/turn_executor.py` | 生成期间到达的消息并进正在跑的 turn，回滚"作废那一版"的状态后**重跑**（只提交一条回复）；万一补发消息错过并入窗口（那一轮已生成完回复），**另起一轮回答它**，绝不吞掉客户消息 |

> 回归测试：`tests/input/test_turn_engine_concurrency.py`（生成期间补发 → 同一 turn、
> 只一条回复、并入方 `joined=True`；错过窗口也不能丢消息）、
> `tests/test_frontend_turn_merge.py`（前端不再等上一轮、并发计数、duplicate 不渲染、
> `app.js?v=` 版本号）。

### 0a+++. 自由问答也看得见语境（others / product_question，2026-09-22）

```text
🤖 … Shall I prepare the quotation?
👤 yew
🤖 Sorry, I'm not sure I caught that…
👤 yes
🤖 Sure, go ahead, what would you like to know?        ← 没绑到"要不要出报价"
👤 give me quatatio
🤖 … The exact installation type … The final dimensions …   ← 又问已经答过的
```

根因：销售那条链（classify / 需求理解 / 话术出口）都带最近 50 条记忆，但
**自由问答这条路径一次都不带**；而且它的"已确认需求"来自 legacy 投影字典
（indoor / is_rental / size），护栏只认 environment / installation /
target_width_mm → 环境、安装方式、尺寸全丢。

| 位置 | 改动 |
|------|------|
| `src/agents/solution/nodes/others.py` | 提示词加入 `Recent conversation`（最近 50 条、带 role，与 Sales 同一个窗口）；规则明确：客户短回复（yes / ok / 打错字）先当作"在回答你最后问的那句"，`Confirmed customer requirements` 里的信息不许再问；检索关键词也改成带最近几轮语境 |
| `src/rag/model_guard.py` | `requirement_summary()` 同时认 legacy 键（indoor/outdoor → environment，is_rental → installation，size → 尺寸），于是"已确认需求"清单 / 检索过滤 / 矛盾片段过滤在这条路径重新生效 |
| `src/memory/store.py` + `src/orchestrator.py` | 新增 `replace_last_assistant()`：自由问答这一轮 Sales 只写了占位符（"Sure."），收口时用**真正发给客户的答复**替换它 —— 否则下一轮那 50 条窗口里看不到 AI 自己说过什么 |

> 回归测试：`tests/dialogue/test_others_context.py`（提示词带语境 / 已确认需求完整 /
> legacy 字典被识别 / 占位符被替换）。

### 0a++++. LCD / IFP 也进 Model 级语料（2026-09-22）

```text
data/LCD display.txt（客户资料，IFP 混在 LCD 里）
    → data/lcd_products.json   21 个 LCD 型号：P 系列单屏 8 / B 系列拼接 7 / P TM 触摸 6
    → data/ifp_products.json   Omni G4/G4C + Omni K4/K4C，10 个型号（按尺寸拆成 Model 级）
    → 向量库 Model 级语料 87 条 = 56 LED + 21 LCD + 10 IFP（35 个 Series）
```

- 资料里没有的字段一律留空（IFP 的价格 / 适用面积、单体显示器的拼缝），**不编数据**
- **拼接口径**（客户口径 2026-09-22）：**LCD 可以拼接，但拼缝可见**（B 系列拼接屏
  bezel 0.88mm / 1.7mm / 3.5mm；P 系列与 P TM 拼起来也有可见边框）；**IFP 不能拼接**。
  这条写进了 `data/LCD display.txt`、两份 JSON（`splicing_supported` / `splicing_note`）
  以及检索文本与 metadata，模型回答"能不能拼"时以它为准
- `src/rag/json_loader.py` 新增非 LED 的 Model 级文档构建：与 LED 共用同一套 metadata
  （`level=model` / `indoor,outdoor` / `display_type` / `environment_metadata_version` …），
  但**不写**点间距 / 模组 / 箱体这些 LED 专有字段
- 语料 schema 升到 **v7**（新增 LCD/IFP + 拼接口径）→ 旧向量库会被启动自检判为过期并自动重建
- 推荐引擎目前仍然只吃 LED 的 `CanonicalModel`（LCD/IFP 先供检索问答用）；
  要让 LCD/IFP 也能被"推荐"，是下一步的事
- 重建：`python init_vectorstore.py`（或直接启动服务，启动自检发现条数不符会自动重建）
- 回归：`tests/test_api_http.py`（`record_count=87`、`lcd_records=21`、`ifp_records=10`）

### 0. 提问顺序与节奏（v2.4）

```text
提问池（顺序按会话随机洗牌，同一会话内稳定可复现）
    室内外 · 使用场景 · 固装租赁 · 价位取向 · P值 · 观看距离 · 尺寸

Phase 1 随机轮
    · 每轮从"还没问过"的池子里随机挑一个问
    · 客户说"不知道" / 没回答 / 答非所问 → 本轮不再问这一项，换下一个
      （例外：室内外没定时不让位，见上文 0a —— 按 v2.6 §12/§13）
    · 能推导的项不问（给了观看距离就不问 P 值；给了 P 值就不问观看距离）

Phase 2 复问（一轮走完之后）
    · 只问还缺的硬性条件，并说明"为什么需要知道"
    · 第二次仍拿不到 → DEFERRED（推荐照常降级）；室内外拿不到 → BLOCKED
    · 软问题（使用场景 / 价位取向）不再补问

立即推荐（跳过 Phase 1）
    · 客户明确要推荐（recommend / 帮我推荐 / 选一款…）
    · 客户授权 AI 决定（you decide / 你决定 / 都行…）
    · 只是"把硬性条件凑齐"不触发推荐
```

实现位置：`src/dialogue/question_flow.py`（随机轮 / 复问状态机）、
`src/dialogue/question_order.py`（会话种子洗牌）、`src/agents/sales/nodes/requirement.py`（接线）。

### 0b. 分辨率口径（v2.5+）

```text
客户只要提了分辨率（4K / 1080P / 3840x2160…）→ 一律按"屏体大约要达到"处理
    · 不区分"4K 输入 / 4K 屏体 / 只说 4K"（INPUT / DISPLAY / UNKNOWN 只是记录）
    · 不再向客户提任何澄清问题（"要输入还是要屏体"这类问题已删除）
    · 屏体拼出来的像素数「达到或超过」目标即算达到（MEETS_OR_EXCEEDS），不要求一模一样

尺寸 + 点间距 → 实际可拼接分辨率 → 与目标比：
    够            → 正常推荐
    换更细 P 值能做 → 直接交给引擎在更细档位选型（不打断对话）
    连最细 P 值也不行 → 不推荐，直接告诉客户：
        "这个尺寸下（即使最细点间距）只能到 AxB，达不到你要的 4K；
         要 4K 的话屏幕需要做到 4.8m × 2.7m（标准尺寸），
         或者保持这个尺寸接受 AxB。"
客户没提分辨率 → 不做任何分辨率约束，按默认推荐

分辨率优先于"视距推点间距"（v2.5++）
    · 客户没锁死 P 值时：视距/环境推出来的点间距区间**只当倾向**，
      不再把"能拼到 4K 的更细点间距"一票否决
      （实测 bug：室内 5m → P3.0 以上，把 P0.7 能达标的型号整个挡掉）
    · 客户把 P 值定死了 → 不偷偷换，直接告诉他做不到 + 需要多细 / 多大

尺寸 "x × y" 不区分哪边是宽（v2.5++）
    · 客户说 "3*5" 时不再追问哪个是高哪个是宽，也不再只按一种摆法算
    · 两种摆法 × 箱体横拼/竖拼 一共四种几何都算，哪种能拼到客户要的分辨率就按哪种
      （实测：3m×5m 要 4K → 按 5m 作宽，P1.2 正好 3840×2160）
```

实现位置：`src/engineering/resolution.py`（`MEETS_OR_EXCEEDS`）、
`src/engineering/screen_geometry.py`（`size_orientations`）、
`src/engineering/feasibility.py`（`check_feasibility` / `check_model_feasibility`）、
`src/rag/recommendation_engine.py`（`resolution_driven_pitch`）、
`src/rag/recommendation_coordinator.py`（把结论话术与摆法透出给客户）。

### 0c. 硬性条件问过就不再问（v2.5++）

```text
客户回答过"室内/户外" → 档案里就是"客户确认"，任何轮次都不会再问一遍：
    · 关键词命中（indoor / 室内…）        → 客户明说
    · 语义模型读出且带客户原话证据        → 客户明说
    · 客户正在回答这一问（哪怕只有一个词）→ 直接采信（不再要求"证据片段"）
    · 客户已经被问过这一项、档案里已有值   → 一律不再重复问
客户从没说过、系统只是猜的 → 仍然会跟他确认一次（这条口径不变）
```

实现位置：`src/core/requirement_extractor.py`（`_merge_extractions` / Step 6）、
`src/rag/readiness.py`（`_environment_settled`）。

### 0d. 话术出口（v2.5++，僵硬话术优化）

```text
旧：业务逻辑 → 固定模板（ACK + 复述 + 过渡词 + 问题）→ LLM 润色 → 客户     （僵硬）
新：业务上下文 → LLM 原生生成 → Validator → 客户                          （默认）

Python 决定"说什么"：Action / 必须问的那一项 / 必须答的内容 / 推荐型号 / 工程结论
LLM   决定"怎么说"：要不要接话、要不要复述、句子长短、直接问还是先铺垫
Validator 只管边界：客户问题是否回答、有无编造参数/型号/价格/交期、
                    有无改动工程结论、是否超过一个问题、有无内部术语泄漏
不再强制：ACK、连接词、过渡句、复述客户

唯一客户文本出口：src/dialogue/response_generator.py
    Strategy B（默认）ResponseContext → LLM 原生生成（没有草稿）
    Strategy A（对比/回退）草稿 → 润色
    无 LLM / 生成不合格 → 结构化拼装 → 旧模板兜底（旧链路保留，未删除）

十项自然度指标（src/dialogue/response_validator.py）：
    generic_ack_rate / question_repeat_rate / connector_repeat_rate /
    customer_echo_rate / questionnaire_pattern_rate / customer_question_answer_rate /
    one_question_compliance / unsupported_fact_rate / internal_term_leak_rate /
    response_length

黄金数据集（行为约束，不是固定文案）：eval/naturalness_golden.json
A/B 对比：src/dialogue/ab_test.py::compare_strategies()
```

实现位置：`src/dialogue/response_context.py`、`response_generator.py`、
`response_validator.py`、`ab_test.py`、`src/agents/sales/nodes/script_generator.py`；
计划与逐项标注见 `LED_RAG_僵硬话术优化工程计划.md`。

### 1. 架构收敛（v2.3 / v2.3.1）

```text
UserTurn → Orchestrator（只编排：1208 行 → 564 行）
             ├── vision.pipeline        图片结果并进需求档案
             ├── rag.multi_screen       多屏拆分 / 切换 / 共享 / 逐屏推荐
             ├── Sales / Solution Agent（LangGraph，未改）
             ├── rag.recommendation_coordinator  推荐唯一出口
             └── dialogue.ResponseCoordinator    回复唯一出口
```

```text
RequirementProfile（唯一事实源）
    ├── 字段值 + sources
    ├── field_decisions（MISSING/CONFIRMED/INFERRED/UNKNOWN/DELEGATED/DECLINED/DEFERRED）
    ├── conflicts / conflict_slots
    └── FieldValue provenance（value / unit / source / status / confidence / formula_id）

推荐唯一出口（src/rag/recommendation_coordinator.py）：
    Conflict 检查 → Recommendation Gate → Engineering Derivation → Provenance Guard
        → RecommendationEngine → Validation → Final Recommendation + 决策审计日志

工程规则唯一来源：src/engineering/（constants / viewing_distance / pitch_window /
    screen_geometry / constraints / provenance / conflicts / requirement_classes）
```

- **Orchestrator 只做编排（v2.3.1）**：多屏业务在 `src/rag/multi_screen.py`、
  Vision 接入在 `src/vision/pipeline.py`、性能埋点在 `src/observability/perf.py`、
  回复组装在 `src/dialogue/response_coordinator.py`；旧方法保留同名薄包装（兼容层），
  边界由 `tests/test_v231_orchestrator_boundary.py` 锁死。
- **没有合法来源就不推荐**：点间距 / 室内外等关键参数必须能说清来源
  （customer / confirmed / derived / inferred / vision）；凭空出现的值 → `REJECTED`。
- **冲突独立状态**：如"屏比房间大""室内却要 P10" → `CONFLICT`，先澄清、不推荐。
- **统一 Validation**：LLM / Vision 输出先过字段、单位、范围、来源、覆盖权限校验才入档。
- **决策审计日志**：每条推荐记录 turn id / facts / decisions / 推导 / 约束 / Gate /
  候选 / 被拒原因 / 选中型号 / provenance / 校验结果（`[DecisionAudit]` JSON）。
- **对话层**：提问只有一个出口（Question Planner，自动换说法、不重复），
  表达结构由 Response Planner 决定（ACKNOWLEDGE → EXPLAIN_WHY → ASK_ONE_QUESTION），
  Conversation State 记住上一轮问了什么，推荐第一轮短、客户追问再展开。

这一节描述**系统现在实际怎么说话、怎么决策**。实现的唯一入口是
`src/rag/field_policy.py`（字段策略 + Action Planner）与 `src/rag/readiness.py`
（两条 Gate），LLM 只负责表达，不参与参数取值。

### 1. 需求字段的决策状态

| 状态 | 含义 | 后续动作 |
|---|---|---|
| `MISSING` | 从没问过 | 正常提问 |
| `CONFIRMED` | 客户明确提供 | 直接使用 |
| `INFERRED` | 系统推导（如"教堂"→室内） | 可用，但不算客户确认 |
| `UNKNOWN` | 客户说过"不知道" | **先换下一个问题**，最后一轮再问一次 |
| `DELEGATED` | 客户说"你决定 / 都行" | 不再问，交给 Python 确定性推导 |
| `DECLINED` | 客户明确不提供 | 不再问 |
| `DEFERRED` | 问满两次仍拿不到 | 不再问，只阻塞依赖它的动作 |

核心原则：**一个字段缺失不等于整个流程停下**；字段只阻塞"依赖它的 Action"
（例如尺寸延后只影响箱体计算，不影响选型）。

### 2. 问答顺序（软问题只在硬性条件没问完时顺带问）

```text
室内外 → 使用场景 → 固装租赁 → 价位取向(价格/质量/都行) → 点间距 → 观看距离 → 尺寸 → 推荐
```

| 类别 | 字段 | 规则 |
|---|---|---|
| 硬性条件 | 室内外、固装租赁、点间距、（反推点间距用的）观看距离、尺寸 | 只要还有没问完的，就按优先级问 |
| 软问题 | 使用场景、价位取向 | 只在硬性条件没问完时顺带问；**硬性条件一齐就立刻推荐，不再问别的** |
| 只记录 | 内容类型 | 不主动问（客户说了就记，不影响选型） |

### 3. 客户说"不知道"之后的节奏

```text
问 A → 客户"不知道" → 本轮不再问 A，换 B、C、D…
其它都问完、真要推荐时 → 回头用"降门槛"的问法再问 A 一次
第二次仍"不知道" → DEFERRED（绝不再问第三次）
客户后来主动补上 → 立即采纳（DEFERRED 不是客户确认过的值）
```

- 软问题（场景 / 价位取向）客户说不知道就**不再回头问**；
- "每字段最多两次"由提问记账保证，与"能不能听懂客户的话"解耦
  （客户拼错 `i dont konw`、回 `idk` / `dunno` 也照样识别）；
- 客户回答的归属由 Gate 回报的 `next_slot`（本轮实际问的槽位）决定，
  不会把答案记到错误的字段上（例如只回一个 "both"）。

### 4. 观看距离推导 + 点间距窗口

客户给的任何信息都折成**观看距离区间**，再映射到**点间距窗口**（不是按场景枚举规则）：

| 客户给的信息 | 折算方式 |
|---|---|
| 观看距离 | 直接用（最高可信） |
| 场地进深 | 最远 ≈ 进深 − 0.5m；最近 ≈ 1.5 × 屏高 |
| 场地面积 / 房间尺寸 | 进深 ≈ 面积 ÷ 屏宽（或 √(面积÷2)）；再按上一行 |
| 观众人数 | 每排座位 ≈ 屏宽 ÷ 0.6m；排数 = 人数 ÷ 每排；最远 ≈ 排数 × 0.9m + 2.5m |
| 只有屏尺寸 | 最近 ≈ 1.5 × 屏高，最远 ≈ 3 × 屏高 |
| 什么都没有 | 按环境的保守兜底档（室内 P2.5~P4） |

点间距窗口 = `[最远观看距离 ÷ 5, 最远观看距离 ÷ 1.5]`（再用最近观众收紧上限），
窗口内取接近业务首选值的型号；**并列时取偏粗的一端**（太细只是多花钱）。

实测（同一块 10×5m 室内固装墙屏）：

| 客户说法 | 距离区间 | 窗口 | 推荐 |
|---|---|---|---|
| 100 人 | 6.2 ~ 8.8m | P3.0 ~ P5.9 | TW11-3216-P3.0 |
| 50 人 | 4.3 ~ 6.1m | P3.0 ~ P4.1 | TW11-3216-P3.0 |
| 进深 8 米 | 5.3 ~ 7.5m | P3.0 ~ P5.0 | TW11-3216-P3.0 |
| 只说 10×5m | 7.5 ~ 15m | P3.0 ~ P7.5 | TW11-3216-P3.0 |
| 什么都不说 | 推不出来 | 兜底 P2.5 ~ P4.0 | TW11-3216-P3.0 |
| 25㎡ 房间 + 3m 宽屏 | 2.5 ~ 7.8m | P1.6 ~ P2.5 | TW11-3216-P2.5 |

### 5. 服务口径（客户问才答，且不自相矛盾）

| 客户问 | 回答口径 |
|---|---|
| 包不包安装 | 一般不提供现场安装（建议当地找安装公司更省成本），但每个订单都提供**安装指导说明书**，随货一并发给客户 |
| 有没有说明书 / 图纸 | 下单后随货一并发给客户，也可以提供安装图纸、说明书等技术资料 |
| 质保 | 默认 **1 年**，可付费延长；**客户不问就不提** |

回复里任何来源（模型润色 / 自由问答）如果出现 `installation is included`、
`we can provide on-site installation`、"包安装" 这类与口径矛盾的句子，会被
`strip_contradictory_installation_claims()` 删掉后再接标准回答；否定句
（`we do not provide on-site installation`）不会被误删，型号名也不会被切断。

### 6. 话术与语言口径

- 默认**英文**回复（`RESPONSE_LANGUAGE_POLICY`），中文提问也用英文答；
- **不主动提价格**、不主动提质保；报价由销售另行处理；
- 一次回复**只给一个型号**（箱体两种拼法都给：横拼 / 竖拼，含箱体数与实际尺寸）；
- 客户要"再推荐一个"时，按引擎排名依次给**还没给过**的下一款；
- 追问话术轮换、第二次问同一项时换成"降门槛"的说法；
- **绝不把内部字段名**（`distance` / `size` / `pixel_pitch`）抛给客户 ——
  未知槽位一律退回通用人话问句；
- 室内外是**不可绕过的硬阻塞**：无法判断时 BLOCKED，并说明"需要先确认室内还是室外"。

### 7. 版本历史（详细变更见下文「变更明细」）

| 版本 | 内容 |
|---|---|
| v2.5++ | **僵硬话术优化**：ResponseContext 纯结构化 + ResponseGenerator 成为唯一客户文本出口（LLM 原生生成，不再"模板 + 润色"）；Validator 只守事实边界（不强制 ACK / 连接词 / 复述）；10 项自然度指标 + 黄金数据集 + A/B 对比 |
| v2.5++ | 分辨率优先于视距点间距（客户没锁死 P 值时，视距区间只当倾向，不再挡掉能拼到 4K 的细点间距）；尺寸 "x×y" 不区分宽高，两种摆法 × 横拼/竖拼四种几何取最优；"室内/户外"答过就不再重复问（含"只答一个词"和语义模型带证据的情形） |
| v2.5+ | 分辨率口径调整：一律按"屏体大约能达到"处理（不再区分输入/屏体、不再提澄清问题）；达到或超过目标即算可行；连最细点间距也达不到时直接告诉客户需要的标准尺寸或降低分辨率，不再输出自相矛盾的提问 |
| v2.5 | 多条消息聚合成一个 UserTurn（**前端已接入**：连续发送先聚合再一次性请求；debounce / 幂等 / 会话锁）；DialogueAction + ResponseContext（话术不再像问卷，附 7 项话术指标）；分辨率需求（INPUT/DISPLAY/UNKNOWN）+ 实际拼接分辨率 + Resolution Fit + Engineering Feasibility（不可绕过） |
| v2.4 | 提问顺序随机化（会话种子、可复现）+ 问过/没答不再重复 + 一轮走完复问硬性条件（附"为什么需要知道"）+ 客户明确要推荐才立即推荐 |
| v2.3.1 | Orchestrator 职责收敛：多屏业务 / Vision 接入 / 性能埋点 / 回复组装迁出，Orchestrator 只编排（1208 → 564 行），旧接口保留兼容层 |
| v2.3 | 架构收敛：唯一事实源 + 统一工程规则 + Provenance Guard + 统一出口 + 统一 Validation + 冲突状态 + 决策审计 + 对话层（Response Planner / Conversation State） |
| v2.2.5 | 客户说"不知道"先换下一问、最后一轮再问；服务口径自相矛盾清洗 |
| v2.2.4 | 恢复使用场景 / 价位取向提问（硬性条件 vs 软问题分层）；Gate 回报 `next_slot` |
| v2.2.3 | 追问话术不再漏出内部字段名；已有场地事实不再问观看距离 |
| v2.2.2 | 面积 / 房间尺寸解析补全（场地 ≠ 屏体尺寸） |
| v2.2.1 | 拼写容错（`i dont konw` / `idk` / `dunno`）+ 问满上限守卫 |
| v2.2 | 观看距离确定性推导 + 点间距物理窗口；去掉"并列取最细点间距" |
| v2.1 | 客户决策状态（DELEGATED / UNKNOWN / DEFERRED / DECLINED）+ Field Policy |
| v2.0 | 确定性选型：两条 Gate + 确定性推荐引擎 + 工程计算 |

---

## 历史变更明细

> 早前版本的逐条变更（v2.0 ~ v2.9.x 的开发记录）已迁到 [`docs/history/CHANGELOG_archive.md`](docs/history/CHANGELOG_archive.md)。
> 版本历史以 Git 为准：`git log --oneline`。

---

## 技术栈

| 层级 | 技术选型 | 说明 |
|------|----------|------|
| **大语言模型** | DeepSeek Chat（`deepseek-chat`） | 通过 LangChain `ChatOpenAI` 封装调用 |
| **Embedding** | BGE-M3（`BAAI/bge-m3`） | HuggingFace 32 语言模型，同时输出 Dense + Sparse 向量 |
| **向量数据库** | ChromaDB `0.5.5` | 本地持久化存储，向量 + metadata |
| **Agent 框架** | LangGraph `0.2.56` | 状态机驱动的多 Agent 编排 |
| **API 服务** | FastAPI `0.115` + Uvicorn | RESTful 接口 |
| **稀疏检索** | BGE-M3 Sparse（FlagEmbedding `2.7.3`） | SPLADE 风格稀疏向量，替代传统 BM25 |
| **关键词检索** | BM25Okapi（`rank_bm25`） | 统计词频模型，与 M3 Sparse 同时启用 |
| **前端** | 原生 HTML/CSS/JS | 单页面聊天界面 |

---

## 系统架构

### 整体流程

```
用户输入
    │
    ▼
┌─────────────────────────────────┐
│      DualAgentOrchestrator       │
│         (协调器)                  │
└──────────────┬──────────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
┌──────────┐        ┌─────────────┐
│ Sales    │        │  Solution   │
│ Agent    │───────►│  Agent      │
│ (销售引导) │ 触发   │  (RAG 检索) │
└──────────┘        └──────┬──────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │    Hybrid Retrieval        │
              │  ┌─────────┐ ┌─────────┐  │
              │  │ Vector  │ │BGE-M3   │  │
              │  │ Search  │ │ Sparse  │  │
              │  └─────────┘ └─────────┘  │
              │  ┌────────────────────┐  │
              │  │ BM25 (备选)        │  │
              │  └────────────────────┘  │
              │            │               │
              │            ▼               │
              │   Reciprocal Rank Fusion   │
              │     (权重:向量1.0 /        │
              │      M3:1.5 / BM25:1.0)   │
              │            │               │
              │            ▼               │
              │      LLM Reranking         │
              │  + 环境冲突检测 + 类别过滤  │
              └──────────────────────────┘
                           │
                           ▼
                    推荐结果 + 自然语言回复
```

### Agent 节点图

#### Sales Agent（LangGraph 状态机）

```
classify (意图分类)
    │
    ├─ greeting ──────────────► script_generator ──► 回复
    ├─ need_query ──────────► requirement_mining ──►
    │                              │                 router ──► Solution Agent
    ├─ product_question ───────────────────────────────────────────► Solution Agent (直接)
    ├─ objection ──────────► script_generator ──► 回复
    ├─ industry ───────────► script_generator ──► 回复
    └─ closing ────────────► script_generator ──► 回复
```

#### Solution Agent（LangGraph 状态机）

```
intent_recognition (意图识别)
    │
    ├─ recommendation ──► understand ──► infer_parameters ──┐
    │                         │                              │
    │                    ┌────┴────────┐                    │
    │                    ▼             ▼                    │
    │               缺少参数 ──► clarify ──► understand ────┤
    │                    (追问澄清)           │              │
    │                                       ▼              │
    └──────────────────────────────────► retrieval ────────┤
                                                          │
                                                          ▼
                                                     recommend
                                                          │
                                                          ▼
                                               reflection (最多循环 3 轮)
                                                          │
                                                          ▼
                                               回复 + 推荐结果
```

---

## 检索流程（Hybrid Retrieval）

### 1. 参数推断（Parameter Inference）

输入：用户自然语言需求 → 输出：结构化过滤条件

| 参数 | 类型 | 说明 |
|------|------|------|
| `display_type` | `LED` / `LCD` / `IFP` | 产品类型 |
| `brightness_min` | int | 最低亮度（cd/m²） |
| `brightness_max` | int | 最高亮度（cd/m²） |
| `pitch_min` | float | 最小点间距（mm） |
| `pitch_max` | float | 最大点间距（mm） |
| `is_rental` | bool | 租赁场景标识 |

### 2. 三路并行检索

| 通道 | 模型 | 权重 | 说明 |
|------|------|------|------|
| 向量检索 | BGE-M3 Dense | 1.0 | 语义相似度（display_type 已知时跳过） |
| M3 稀疏检索 | BGE-M3 Sparse（`lexical_weights`） | 1.5 | SPLADE 风格 token 激活权重，效果优于 BM25 |
| BM25 检索 | BM25Okapi | 1.0 | 传统词频/逆文档频率统计模型 |

### 3. 融合排序（RRF）

$$
\text{RRF}(d) = \sum_i \frac{w_i}{k + \text{rank}_i(d)}
$$

- `k = 60`
- 三路得分加权求和，按最终分数排序

### 4. LLM 重排（Rerank）

- 过滤非显示屏产品（category != `display`）
- 环境冲突检测（室内产品用于室外场景）
- 内部数据泄露过滤
- 最终选取 Top 3 推荐

---

## 三层路由（Three-Layer Routing）

| 路由类型 | 触发条件 | 处理方式 |
|----------|----------|----------|
| **Fast Path** | 纯参数查询、问候语、保修/异议等固定场景 | 结构化过滤 + 模板回复，**无需 LLM 调用**；输出为 **Model 级** |
| **Normal Path** | 简单场景需求（户外广告屏、会议室P2.5等） | 轻量 RAG + 快速推荐 |
| **Agent Path** | 复杂推理需求（"怎么选"、"哪个合适"、多轮对话） | 完整 LangGraph Agent + Reflection 质量评估 |

---

## 记忆系统（Memory）

### EnhancedMemoryStore — 三层记忆架构

```
┌─────────────────────────────────────────────┐
│              Session Memory                  │
├─────────────────────────────────────────────┤
│  Short-term Memory (短期记忆)                 │
│  - 最近 50 条对话消息                        │
│  - FIFO 自动截断                             │
├─────────────────────────────────────────────┤
│  Structured Profile (结构化客户画像)         │
│  - 预算、场景、屏幕类型、室内/室外            │
│  - 租赁/固装、感兴趣/拒绝产品                │
│  - 待确认问题、已提出异议                    │
├─────────────────────────────────────────────┤
│  Conversation Summary (长期摘要)              │
│  - 自动压缩历史对话                          │
│  - LLM 生成关键要点摘要                      │
│  - 不覆盖结构化事实                          │
└─────────────────────────────────────────────┘
```

> ⚠️ **注意**：当前 Memory 为进程内存储（`OrderedDict`），服务重启后会话数据丢失。SQLite 持久化功能正在规划中。

---

## 目录结构

```
led-rag-system/
├── .env                              # 环境变量（API Key、路径等）
├── .github/workflows/ci.yml          # CI 自动测试
├── requirements.txt                   # Python 依赖
├── pytest.ini                        # Pytest 配置
├── init_vectorstore.py               # 向量库初始化脚本
│
├── data/                             # 产品数据源
│   ├── LED display.txt               # LED 屏原始规格数据
│   ├── led_products.json             # LED 产品结构化数据（ProductFilter 用）
│   ├── LCD display.txt               # LCD / IFP 原始规格数据（同一份资料）
│   ├── lcd_products.json             # LCD 结构化数据（21 个型号）
│   ├── ifp_products.json             # IFP 结构化数据（Omni G4/G4C、K4/K4C，10 个型号）
│   ├── company_profile.txt           # 公司与销售人员信息配置
│   ├── first_contact/                # 首次客户接待素材目录
│   │   ├── README.md               # 素材配置说明
│   │   ├── case_video_01.mp4      # 案例视频 1
│   │   └── case_video_02.mp4      # 案例视频 2
│   └── buisness_card/              # 名片图片目录
│       └── business_card.png       # 销售人员名片
│   └── led_rag_system.db             # (规划中) SQLite 持久化数据
│
├── models/                           # 本地模型文件
│   └── BAAI--bge-m3/snapshots/master/  # BGE-M3 Embedding 模型
│
├── src/
│   ├── api.py                        # FastAPI 端点（/chat, /health, /memory 等）
│   ├── config.py                     # 配置管理（.env 加载）
│   ├── orchestrator.py               # 双 Agent 协调器
│   │
│   ├── agents/
│   │   ├── sales/                   # 销售 Agent（销售引导 + 需求挖掘）
│   │   │   ├── graph.py            # LangGraph 状态机定义
│   │   │   ├── runner.py           # Agent 入口
│   │   │   ├── state.py            # 状态 schema
│   │   │   └── nodes/
│   │   │       ├── classify.py     # 意图分类（greeting / need_query / objection 等）
│   │   │       ├── requirement.py  # 需求挖掘 + LLM + 规则推断
│   │   │       ├── router.py       # 触发 Solution Agent
│   │   │       └── script_generator.py  # 话术生成（所有意图）
│   │   │
│   │   └── solution/               # 方案 Agent（RAG 检索 + 推荐）
│   │       ├── graph.py           # LangGraph 状态机定义
│   │       ├── runner.py          # Agent 入口（支持 Fast/Agent 双路径）
│   │       ├── state.py           # 状态 schema
│   │       └── nodes/
│   │           ├── intent.py      # 意图识别（recommendation / conversation 等）
│   │           ├── requirement.py # 需求理解 + 参数推断 + 追问澄清
│   │           ├── retrieval.py   # 混合检索调用
│   │           ├── recommend.py  # 产品推荐（LLM 生成自然语言推荐）
│   │           ├── reflection.py # 质量评估（Reflection Quality Gate）
│   │           └── others.py      # 通用自由问答
│   │
│   ├── core/
│   │   ├── embeddings.py           # BGE-M3 Embedding + ChromaDB 向量库初始化
│   │   ├── llm.py                 # DeepSeek LLM 封装 + 降级机制
│   │   └── fallback.py            # LLM 降级 + 熔断器（Circuit Breaker）
│   │
│   ├── rag/                        # RAG 检索管线
│   │   ├── loader.py              # 产品文档加载 + metadata 提取
│   │   ├── chunker.py             # 文档分块（RecursiveCharacterTextSplitter）
│   │   ├── vector_store.py        # ChromaDB 封装
│   │   ├── retriever.py           # 向量相似度检索
│   │   ├── sparse.py              # BGE-M3 Sparse 检索（SPLADE）
│   │   ├── bm25.py                # BM25 检索
│   │   ├── fusion.py              # RRF 混合融合
│   │   ├── rerank.py              # LLM 重排 + 冲突检测 + 类别过滤
│   │   ├── parameter_inference.py # LLM+规则 参数推断
│   │   ├── router.py              # 三层路由（Fast/Normal/Agent Path）
│   │   ├── fast_path.py           # Fast Path 结构化过滤 + 模板回复
│   │   └── json_loader.py         # JSON 产品数据加载（ProductFilter）
│   │
│   ├── models/
│   │   └── product.py             # Pydantic 产品数据 Schema（LED/LCD/IFP）
│   │
│   ├── memory/
│   │   ├── store.py               # MemoryStore 单例（Session FIFO，50 条上限）
│   │   └── enhanced.py            # EnhancedMemoryStore（三层记忆架构）
│   │
│   ├── tools/                     # Agent 可用工具
│   │   ├── search_tool.py         # 混合检索工具（供 Agent 调用）
│   │   ├── product_tool.py        # 产品信息查询
│   │   └── customer_tool.py       # 客户信息查询
│   │
│   ├── tasks/                     # 异步任务管理
│   │   └── __init__.py            # TaskManager（向量库重建等异步任务）
│   │
│   ├── observability/             # 可观测性
│   │   └── __init__.py           # 性能追踪 + 可选 Langfuse 集成
│   │
│   └── utils/
│       ├── ifp_intent.py         # IFP 交互平板意图检测（确定性规则）
│       ├── message.py            # 消息格式化
│       └── text.py               # 文本处理
│
├── first_contact/                  # ✅ 已完成：首次客户固定工作流
│   ├── __init__.py               # 首次接待模块入口
│   ├── handler.py                # 首次接待流程控制器
│   ├── profile.py                # 销售人员/公司信息加载与解析
│   └── assets.py                 # 案例视频/产品手册素材配置
│
├── static/                        # Web UI
│   ├── index.html                 # 聊天界面
│   ├── style.css                  # 样式
│   └── app.js                    # 聊天逻辑（移除产品卡片渲染后）
│
├── vectorstore/                    # ChromaDB 持久化数据
│   └── chroma.sqlite3
│
├── eval/                          # 评估体系
│   ├── golden_dataset.json        # Golden Dataset（82 条标注用例）
│   ├── metrics.py                 # 公共指标（Recall@K / MRR / Slot Accuracy …）
│   ├── retrieval_eval.py          # 检索评测（Recall@K / MRR / 硬约束违规）
│   ├── recommendation_eval.py     # 需求理解 / 路由 / 推荐引擎评测
│   ├── calculator_eval.py         # 箱体 / 模组计算评测
│   ├── dialogue/                  # 对话链路行为约束评测（12 场景）
│   └── reports/                   # 评测输出目录（跑一次才生成，已 gitignore）
│
├── conftest.py                    # 测试环境变量 + fixtures（放在 rootdir）
│
└── tests/                         # 全量测试（约 591 条，约 1.5 分钟）
    ├── dialogue/                  # 对话决策（承接 / 重复提问 / 单一回复 / Turn Action / others 语境）
    ├── input/                     # 输入层 Turn 引擎（去重 / 聚合 / 会话锁 / 幂等 / 生成期间补发）
    ├── memory/ replay/ vision/    # 会话记忆窗口 / 真实日志 Replay / 图片载荷
    ├── test_session_reset.py      # 会话内需求重置
    ├── test_service_faq*.py       # 服务口径（安装 / 说明书 / 质保 / 交期）
    ├── test_vision_pipeline.py    # 视觉链路 smoke（图片确认 / Gate / 前端入口）
    ├── test_screen_calculator.py  # 工程计算（箱体 / 模组）
    └── test_api_http.py           # HTTP 端到端
```

---

## 数据模型

### Chunk Metadata（存入向量库）

> v2.0 起语料为 **Model 级**（一个实际销售型号一个 Document），Series 作为 `series_id` metadata 保留。

| 字段 | 类型 | 说明 |
|------|------|------|
| `model` | str | 完整型号，如 `TW21-3216-P2.5` |
| `series_id` | str | 产品族，如 `TW21-3216` |
| `level` | `model` | 语料层级标记 |
| `display_type` | `LED` / `LCD` / `IFP` | 产品类型 |
| `indoor` | bool | 室内适用 |
| `outdoor` | bool | 室外适用 |
| `installation` | `fixed` / `rental` | 安装方式 |
| `brightness_min_cd` | int | 最低亮度（cd/m²） |
| `brightness_max_cd` | int | 最高亮度（cd/m²） |
| `pixel_pitch_min_mm` | float | 最小点间距（mm） |
| `pixel_pitch_max_mm` | float | 最大点间距（mm） |
| `is_rental` | bool | 租赁场景 |
| `modules_per_cabinet` | int | 每箱模组数 |
| `gob` / `flexible` | bool | GOB 灌胶 / 柔性屏功能标记 |
| `product_category` | `display` / `mount` / `module` / `software` | 产品类别 |
| `environment_metadata_version` | int | metadata 版本（当前 v4） |

### CustomerProfile（结构化记忆）

| 字段 | 类型 | 说明 |
|------|------|------|
| `budget` | str | 客户预算 |
| `scene` | str | 应用场景 |
| `display_type` | str | 屏幕类型（LED/LCD/IFP） |
| `environment` | str | 室内/室外 |
| `is_rental` | bool | 租赁/固装 |
| `interested_products` | list | 感兴趣的产品 |
| `rejected_products` | list | 拒绝的产品 |
| `pending_questions` | list | 待确认问题 |
| `stated_objections` | list | 已提出异议 |
| `interaction_count` | int | 交互次数 |

---

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/chat` | 主聊天接口，支持 SSE 流式输出 |
| `GET` | `/health` | 服务健康检查 |
| `GET` | `/diagnostics/retrieval` | 向量库覆盖率诊断报告 |
| `GET` | `/memory/{session_id}` | 获取指定会话的历史消息（需鉴权） |
| `POST` | `/memory/clear` | 清除指定会话内存（需鉴权） |
| `POST` | `/rebuild` | 重建向量库（需鉴权） |

**`/chat` SSE 流式模式**：
请求头 `Accept: text/event-stream` 启用流式输出，每条 SSE 事件格式：
```
data: {"type": "start", "session_id": "..."}
data: {"type": "ttft", "ms": 123.4}
data: {"type": "chunk", "content": "推荐"}
data: {"type": "done", "total_ms": 567.8}
```

**API 鉴权**：
请求头 `X-API-Key: <your-key>` 鉴权（`.env` 中配置 `LED_API_KEY`，空值 = 开发模式不鉴权）

---

## 核心配置项（.env）

```env
DEEPSEEK_API_KEY=         # DeepSeek API Key（必填）
MODEL_NAME=deepseek-chat   # 模型名称（默认 deepseek-chat）

HOST=0.0.0.0             # 服务监听地址
PORT=8000                 # 服务监听端口

VECTORSTORE_DIR=./vectorstore        # 向量库存储路径
DATA_DIR=./data                      # 产品数据文件路径
TOP_K=5                              # 默认检索结果数量

# ── Reflection 预算控制 ──
REFLECTION_MAX_ROUNDS=3              # 最大反射轮数（默认 3）
REFLECTION_MAX_TOKENS=2000           # 单轮最大 token 估算
REFLECTION_MAX_TIME_MS=8000          # 单轮最大耗时（毫秒）
REFLECTION_NO_IMPROVE_STOP=2         # 连续 N 轮无改进终止

# ── 安全与可观测性 ──
LED_API_KEY=                         # API 鉴权密钥（空 = 开发模式不鉴权）
LLM_TIMEOUT_SECS=30                  # LLM 单次调用超时（秒）
LLM_MAX_RETRIES=2                   # LLM 失败自动重试次数

# ── v2.0：回复语言策略 ──
RESPONSE_LANGUAGE_POLICY=en          # en=始终英语（默认）；auto=跟随客户语言
```

---

## 评测结果

在 **82 条 Golden Dataset**（`eval/golden_dataset.json`）上的一次实测快照
（报告文件不再入库，随时可用下面的命令重跑）：

| 指标 | 结果 |
|------|------|
| Requirement Slot Accuracy | 99.6% |
| Top-1 推荐准确率 | 92.7% |
| Top-3 推荐覆盖 | 100% |
| 点间距区间命中率@3 | 100% |
| 硬约束违规率 | 0.0% |
| 计算准确率（14 用例） | 100% |
| 检索 Series Recall@5 | 95.0% |
| 检索 Model Recall@10 | 91.9% |

> 检索指标为候选集质量；最终推荐由确定性引擎收敛（Top-1/Top-3）。
> 复现：`python -m eval.recommendation_eval` / `python -m eval.retrieval_eval` /
> `python -m eval.calculator_eval`，详见 [eval/README.md](eval/README.md)。

---

## 实施进度总览

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 0 | Golden Dataset + 性能基线 | ✅ 完成（部分） |
| Phase 1 | 三层路由（Fast/Normal/Agent） | ✅ 完成 |
| Phase 2 | 消除 Sales/Solution 重复推理 | ✅ 完成 |
| Phase 3 | Fast Path 强化（结构化过滤 + 模板） | ✅ 完成 |
| Phase 4 | 产品数据结构化（ProductFilter） | ✅ 完成 |
| Phase 5 | Agent Path 精简（Graph 瘦身） | ✅ 完成 |
| Phase 6 | Reflection → Quality Gate | ✅ 完成 |
| Phase 7 | 混合 RAG（Dense + Sparse + BM25 + RRF） | ✅ 完成 |
| Phase 8 | 参数推断规则（LLM + 规则混合） | ✅ 完成 |
| Phase 9 | Orchestrator 简化 | ✅ 完成 |
| Phase 10 | Memory 分层（requirements / history 分离） | ✅ 完成 |
| Phase 11 | 性能优化（Embedding/向量库缓存） | ✅ 完成 |
| Phase 12 | 可观测性（PerfTracker + 可选 Langfuse） | ✅ 完成 |
| Phase 13 | 测试套件 | ✅ 完成（660 条通过，4 条按需跳过：非中英文场景需 LLM） |
| Phase 14 | Memory 持久化（SQLite） | 🚧 规划中 |
| Phase 15 | 首次客户固定工作流（First Contact） | ✅ 完成 |

> 之后又完成了一轮 **v2.0 工程化升级**（详见上方「最近更新」与
> `LED_RAG_推荐与智能选型工程化优化计划_v2.0.md`）：推荐就绪 Gate、计算就绪 Gate、
> 确定性推荐引擎、确定性校验、Model 级语料、多语言 Query Understanding、LLM 分工收紧，
> 以及**会话内需求重置**（客户拿到推荐后换产品 / 换项目时清空需求并重新采集）。
>
> 再之后完成了 **全量需求捕获 + Unknown 容错**（详见上方「最近更新（全量需求捕获 + Unknown 容错）」
> 与 `LED RAG 智能销售系统：全量需求捕获与 Unknown 容错优化实施计划.md`）：
> 字段级状态机、最多问两次、客户不知道也能 Best-effort 推荐并说明缺失项。
> 最新测试基线：**759 passed, 4 skipped**。

---

## 首次客户固定工作流（First Contact）

首次客户接待模块为首次联系客户提供标准化的固定流程，包括自我介绍、案例视频发送、名片发送等环节。

### 核心功能

| 功能 | 说明 |
|------|------|
| **自我介绍生成** | 基于 `company_profile.txt` 使用 LLM 生成自然语言自我介绍 |
| **案例视频发送** | 依次发送案例视频（支持文件不存在时跳过） |
| **名片发送** | 自动发送销售人员名片图片 |
| **过渡语生成** | 素材发送后的自然过渡话术 |
| **名片话术生成** | 名片发送后的礼貌请求话术 |
| **英语强制模式** | 所有回复强制使用英语，无论客户语言 |

### 语言策略

- **核心原则**：首次接待的所有回复强制使用英语，不受客户语言影响
- **实现方式**：Prompt 中明确指定 "ALWAYS use English"
- **降级方案**：LLM 调用失败时使用基于 Profile 的模板降级回复

### 配置说明

#### 1. 公司信息配置（`data/company_profile.txt`）

```txt
# ── 销售人员信息 ──
Sales Name: Mike

# ── 公司信息 ──
Company: iSEMC
Location: Shenzhen, China

Company description:
iSEMC was founded in 2013.
LEMA Technology is a high-tech enterprise specializing
in video processing and display products development...

# ── 主营产品 ──
Main Products:
Indoor LED Display
Outdoor LED Display
Rental LED Display

# ── 公司优势 ──
Company Advantages:
12 years of international service experience
10 years of production and manufacturing capacity
```

#### 2. 素材配置（`src/first_contact/assets.py`）

```python
ASSETS: list[FirstContactAsset] = [
    FirstContactAsset(
        name="Case Video 1",
        filename="case_video_01.mp4",
        asset_type="video",
        description="Check out our latest project case",
    ),
    FirstContactAsset(
        name="Case Video 2",
        filename="case_video_02.mp4",
        asset_type="video",
        description="Here is another real application demo",
    ),
]
```

#### 3. 素材文件放置

将视频/PDF 文件放入 `data/first_contact/` 目录：
- `data/first_contact/case_video_01.mp4` — 案例视频 1
- `data/first_contact/case_video_02.mp4` — 案例视频 2
- `data/first_contact/product_catalog_01.pdf` — 产品手册 1（可选）

名片图片放入 `data/buisness_card/` 目录：
- `data/buisness_card/business_card.png` — 名片图片

### 使用方式

```python
from src.first_contact.handler import first_contact_handler

# 执行首次接待
result = first_contact_handler.run(
    session_id="session-001",
    customer_message="Hi, I want to know about your LED displays"
)

# 发送消息到前端
for msg in result.to_messages():
    await send_to_client(msg)

# 继续进入 Sales Agent
if result.should_continue:
    await sales_agent.handle(customer_message)
```

### 测试覆盖

| 测试类 | 测试内容 |
|--------|----------|
| `TestCompanyProfile` | Profile 加载与解析测试 |
| `TestProfileParsing` | 文本解析边界测试 |
| `TestFirstContactHandler` | Handler 核心流程测试 |
| `TestFirstContactAssets` | 素材配置与路径解析测试 |
| `TestLanguageStrategy` | 语言策略强制英语测试 |

---

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY

# 3. 初始化向量库（首次运行）
python init_vectorstore.py

# 4. 启动服务
python -m uvicorn src.api:app --reload --port 8000

# 5. 打开浏览器
# http://localhost:8000
```
