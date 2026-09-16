# LED RAG 智能销售系统

> 基于大模型（DeepSeek）的 LED/LCD/IFP 全品类显示产品智能销售助手，采用多 Agent 协作 + 混合检索（RAG）架构，为销售团队提供实时产品推荐和技术咨询能力。

---

## 项目状态

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
| Model 级产品数据与语料（49 Model / 9 Series） | ✅ 已完成 |
| 推荐就绪 Gate（Recommendation Ready Gate） | ✅ 已完成 |
| 计算就绪 Gate（Calculation Ready Gate） | ✅ 已完成 |
| 确定性推荐引擎（Recommendation Engine） | ✅ 已完成 |
| 确定性校验（替代 LLM Reflection 重新决策） | ✅ 已完成 |
| 箱体 / 模组工程计算（Screen Calculator） | ✅ 已完成 |
| 多语言 Query Understanding + Query Rewrite | ✅ 已完成 |
| **首次客户固定工作流（First Contact）** | ✅ 已完成 |
| **全量需求捕获 + Unknown 容错（最多问两次）** | ✅ 已完成 |
| **智谱视觉需求提取（客户发图片识别需求）** | ✅ 已完成 |
| **Memory 持久化（SQLite）** | 🚧 规划中 |

---

## 最近更新（v2.0 工程化升级，2026-09-13/14）

本阶段把系统从"RAG 找产品 + LLM 做推荐"升级为**确定性选型 + 工程计算**，核心是两条 Gate 和一套"客户明确 vs 系统推断"的溯源机制。

### 新的推荐主链路

```text
客户消息
   ↓
Sales Agent（只问关键问题，一次一个）
   ↓
Requirement Profile（结构化需求 + 来源标记）
   ↓
Recommendation Ready Gate ── 不满足 → 继续询问（绝不推荐）
   ↓ 满足
技术参数推理（纯 Python，LLM 不参与）
   ↓
Recommendation Engine（硬过滤 + 确定性评分 → 唯一最佳型号）
   ↓
RAG（只提供产品事实证据，不决定选型）
   ↓
Calculation Ready Gate ── 无屏体尺寸 → 只推荐、暂不计算
   ↓ 有尺寸
Screen Calculator（箱体 / 模组 / 实际尺寸 / 分辨率）
   ↓
确定性 Reflection（校验，不重新决策）
   ↓
LLM 最终销售话术（全程仅 1 次）
```

### 推荐何时才发生（Recommendation Ready Gate）

只有满足以下任一条件才会推荐，否则**先问，绝不猜**：

1. 客户点名**型号 / 系列**；
2. 客户明确给出**点间距（P2.5）或亮度（600nit）**；
3. **环境 + 场景 + 安装方式 + 观看距离** 四项齐备，且安装方式与观看距离必须来自**客户原话**。

系统只认"客户说过的事实"（`confirmed`）打开 Gate；规则估算、场景默认、LLM 自行补全的字段一律标记为 `inferred`，只能用于打分提示，不能作为推荐依据。

环境的唯一例外：**会议室 / 教室 / 教堂 / 展厅 / 机场… 这类场景本身就等于室内，户外广告 / 体育场等于室外**，系统直接落定环境、不再追问"室内还是室外"；舞台 / 演唱会 / 租赁这类室内外都可能，仍然要问。

### 职责边界（v2.0 核心原则）

| 角色 | 允许做 | 不允许做 |
|---|---|---|
| LLM | 理解场景、提取特殊要求、生成最终销售话术 | 推断/补全工程参数，决定推荐哪个型号 |
| Python 规则 | 参数推理、硬过滤、评分、箱体模组计算、校验 | 无 |
| RAG | 提供产品事实与证据 | 决定最终选型 |

### 本轮新增 / 重构模块

| 文件 | 作用 |
|---|---|
| `src/rag/readiness.py` | Recommendation Ready Gate + Calculation Ready Gate |
| `src/rag/recommendation_engine.py` | 确定性 Model 级选型（硬过滤 + 5 维加权评分） |
| `src/rag/validation.py` | 10 项确定性校验（替换 LLM Reflection 重新决策） |
| `src/rag/corpus.py` | 统一构建 Model 级检索语料 |
| `src/rag/query_understanding.py` | 多语言 Query Understanding + Query Rewrite + 溯源合并 |
| `src/rag/hard_filter.py` | 硬约束元数据过滤（环境/安装/型号/功能） |
| `src/tools/screen_calculator.py` | 箱体 / 模组工程计算 |
| `src/models/requirement.py` | RequirementProfile（来源三态 + 置信度） |
| `src/agents/sales/question_planner.py` | 需求采集顺序（一次只问一个高价值问题） |
| `src/rag/session_switch.py` | 会话内需求重置检测（换产品 / 换项目 / 改需求） |
| `src/rag/reply_composer.py` | "先回应客户 + 再追问需求"的组合回复（不增加 LLM 调用） |
| `src/models/legacy_adapter.py` | RequirementProfile → 旧字段的**只读**兼容层（单向投影） |
| `src/rag/company_info.py` | 公司 / 办事处 / 地址类提问的照实回答（读 `company_profile.txt`） |
| `src/core/requirement_extractor.py` | 统一需求理解入口（规则 + 短语快速路径 + LLM 语义 + 证据校验 + 冲突） |
| `src/core/purpose_normalizer.py` | 场景 → 22 个 canonical purpose token |
| `src/core/environment_installation_resolver.py` | 环境/安装方式统一解析（优先级 + 冲突检测） |

### 行为变化（相对旧版）

- **Fast Path 改为 Model 级输出**：不再把 `TW31-COB series` 当推荐结果，返回真实型号（如 `TW31-COB-P0.7H`）。
- **SSE 与普通 `/chat` 统一**：都走完整 `orchestrator.process_message` 管线，不再绕过 Sales Agent 与需求采集。
- **回复语言策略**：新增 `.env` 配置 `RESPONSE_LANGUAGE_POLICY`（`en`=始终英语，默认；`auto`=跟随客户语言）。
- **销售侧 LLM 只提取 `usage` + `additional_requirements`**：工程参数全部由确定性解析器从客户原话提取。
- **追问话术不再固定**：同一待问项提供多种自然说法（`src/rag/readiness.py::QUESTION_VARIANTS`），按"会话 + 轮次"轮换，**问到的内容完全一致、措辞不重复**（不额外消耗 LLM）。
- **推荐话术不再固定**：LLM 指令改为"措辞与句式随每次回复变化"，离线模板也准备了多种开头轮换。
- **缺尺寸必追问、有尺寸立即算**：Calculation Gate 未就绪时一定追问屏体尺寸（确定性兜底，不依赖 LLM 是否听话）；客户给出尺寸后当轮直接计算箱体/模组/实际尺寸。
- **尺寸单位自动换算**：支持 `5m x 3m` / `5000x3000mm` / `500cm x 300cm` / `16ft x 9ft` / `10 m by 6 m` / `120 x 90`（无单位按量级推断），只写一侧单位时自动继承；`5米宽`、`3米高` 也支持单独识别。
- **同一会话里可以换产品 / 换项目**：客户在拿到推荐之后说"我想换个产品 / 重新推荐 / 其实是户外的 / 另外还有一块屏"，系统会**清空这次咨询的需求档案并重新采集**，不再拿旧需求（旧场景、旧视距）接着推荐；重置后的第一句是口语确认（多种说法轮换），随后回到"一次只问一个关键问题"的采集流程。
- **需求重置有明确边界**：尺寸 / 点间距 / 亮度的单独修改只更新对应槽位，不清空重来；"还有别的型号吗 / 再推荐几款"属于"想看更多选项"，也不会清空需求。
- **回复"先回应、再追问"，不再只会反问**：客户问"有没有 P1.2 COB LED"会先拿产品数据**核实后**正面回答（`Yes — we do carry a 1.2 mm COB LED.`），再问还缺的那一项；客户问别的事（如"印尼有你们的人吗"）会先给出答复，再把需求问题接在后面。核实不到规格时如实说"我帮您确认一下"，绝不编造。
- **开场白 / 自我介绍 / 要规格要报价都会被回应**：需求抽取那一次 LLM 调用同时产出一句口语回应（`ack`，不额外增加调用与延迟），例如客户说"My name is Ar Majeed Akbar from Pakistan … university … need quotation"时，回复是 `Nice to meet you, Ar Majeed — happy to help with the smart classroom project.` **再**接需求问题。回应优先级：产品数据核实结果 > LLM 回应 > 规则复述 > 中性兜底，任何情况下都不会只丢一个问题给客户。
- **客户抢答 / 自己先说出的需求会被采纳**：观看距离支持英制单位（`100 feet` / `100ft` / `15 ft away` / `40 inches` / `yards`，统一换算成米，例如 `100 feet → 30.48 m`）——之前英制不识别，客户明明说了视距，系统还是一直追问，而且回复变成"已确认 100 英尺视距 + 再问视距"的自相矛盾。
- **回复绝不自相矛盾**：拼装回复时，如果"先回应"的内容已经确认了接下来要追问的那一项，就换成中性回应（复述也会跳开这一项），保证不会出现"刚确认完又问同一件事"。
- **回应语言跟策略一致**：需求抽取那一次调用会带上语言要求（默认策略下要求纯英文），并且中文回应在英文策略下会被直接丢弃 —— 之前出现过 `Sello 你好，很高兴认识你。Will it be an indoor or outdoor setup?` 这种中英混排。
- **短期记忆上限 20 → 50**：每会话保留的对话消息从最近 20 条提升到最近 50 条（`src/memory/store.py::MAX_MEMORY_SIZE`、`src/memory/enhanced.py::MAX_SHORT_TERM_MESSAGES`），多轮选型对话不会过早丢掉前文；各节点的 LLM 提示仍只取最近 3~6 条，不会因为窗口变大而增加 token 开销。
- **一眼室内/室外的场景直接落定环境**：会议室 / 教室 / 门店 / 展厅 / 博物馆 / 大厅 / 指挥中心 / 办公室 / 医院 / 银行 / 酒店 / 餐厅 / 机场 / 展会 / 教堂 → 室内，户外广告 / 体育场馆 → 室外，不再追问"室内还是室外"；舞台 / 演唱会 / 租赁这类室内外都可能，仍会问。
- **需求没问清时问价格 → 先说明报价规则，紧接着继续问需求**：客户在需求采集阶段问 `What is the price...` / `多少钱` 时，回复是"价格要按具体型号和箱体配置来算，确认好产品后马上报价" + 当前那个待问项（一次只问一个），不再答非所问或直接去推荐。
- **公司 / 办事处 / 经销商 / 地址类提问 → 严格按 `data/company_profile.txt` 照实回答**：这类问题会回答"我们公司是 iSEMC，位于 Shenzhen, China，这是唯一的所在地"（+ 继续问需求），不会凭空说"有当地代表"；也不会再被"有没有某规格"的核实逻辑抢答。
- **追问话术更口语化、连续多轮不重复**：安装方式的问法扩到英文 8 句 / 中文 6 句（"Quick one — fixed install or rental?"、"Just so I quote the right setup…"），轮换步长按"客户说过第几句话"推进，连续 8 轮不会出现同一句；**回答完客户的问题（公司 / 价格 / 规格核实）之后再追问会带自然过渡**（"Meanwhile —"、"By the way —"、"On that note —"），不再像书面条款那样硬接问句。
- **三套需求状态统一为一套（Profile 是唯一真相）**：`RequirementProfile` 成为唯一主状态 —— 每轮只做"加载已持久化的 Profile + merge 本轮消息"，旧 `requirements` 由 `src/models/legacy_adapter.py` **单向投影**生成（不再反向重建、不会漂移、线索字段也不会"复活"）；Solution 直接消费 Sales 传下来的 Profile，不再自己解析对话重建需求。
- **来源四态**：`confirmed`（客户明说）/ `scenario_derived`（会议室、教堂、户外广告、体育场等由客户原话场景直接判定）/ `default`（系统默认，如场景默认固装）/ `inferred`（算法估算）。前三者可用于打开 Ready Gate，`default` 只参与打分，`inferred` 永远不能当客户确认。
- **推荐只有一道闸门**：全项目里 `should_generate_solution` 只能由 Ready Gate 置为 `True`；Gate 抛异常时显式安全降级（不推荐）；`required_met` / `required_missing` 降级为 Gate 结果的投影，旧的 `should_trigger_solution()` / `REQUIRED_KEYS` 判定已删除。
- **需求理解收敛到一个入口（关键词依赖优化）**：Sales / Solution 都改成调用 `RequirementExtractor` —— 规则解析负责确定性事实与数字，规范短语表（canonical phrase）作为快速路径，LLM 只补语义并**必须给出客户原话证据**（防止替客户猜参数）；同一轮只在 Sales 调一次 LLM，Solution 复用缓存结果。场景从"必须命中关键词"升级为"语义归一化到 22 个 canonical purpose"。
- **冲突可检测、可阻断**：同一轮出现两个互相矛盾的明确信号（例如规则说室内、语义说室外）会记录到 `RequirementProfile.conflicts`，Ready Gate 直接阻断推荐并要求澄清；而"客户明确说租赁 + 会议室场景"这类正常业务不再被误判为冲突（客户明确事实优先）。
- **点间距按"环境 + 观看距离"选（客户口径，2026-09-15）**：
  - **室外**：≤4m → **P4**（客户 4m 就推 P4）｜4–5m → P4 或 P5｜**6–20m → P5**（这个区间 P5 最合适）｜20–25m → P6.67｜**25–30m → P8**｜**超过 30m → 一律 P10**；半户外按室外口径。
  - **室内**：**≤3m → P2.5 及以下**（越近越细）｜**>3m → P3 及以上**（首选 P3，如 `TW11-3216-P3.0`）。
  - 客户**明确点名点间距**时一律以客户为准（连场景偏好都不再参与打分）。
  - 实现：`parameter_inference.preferred_pitch_for_environment()` 给出区间与首选值，`recommendation_engine` 在有业务首选值时把 pitch 权重提高 2.5 倍（保证规则压过场景/预算等软偏好），并在 `_violation` 里把该区间当硬约束执行。
- **客户只报一个长度时会先确认方向**：客户回 `129,2cm` / `1292 mm` / `51 inch` 这类裸尺寸时，系统记为"尺寸线索"并追问"这是宽度、高度还是对角线？"（把客户给的数字填进问题里）；客户回"宽度/高度"即落成对应尺寸，"对角线"则回到问宽高 —— 全程不替客户猜。
- **一句里给出两个尺寸也能同时接住**：支持"数字 + 单位 + 方向词"的两种语序（`45cm is the width` / `width is 45cm` / `长1.29米，宽0.45米`），并把客户口中的"长/长边(length)"理解为水平方向的"宽"——当一句话里同时出现"长 + 宽"时，长边落成宽、另一条边落成高（`129,2cm us the length and 45xm is width` → 129.2 cm 宽 × 45 cm 高）。`45xm` 这类把 `cm` 打成 `xm` 的笔误也会纠正。
- **报需求的话不会被当成"闲聊问题"**：客户回答室内外 / 安装方式 / 视距 / 尺寸（如 `129,2cm`、`about 100 feet`）时，一律留在需求采集流程；之前会被判成 `others` 走自由问答，在 Gate 未通过时就把 indoor/outdoor 各系列型号一股脑倒出来。
- **追问话术池扩大**：每个待问项的英文/中文说法从 3~4 句扩到 5~7 句，并按"客户说过第几句话"逐轮轮换（之前按消息条数跳两步，问法轮换不完整），问到的内容始终一致。

### 已修复的关键问题

- 之前"知道室内外 + 场景就直接推荐"的根因：规则层/LLM 会凭空估算观看距离，恰好凑齐 Gate 条件；已彻底移除"人数/面积→视距"估算，并禁止 LLM 补全工程参数。
- 场景回答（如 `church`）被 `classify` 误判成 `others`，从而绕到 Solution 自由问答检索、绕过 Gate；已补场景关键词并在识别到 `usage`/`purpose` 时强制纠正为 `need_query`。
- 客户回答裸数值（`5m` / `about 5m` / `5米`）时观看距离无法解析、导致系统反复追问；已增加"裸数值 + 单位"兜底解析并排除面积/尺寸误判。
- 跨轮"推断标记"残留导致客户明说"固定安装"仍被当作没说过；已用 `merge_slots()` 修正溯源。
- `POST /rebuild` 之前会把向量库从 49 条翻倍成 98 条（追加而非重建）；已改用 `recreate_vectorstore`。
- `POST /rebuild` 返回的 task_id 之前永远查不到状态（404）；已让任务管理器接受显式 task_id。
- 同一会话里客户想换产品时，旧需求（室内 / 教堂 / 5 米）会被继续沿用到下一次推荐；现在由 `src/rag/session_switch.py` 纯规则识别"换产品 / 换项目 / 需求冲突 / 换屏幕类型"，清空需求档案后重新采集，并顺手修掉 `different` 里的 `rent` 被误判成"租赁"的整词匹配问题。
- 客户问产品 / 业务问题时，销售只会回一句反问、答完就断线（"只会问问题"）；现在由 `src/rag/reply_composer.py` 统一拼成"先回应 + 再追问"，Sales 与 Solution 两条路径共用同一套逻辑。
- 英文关键词用子串匹配导致的误判：客户名字 `Akbar` 里的 `bar` 被当成"餐厅场景"，系统回了 `Alright, a restaurant.`；`different` 里的 `rent` 被当成"租赁需求"。现在英文关键词按**整词（可复数）**匹配，且用"前后不是英文字母/数字"界定边界（`\b` 对中文无效，会让 `会议室用LCD` 匹配不到）。
- 需求回写会用 `or 0` 把缺失的一维补成 `0`，产生 `size='9.144米x0米'` 的脏数据；现在只写客户真正给过的那一维（`9.144米宽`）。
- 观看距离解析只认公制（米/m），客户说 `about 100 feet viewing distance` 时判定为"没说"；于是回复先确认 100 英尺视距、紧接着又追问观看距离。现在支持英制并换算（`100 feet → 30.48 m`），客户自己说出的需求当轮即被采纳。
- 客户回答裸尺寸 `129,2cm` 时：既没解析出尺寸，又被 `classify` 判成 `others` → 绕到自由问答，Gate 未通过就把 `TW31-COB-P0.9H / TW31-HOD-P5.7E …` 等型号全列了出来。现在裸尺寸会变成"尺寸线索 + 方向确认追问"，报需求的句子也不会再走自由问答。
- 客户答 `129,2cm us the length and 45xm is width`（一句里两个尺寸）时：两个数字一个都没解析到，系统把刚问过的方向问题又问了一遍，回复还自相矛盾（先复述尺寸、再问这个尺寸是什么）。现在两个尺寸会分别落成宽 129.2 cm / 高 45 cm，且 `size_axis` 也纳入"不让回应与追问打架"的护栏。
- 尺寸被误判成观看距离：`宽度 1.29m` 曾解析出"视距 1.29 m"，而十进制视距还会被从小数点后面重新匹配（`1.29米 → 29 米`）。两处都已修正。
- 客户说了 `It for church` 这种"一眼室内"的场景，系统仍在追问"室内还是室外"；现在这类场景直接落定环境，只对真正有歧义的场景（舞台 / 演唱会 / 租赁）发问。
- 客户在采集需求阶段问价格时，`script_generator` 的异议分支还留着旧判定 `should_trigger or requirements.get("usage")` —— 只要需求里已有场景就直接 `trigger_solution`，于是客户听到 "Sure, let me find the right products for you..."（实际 0 个产品），价格没答、需求也没继续问。现在推荐只由 Ready Gate 决定，该分支改为"报价规则 + 继续问需求"；同时删掉了 `_infer_from_message()` 那套"人数 → 面积 → 视距"的估算代码（过早推荐的老根因，已无调用方）。
- `trigger_solution` 但产品数为 0 时也会被标记成"已推荐"，会让下一轮"换个产品"被误判；现在只有真的给出产品才标记。
- 客户问 "Do you have a representative in western India" 时，`availability_answer` 只凭句子里有 LED 就回了 `Yes — we do carry an LED.`：既答非所问、又凭空说 Yes。现在公司类问题一律交给 `src/rag/company_info.py` 按公司信息照实回答；"有没有某规格"也必须**带具体规格**（点间距 / COB / HDR / 防水 / 型号）才回答，泛问不再乱说 Yes。
- 公司信息此前完全没被问答链路使用（连 `company_profile.txt` 里的 `Location: Shenzhen, China` 都没解析）——现在 `CompanyProfile` 增加 `location`、兼容 `Company description` 键，并新增公司信息问答模块。
- **语言识别把英文判成德语**：`_LANGUAGE_HINTS` 的德语提示里混进了英文单词 `display`，任何含 "display" 的英文句子都被判成 `de`（影响回复语言策略与多语言关键词表）。现在提示词只保留该语言特有词。
- **英文词形变化识别不到**：`permanently` / `banking` / `wall mounted` 都匹配不上关键词（当时只支持可选复数 s）。现在支持 `s/es/d/ed/ing/ly` 词尾。
- 中文视距 `观众大概 6 米远` 解析不到（单位后要求空白/标点）；`open-air` 不在室外关键词里；`retail`（商场）在 EnvironmentResolver 里被当成"室内外都可能"，与 `query_understanding` 不一致。三处均已修正。
- 换成"室内首选 P3.0"后暴露一个误报：型号名里的 `P3.0` 与数据里的实际点间距 `3.076mm` 对不上，被确定性校验判成"虚构参数"（扣 1.5 分）。现在厂商命名里的 P 值也算真实数据。

---

## 最近更新（全量需求捕获 + Unknown 容错，2026-09-16）

本阶段解决的问题（客户实测反馈）：

1. **AI 问 A、客户回答 B，B 信息被丢掉** —— 例如 AI 问"观看距离"，客户回"5m x 3m"，
   尺寸没有被记录。
2. **客户答"我不知道"，AI 仍反复问同一句** —— 甚至出现同一个问题被连问 12 次。

### 核心机制

```text
客户消息
   ↓
全量需求提取（不按"当前问题"过滤字段）
   ↓
RequirementProfile（字段级状态机）
   missing → unknown_pending → unknown → confirmed
   ↓
字段级提问次数（Python 强制，最多 2 次）
   第 1 次：正常问
   第 2 次：降低门槛（给区间 / 二选一）
   仍不知道：标记 unknown → 直接跳过，绝不追问第 3 次
   ↓
Recommendation Ready Gate
   READY          信息齐备 → 正常推荐
   CONTINUE_ASKING 还有可问字段 → 继续问
   DEGRADED_READY  只剩 unknown → 按已确认信息 Best-effort 推荐
   ↓
推荐话术 = 推荐产品 + 当前依据 + 缺少什么 + 可能影响什么
```

### 关键文件

| 文件 | 作用 |
|------|------|
| `src/core/unknown_detector.py` | 纯正则识别客户"不知道 / 跳过某一项"（中英文） |
| `src/models/requirement.py` | 字段级状态机：`ask_counts` / `unknown_reasons` / `slot_status()` / `requirement_basis()` |
| `src/rag/readiness.py` | Gate 三态 + 第二次追问的"降门槛"问法 |
| `src/rag/recommendation_service.py` | `RECOMMENDED` / `DEGRADED` + 推荐依据（confirmed / inferred / unknown） |
| `src/rag/reply_composer.py` | `degraded_note()`：缺什么 + 影响什么（多套自然说法） |
| `src/agents/solution/nodes/recommend.py` | Best-effort 推荐 + 话术确定性兜底 |

### 行为变化

| 场景 | 旧行为 | 新行为 |
|------|--------|--------|
| 客户答非所问（问视距、答尺寸） | 尺寸丢失 | 尺寸入库，视距继续问（最多 2 次） |
| 客户说"不知道" | 同一问题无限重复 | 第二次换"给区间"的问法，之后跳过 |
| 客户两次都不知道 | 卡在需求采集 | 转 `DEGRADED_READY`，按已有信息推荐并说明缺失项 |
| 客户后续补上信息 | - | `unknown → confirmed`，并停止追问该项 |
| 视距未知但尺寸已知 | 不计算箱体 | 照常计算箱体 / 模组 |

### 测试

```bash
python -m pytest tests/test_unknown_tolerance.py -q   # 59 passed
python -m pytest tests -q                             # 759 passed, 4 skipped
```

> 计划文档与逐阶段标注：`LED RAG 智能销售系统：全量需求捕获与 Unknown 容错优化实施计划.md`

---

## 修复：跨会话串台（AI 用别的对话框的内容聊天，2026-09-16）

### 现象

不重启服务时，**刷新页面 / 开新标签页 / 点"新对话"**，新对话里 AI 仍然会带上别的对话框
的需求（例如新会话说 "5m"，AI 却按"教堂 / 室内"来答）；只有重启服务才恢复正常。

### 根因（两条，互相叠加）

1. **进程级语义缓存没有按会话隔离**（主因）

   `src/core/requirement_extractor.py` 里的 `_semantic_cache` 是类属性（进程级），
   缓存键只用**消息文字**：`"5m" → {...}`。
   但语义理解是 LLM 结合"当前对话上下文 + 已收集需求"得出的 ——
   同一条 `"5m"` 在 A 对话框里是"教堂室内"，在 B 对话框里可能什么都不是。
   于是新会话说出同名消息时直接命中旧会话的缓存，凭空继承 `purpose=church` /
   `environment=indoor`，且**只有重启进程才会消失**（正好对应"不停止服务器就一直串"）。

2. **前端所有标签页共用一个 session_id**

   `static/app.js` 用 `localStorage` 存 session_id：刷新、开新标签页、开新窗口
   都会拿回**同一个会话**；而界面又不会加载历史，看起来是"新对话"，
   实际后端还是同一个会话在继续。

### 修复

| 位置 | 改动 |
|------|------|
| `src/core/requirement_extractor.py` | 语义缓存键改为 `session_id::message`；**不传 session_id 就不缓存、不复用**；新增 `clear_session_semantics()` |
| `src/agents/sales/nodes/requirement.py` | 抽取时传入 `state["session_id"]` |
| `src/agents/solution/{state,runner}.py`、`nodes/requirement.py` | Solution Agent 增加 `session_id`，跨 Agent 复用语义结果时也按会话隔离 |
| `src/orchestrator.py` | 调用 Solution Agent 时传入 `session_id` |
| `src/agents/sales/runner.py`、`src/api.py` | 会话内需求重置 / `/memory/clear` 时一并清掉该会话的语义缓存 |
| `src/memory/store.py` | 会话数上限 `MAX_SESSIONS=200`（LRU 淘汰），长时间运行不再无限堆积 |
| `static/app.js` | session_id 改用 `sessionStorage`（**每个标签页一个会话**）；刷新后调用 `/memory/{id}` 把本会话历史渲染回来，界面与服务端状态一致；"新对话"同时清掉旧会话 |

### 行为变化

| 场景 | 旧行为 | 新行为 |
|------|--------|--------|
| 开新标签页 | 与已有标签页共用同一个会话 | 全新会话，互不影响 |
| 刷新当前页 | 界面空白，但后端仍是旧会话（看着像"AI 还记得别的对话"） | 恢复本标签页的历史，界面与后端一致 |
| 不同对话框说同一句话（如 "5m"） | 沿用别的会话的语义理解 | 各自独立理解 |
| 长时间不重启 | 会话与缓存无限增长 | LRU 控制在 200 个会话内 |

> 测试：`tests/test_session_isolation.py`（9 个用例，含 Sales 节点级端到端复现）。

---

## 最近更新（智谱视觉需求提取：客户发图片也能采集需求，2026-09-16）

客户可以把图片（截图 / 现场照片 / 图纸）和文字一起发过来，系统用**智谱视觉模型**
从图片里提取 LED 需求，合并进同一份 `RequirementProfile`，再照常继续问缺失项。

### 链路

```text
客户消息（文字 + 图片）
   ↓
文字：RequirementExtractor
图片：Zhipu Vision（glm-4v-plus）→ Vision Extractor（结构化 + 单位标准化）
   ↓
合并进同一个 RequirementProfile（客户明说 > 图片明确可见 > 场景判定 > 图片推测）
   ↓
Sales Agent / Question Planner（只问仍然缺的）
   ↓
Recommendation Ready Gate → Solution Agent → RAG / Calculator
```

### 三条硬规则

1. **区分"看见"和"猜测"**：`vision_explicit`（图片明确可见）与
   `vision_inferred`（模型推测）分开记录。想看但没看出来的项目一律返回 `null`，
   模型没按格式返回时**默认按"推测"处理**。
2. **图片不能顶掉客户**：客户已经明确说过的信息，图片永远不会覆盖；
   两者矛盾时保留客户的值、记录冲突，并**就问冲突的那一项**。
3. **图片尺寸不进入工程计算**：图片给的宽高只写入 `vision_size_hint_mm`，
   用来追问"图片上看大约 5m × 3m，是这样吗？"；`target_width_m / target_height_m`
   永远只由客户确认的尺寸写入，所以箱体/模组计算不可能用到图片猜的尺寸。

### 关键文件

| 文件 | 作用 |
|------|------|
| `src/vision/client.py` | 智谱视觉客户端（超时 / 重试 / 模型回退 / MIME+大小检查） |
| `src/vision/extractor.py` | 图片 → 结构化需求（JSON 容错解析、单位换算、图片哈希缓存） |
| `src/vision/integration.py` | 合并进 RequirementProfile（优先级 / 冲突 / 尺寸 hint / 指标） |
| `src/vision/schema.py` / `prompts.py` | 数据结构与提示词 |
| `tests/test_vision.py` / `tests/test_vision_pipeline.py` | 63 个测试（单元 + 端到端） |
| `tests/vision_golden/` | Golden Dataset（生成图片 + manifest + 评测脚本 + 报告） |

### 使用方式

```bash
# .env
GLM_API_KEY=你的智谱key
GLM_VISION_MODEL=glm-4v-plus      # 可选
VISION_ENABLED=true                # false = 完全忽略图片
VISION_MAX_IMAGES=3                # 单次最多几张图
```

```bash
# 文字 + 图片（base64）
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{
  "session_id": "s1",
  "question": "Can you recommend something like this?",
  "images": [{"data": "<base64>", "mime_type": "image/jpeg"}]
}'

# 跑 Golden Dataset（需要真实 API）
python tests/vision_golden/run_golden.py
```

页面上传图片有三种方式（最多 3 张，发送后照常对话）：

1. 输入框左边的 **「图片」按钮**（点开选文件）
2. **直接粘贴**（Ctrl+V，页面任意位置都行，不要求输入框聚焦；
   截图、复制的图片文件、以及从网页复制的图片链接都能接住）
3. **把图片拖进页面**，松手即添加

添加后会显示缩略图预览，可单独删除；纯图片（不写字）也能发送。

### 实测与限制

- 接口实测通过（`glm-4v-plus`）；Golden 合成图基线：屏类型 100%、室内外 87.5%、
  场景 100%、"没屏幕就不填"100%、尺寸编造 0%、图片尺寸进工程计算 0。
- ⚠️ Golden 目前是**合成图**，只验证链路与"是否会瞎编"；真实精度需要业务侧提供
  脱敏真实照片后再测。

---

## 修复：客户回答 "close"（近）却被当成结束对话（2026-09-16）

### 现象（客户实测日志）

对话走到"观看距离"这一问：

```text
AI：Will viewers be fairly close to the screen, or more than about 10 metres away?
客户：close
AI：Alright, I'll put together a detailed proposal and quote for you...
```

既不推荐产品，也不问屏幕尺寸。

### 根因（两个叠加）

1. **意图被误判**：客户说的 `close` 意思是"近"，但意图分类器判成了 `closing`（要结束对话）。
   而 `requirement_mining` 里有一条 `closing → should_generate_solution = False`，
   于是**Gate 刚刚放行的推荐被否掉**，只回了收尾话术。
2. **"close / near / far" 没被解析成观看距离**：规则解析器只认 "5m"、"about 8 meters" 这类
   带数字的说法，模糊回答一个都没接住，所以观看距离一直是空的。

### 修复

| 位置 | 改动 |
|------|------|
| `src/rag/query_understanding.py` | 新增模糊回答与区间解析：`close→3m`、`very close→2m`、`far→15m`、`5-10 metres→7.5m`、`more than 10m→15m`、`within 3m→1.8m`；并用上下文/短句约束避免 `close the deal` 之类误判 |
| `src/agents/sales/nodes/classify.py` | 只有 `bye / that's all / 不用了 / 再见` 这类**明确结束语**才判 `closing`；短回答（"close"、"5m"、"permanent"、室内/室外…）一律按需求回答处理 |
| `src/agents/sales/nodes/requirement.py` | `closing` 不再无条件吞掉推荐：**Gate 已就绪 + 客户没有明确结束** → 照常推荐（意图纠回 `need_query`，`next_action=router`） |

### 修复后的同一段对话（真实 DeepSeek + 智谱重放）

```text
第4轮 客户：close
档案：viewing_distance_m = 3.0  （close = 近）
路由：trigger_solution，推荐数 1
回复：For your viewing distance, the TW11-3216-P2.5 is the right fit ... 
      What target screen width and height do you need?
```

→ 既推荐了产品（室内 3m 按业务规则首选 P2.5），也继续追问屏幕尺寸。

> 回归测试：`tests/test_requirement_dialogue.py::TestCloseAnswerIsNotClosing`
> 与 `TestReplayOfReportedConversation`（复刻这段日志）。

---

## 修复 + 优化：推荐之后要"接得住" + 代理商问题的回答（2026-09-16）

### 问题 1：推荐完产品后，客户问什么，AI 都又推荐一遍

实测日志：已经推荐过产品后，客户问 **"你们在肯尼亚有代理商吗？"**，
系统却又走了一遍推荐（"For your viewing distance, the TW11-3216-P3.0 is the right fit…"），
客户的问题完全没被回答。

根因（三处叠加）：

1. **销售侧判"这句话是不是在报需求"时用了历史状态**：
   `has_scenario = 历史 requirements 里已有 usage 或 本轮有 purpose` ——
   场景一旦确定，"历史里已有 usage"永远为真，于是之后**每一句话**都被当成"在报需求"，
   被判 `need_query` → Gate 已就绪 → 又推荐一遍。
2. **方案侧意图规则**：只要句子里检测到任何需求槽位（例如出现"屏 / LED"）就直接判
   `recommendation` —— 客户问"这个屏大概多久能发货？"也被当成"要推荐"。
3. **runner 每轮 `clear()` 会话时把"已推荐"标记清掉了**（只有本轮又出产品才重新写入），
   所以从第 3 轮起标记丢失，又开始重复推荐。

修复：

| 位置 | 改动 |
|------|------|
| `src/agents/sales/nodes/requirement.py` | 只看**本轮这句话**有没有需求信息（规则槽位 + 本轮 LLM 语义 + 场景关键词）；提问句（？/吗/怎么/有没有…）不再被改判成需求回答；**但**"自我介绍+说场景+要规格"这种混合句仍留在需求采集 |
| 同上 | **已推荐过 + 本轮没有新需求 + 不是"再推荐/报价/下单"** → 先回答客户，不重复推荐（走 others / product_question 回答路径） |
| `src/rag/parameter_inference.py` | 提问句不再因为出现"屏/LED"被判成推荐请求；"你们能推荐一款吗？"这类**明确要推荐**仍判 recommendation |
| `src/agents/sales/runner.py` + `src/memory/store.py` | "已推荐"标记跨轮保留（`get_recommendation()` → 重新写入） |
| `src/agents/sales/state.py` | 新增 `already_recommended` 状态字段 |

行为：需求有变化（例如"改成室外的"）或客户明确要"再推荐/报价/下单" → 照旧重新推荐；
其它情况一律先回答客户。

### 问题 2：问"有没有代理商"时的回答

回答固定要说清三件事（按客户口径）：

1. **只有中国（深圳）这一个公司/工厂**，当地没有代理、经销商或办事处；
2. **工厂是我们自己的**，中间没有环节 → 开销/成本更低、报价更有竞争力；
3. 海外客户由**深圳团队直接对接**。

中英文各 4 套说法按轮次轮换（**不是一句话死板重复**），事实写进
`data/company_profile.txt`（已补充"自有工厂""唯一所在地"两条），
`company_answer()` 只依据公司资料回答，不编造。

### 实测（真实 DeepSeek + 智谱，四轮）

```text
第1轮 I need an indoor fixed LED screen for a conference room, 3m x 5m, viewing distance 5 meters
      → 推荐 TW11-3216-P3.0（55 箱体 / 330 模组 / 3.2m x 5.28m）
第2轮 你们在肯尼亚有代理商吗？
      → 不再推荐；答：只有深圳这一个点、没有当地代理、自有工厂、中间无加价、成本更低
第3轮 那付款方式和交期怎么算？
      → 不再推荐；如实回答付款方式与交期（没有确切数据就直说）
第4轮 这个屏大概多久能发货？
      → 不再推荐；就是论事回答发货时间
```

> 回归测试：`tests/test_requirement_dialogue.py::TestAfterRecommendationMustAnswer`（7 条）
> 与 `tests/test_company_info.py::TestAgentDistributorAnswerContent`（4 条）。

---

## 新增：图片识别结果先跟客户确认（2026-09-16）

客户口径：**识别完图片不要直接当事实用，先把"我看到什么"说给客户听，让他确认；
他说得不对就按他说的改，并把客户正确的需求记下来。**

### 行为

```text
客户发图（+ 一句话）
   ↓ 智谱视觉提取 → 合并进 RequirementProfile（来源 vision_explicit）
回复：先接住客户这句话 + "我在图片里看到的是 XXX，对吗？" + 继续问缺失项
   ↓ 客户回应（下一轮）
   ├─ "对 / 是的"     → 这些字段升级为**客户确认**（sources: vision_explicit → confirmed）
   ├─ "不对，是室外的" → 以客户说的为准（Extractor 已按"客户优先"合并），
   │                     并记录纠正：image said indoor, customer said outdoor (environment)
   └─ 没回应/问别的    → 不再重复追问同一件事（核对只做一次）
```

### 实现

| 位置 | 改动 |
|------|------|
| `src/models/requirement.py` | 新增 `vision_confirmation_pending`（待确认字段）、`vision_assertions`（图片当时的判断）、`vision_corrections`（客户纠正记录） |
| `src/vision/integration.py` | `apply_vision_to_profile()` 登记待确认字段；新增 `resolve_vision_confirmation()` 落地"确认 / 纠正 / 跳过" |
| `src/rag/reply_composer.py` | 新增 `vision_confirmation_sentence()`：把识别结果转成一句人话（中英文各 4 套说法轮换，不死板），并支持插到"回应之后、追问之前" |
| `src/agents/sales/nodes/script_generator.py` | 带图那一轮把确认句加进回复（包括"直接给出推荐"的那一轮） |
| `src/agents/sales/nodes/requirement.py` | 下一轮先落地客户对图片识别的确认/纠正，再走原来的需求采集 |
| `src/agents/sales/runner.py` / `state.py` | 新增 `has_vision` / `vision_applied`，让节点知道"这一轮是不是带图" |

### 实测（真实智谱 + DeepSeek）

```text
第1轮 客户：[图片] i need a display like this
      → "Thanks for the photo — it looks like an LED screen, indoor use and a conference room,
         so correct me if I've misread it. So, fixed installation or rental — which one is it for you?"
      档案：display_type=LED、environment=indoor、purpose=conference（待确认 3 项）

第2轮 客户：no, actually it is outdoor, for a shop
      → "Got it, thanks for clarifying. Is this a permanent install, or is it for rental/events?"
      档案：environment=outdoor（客户）、purpose=retail（客户），更正记录：
           image said indoor, customer said outdoor (environment)
           image said conference, customer said retail (purpose)
```

> 话术衔接：确认句是**一句话**（含"说错请纠正"），后面用 `So, / Then, / Now, / Also, /`
> `那么，/ 这样的话，/` 之类的过渡词接上追问（完整问句、英文首字母小写），
> 并且当 LLM 的"回应"只是客套（"Got it — happy to help…"）时会被丢掉，避免三句话各说各的。

> 回归测试：`tests/test_vision_pipeline.py::TestVisionConfirmation`（6 条）。

---

## 修复：客户问"还有其他推荐吗"时的回答格式 + 不再提价格（2026-09-16）

### 问题（客户实测）

客户问 **"你还有其他的推荐吗？"**，系统把**同一款**又讲了一遍（参数 + 价格档位），
然后追问尺寸：

> For your viewing distance, the TW11-3216-P3.0 is a solid fit: 3.076mm pixel pitch, 500nit
> brightness… **It also sits in our low price tier** with a 1 year warranty. If you want higher
> brightness or longer coverage, TW21-3216-P3.0 and TW11-3216-P4.0 are options.
> What target screen width and height should I use…?

### 现在（客户口径）

```text
客户：你还有其他的推荐吗？
回复：…TW11-3216-P3.0 fits well: 3.076mm pitch suits that range, each 640mm*480mm cabinet holds 6 modules.
      If you want higher brightness, TW21-3216-P3.0.
      If you want a wider 4mm pitch, TW11-3216-P4.0.
      Share your target screen size, brightness, delivery or installation needs and I'll lock in the right model.
```

规则：

1. **备选说成条件句**：`If you want <差别>, <型号>.`（差别是程序比出来的：更高亮度 / 更细或更宽的点间距 /
   COB / HDR / 防水 / 更长质保 / 租赁版本…），不再把型号堆成一句 "…are options"。
2. **结尾邀请补充需求**："Share your target screen size, brightness, delivery or installation needs
   and I'll lock in the right model."（这一轮**不再单独催尺寸**）。
3. **绝不提价格**：产品数据里已经去掉 `price tier`，并加了硬规则"NEVER mention price, price tier,
   cost, discount, budget or value for money"；其它分支（others 节点）也从"直接比价"改成
   "价格取决于最终型号与配置，销售团队会出正式报价"。
4. **客户补充需求后重新锁定一款**：新需求（含预算档、亮度、点间距等）算作"新增需求"，
   会按"已有需求 + 新需求"重新推荐，并在话术里先接住他刚说的那句
   （如 "On quality and quick delivery, this model is a solid, readily available choice."）。

### 顺带修掉的一个更严重的问题

同一条日志里，客户中途问别的事时，Solution Agent 会**只拿这一句话重建需求**，
于是又回头问 "Are we talking about an indoor or an outdoor install?"（明明早就知道室内 / 会议室 / 10 米）。

修复：Orchestrator 在 `others` / `product_question` 这两条路由上，把
**本会话已收集的需求档案**和 **Sales 定好的意图**一起传给 Solution Agent；
Solution 的意图节点也不再自己重判意图（只在没有上游意图时才判）。

> 回归测试：`tests/test_sizing_and_expression.py::TestAlternativesReplyFormat`、
> `tests/test_reply_composer.py::TestOrchestratorComposesAnswerAndQuestion::test_others_route_carries_profile_and_intent`。

---

## 新增：交付时间口径（2026-09-16）

客户口径：

1. **问交期** → 从**下单付款**开始计算，常规交付时间约 **15–30 天**；
2. **要求加快** → 可以走**空运**，能提前交付，但**成本会增加**（会在报价里体现）；
3. **提到安装档期**（"想 11 月安装"、"need it by December"）→ 先接住他的话，
   再把交期说清楚，让他自己判断来不来得及（**不承诺具体日期**）。

### 实测（真实 DeepSeek）

```text
客户：i wanna buy a display, and wanna install 11月
→ Got it, 11月 is your target for installation. From order and payment our delivery usually takes
  about 15–30 days, so we'd want the order confirmed in good time.
  In the meantime, is this for indoor or outdoor use?

客户：大概多久能发货？
→ Our standard lead time is roughly 15–30 days from order and payment; I'll pin down the dates
  once we settle the configuration. While we're at it, most installations are indoors.
  Will this one be indoors, or outside?

客户：我们比较急，能不能加快？
→ For a faster schedule we'd switch to air freight: it costs more, but it brings the delivery date
  forward — I can price that option for you. And so I can point you to the right one,
  could you tell me the main use case?
```

注意：答复之后**照常继续问还缺的需求**（不会只回答完就断线），也不会因为客户提到档期
就跳过需求采集。

### 实现

| 文件 | 作用 |
|------|------|
| `src/rag/delivery_info.py` | 交期/加急/档期识别 + 中英文多套说法（15–30 天、空运加成本） |
| `src/agents/sales/nodes/script_generator.py` | 命中交付类话题时直接按口径回答（+ 接上待问问题），且不重复叠加 LLM 客套 |
| `src/agents/solution/nodes/others.py` | 自由问答分支也按同一口径回答交期（不承诺具体日期） |

> 回归测试：`tests/test_delivery_info.py`（24 条）。

---

## 变更：先问点间距，客户不知道再问观看距离（2026-09-16）

客户口径：**先问客户对点间距有没有要求；客户不知道，就转问观看距离，用规则反推点间距。**

### 询问顺序

```text
室内外 → 使用场景 → 安装方式
      ↓
① 点间距（先问）："Do you already know the pitch you want — P3, P4, P5 …?"
      ├─ 客户给出 P 值     → 按客户要的 P 值选型（直接放行推荐，不再问观看距离）
      └─ 客户不知道        → **只问一次**就跳过，转 ②
                             （客户说"你决定 / 随便 / 越清晰越好"也算不知道）
② 观看距离
      ├─ 客户给出距离      → 用规则推点间距（室内 ≤3m→P2.5、>3m→P3；室外 4m→P4、
      │                      6–20m→P5、25–30m→P8、>30m→P10），然后推荐
      └─ 客户也不知道      → 保持老规则：降门槛再问一次 → 仍不知道 → unknown
                              → 降级推荐 + 说明缺失项
```

关键规则：

- **点间距是"单次询问"**：客户答"不知道"就跳过（`SINGLE_ASK_SLOTS`），不会问第二遍；
  观看距离仍然是"最多问两遍"。
- **观看距离已经有了就不再问点间距**（距离能推 P 值），所以老会话不会被这个新问题拦下来。
- 客户给出的 P 值（`P3` / `p2.5` / `3mm pitch` / `点间距3mm`）都会解析进
  `pixel_pitch_mm`，并作为"客户明确规格"直接放行推荐。

### 实测（真实 DeepSeek，四轮）

```text
客户：I need an indoor LED screen for a conference room
→ …is this a fixed install or a rental?

客户：fixed installation
→ Understood — a fixed installation it is. Do you already know the pitch you want — P3, P4, P5 …?

客户：I don't know
→ No problem at all — pitch is something we can work out together once we know the room and viewing
  setup. How far will the audience typically be sitting from the screen?
   （点间距只问了 1 次：ask_counts={'pixel_pitch': 1}）

客户：about 10 meters
→ …TW11-3216-P3.0 is a solid fit: its 3.076mm pitch suits that range…（室内 10m → P3）
  If you want higher brightness, TW21-3216-P3.0. … What target width and height should I use…?
```

### 改动位置

| 文件 | 改动 |
|------|------|
| `src/rag/readiness.py` | `MISSING_ORDER` 把 `pixel_pitch` 插到 `viewing_distance` 之前；新增点间距的问法 / 降门槛问法 / 缺失标签；Gate 只在"点间距和观看距离都不知道"时先问点间距 |
| `src/models/requirement.py` | 新增 `SINGLE_ASK_SLOTS = {pixel_pitch}`：客户答"不知道"即跳过，不问第二遍 |
| `src/core/unknown_detector.py` | "你决定 / 随便 / 越清晰越好 / up to you" → 按"不知道"处理 |
| `src/rag/query_understanding.py` | 点间距解析补 `3mm pitch` / `间距3mm` 这类写法 |
| `src/agents/sales/question_planner.py` | 采集顺序同步插入点间距 |

> 回归测试：`tests/test_requirement_dialogue.py::TestPitchAskedBeforeDistance`（7 条）。

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
│   ├── golden_dataset.json        # Golden Dataset（72 条标注用例）
│   ├── dataset.json               # 路由/评测数据集
│   ├── reports/                   # 评测报告输出目录
│   └── run_eval.py                # 评测脚本
│
└── tests/                         # 测试套件
    ├── conftest.py                # Pytest fixtures
    ├── test_router.py             # 路由模块测试（41 条）
    ├── test_retrieval.py          # 检索模块测试
    ├── test_solution.py           # Solution Agent 测试
    ├── test_regression.py         # 回归测试
    ├── test_filter.py             # 产品过滤测试
    ├── test_memory.py             # Memory 模块测试
    ├── test_query_understanding.py  # Query Understanding 测试
    ├── test_observability.py     # 可观测性测试
    ├── test_tasks.py              # 异步任务测试
    └── test_first_contact.py      # 首次接待模块测试
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

基于 **72 条 Golden Dataset**（`eval/golden_dataset.json`）的最新评测结果：

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
> 复现命令见 [eval/README.md](eval/README.md)。

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
