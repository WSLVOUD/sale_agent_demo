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
| **首次客户固定工作流（First Contact）** | 🚧 规划中 |
| **Memory 持久化（SQLite）** | 🚧 规划中 |

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
| **Fast Path** | 纯参数查询、问候语、保修/异议等固定场景 | 结构化过滤 + 模板回复，**无需 LLM 调用** |
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
│  - 最近 20 条对话消息                        │
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
│   │   ├── store.py               # MemoryStore 单例（Session FIFO，20 条上限）
│   │   └── enhanced.py            # EnhancedMemoryStore（三层记忆架构）
│   │
│   ├── prompts/                    # 所有 Prompt 模板（模块化管理）
│   │   ├── common/                # 通用格式/系统 Prompt
│   │   ├── sales/                 # Sales Agent Prompt（classify / requirement / reply）
│   │   └── solution/              # Solution Agent Prompt（intent / recommend / reflection）
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
├── first_contact/                  # 🚧 规划中：首次客户固定工作流
│   ├── handler.py                 # 首次接待流程控制器
│   ├── profile.py                 # 销售人员/公司信息加载
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
├── vectorstore_rebuilt/            # 重建后的向量库
│
├── eval/                          # 评估体系
│   ├── dataset.json               # Golden Dataset（20 条标注 Query）
│   ├── report.json                # 评测报告（JSON 格式）
│   └── run_eval.py               # 评测脚本
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
    └── test_tasks.py              # 异步任务测试
```

---

## 数据模型

### Chunk Metadata（存入向量库）

| 字段 | 类型 | 说明 |
|------|------|------|
| `display_type` | `LED` / `LCD` / `IFP` | 产品类型 |
| `indoor` | bool | 室内适用 |
| `outdoor` | bool | 室外适用 |
| `brightness_min_cd` | int | 最低亮度（cd/m²） |
| `brightness_max_cd` | int | 最高亮度（cd/m²） |
| `pixel_pitch_min_mm` | float | 最小点间距（mm） |
| `pixel_pitch_max_mm` | float | 最大点间距（mm） |
| `is_rental` | bool | 租赁场景 |
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
```

---

## 评测结果

基于 20 条 Golden Dataset 测试用例的最新评测结果：

| 指标 | 结果 |
|------|------|
| Query Understanding 准确率 | 75%（15/20） |
| 约束条件匹配率（均值） | 31.8% |
| Recall@5（均值） | 32.5% |
| Recall@10（均值） | 37.5% |
| MRR（均值） | 29.7% |
| Factual Accuracy（均值） | 100% |
| 平均延迟 | 12.8 秒 |

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
| Phase 13 | 测试套件 | ✅ 完成（154 条通过，7 条跳过） |
| 🚧 Phase 14 | Memory 持久化（SQLite） | 🚧 规划中 |
| 🚧 Phase 15 | 首次客户固定工作流（First Contact） | 🚧 规划中 |

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
