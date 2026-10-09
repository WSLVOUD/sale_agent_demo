# LED 销售顾问

面向 LED、LCD 和 IFP 显示产品的对话式销售顾问。系统通过多轮对话收集客户需求、在需求明确后检索合适的产品，并返回面向客户的答复；也提供需求图片提取、会话历史、知识库重建和离线评估能力。

本文说明项目用途、主要处理流程、数据约定、本地运行方式和验证入口。API 的完整交互定义以 `src/api.py` 为准，离线评估细节见 [`eval/README.md`](eval/README.md)。

## 目录概览

```text
src/
  api.py                    FastAPI 应用、HTTP 路由和请求/响应模型
  agents/                   销售对话及方案生成流程
  core/                     需求抽取、对话编排等核心逻辑
  dialogue/                 对话状态、轮次约束及回复策略
  memory/                   会话历史与需求状态
  models/                   需求模型及兼容适配
  rag/                      查询理解、检索、推荐与回复合成
  vision/                   图片输入的视觉需求提取
  utils/                    公共工具
tests/                      自动化测试
eval/                       离线评估数据、脚本和使用说明
data/                       产品目录及业务知识数据
models/                     本地文本向量模型（不从 Hugging Face 在线下载）
```

## 一轮对话如何处理

```text
客户端
  → POST /chat
  → TurnExecutor（合并/去重消息、会话锁、幂等处理）
  → 对话编排与需求理解
  → RequirementProfile（该会话需求的唯一事实来源）
  → 检索、硬约束过滤与方案生成
  → 回复护栏与答复合成
  → 返回一个 turn 的最终答复
```

- 当需求尚不完整时，系统根据当前需求状态继续澄清；达到推荐条件后再检索和推荐。
- 检索将本地向量搜索与 BM25 等能力组合使用，并对产品类型、使用环境、安装方式等条件执行约束。
- 图片提取为可选输入能力；提取出的信息合并到同一个 `RequirementProfile`，而不是建立另一套需求状态。
- 一个 turn 对客户端返回一条最终答复。`extra_messages` 为旧客户端兼容字段，正常情况下为空；不要将 `debug_context` 渲染成面向客户的消息。

### 需求状态和旧模块兼容

`RequirementProfile` 是会话需求的唯一事实来源。部分旧模块仍然接收字典，因此 [`src/models/legacy_adapter.py`](src/models/legacy_adapter.py) 提供只读投影：

- `profile_to_legacy()`：将 Profile 投影成旧销售/对话模块使用的字典。
- `profile_to_solution_requirement()`：投影成旧方案模块需要的需求字典。
- `rebuild_legacy_view()`：重建目标旧字典里的适配字段；它不会把旧字典的修改写回 Profile。

维护需求字段时，应先更新规范 Profile 及其解析/合并逻辑，再调整适配投影和相应测试。不要让旧字典成为第二份可独立写入的需求状态。

## 环境要求

- Python 3.11（当前项目验证环境）。
- 项目依赖，详见 [`requirements.txt`](requirements.txt)。
- 本地 BGE-M3 文本向量模型：默认从 `models/BAAI--bge-m3/snapshots/master` 查找，也可通过 `EMBEDDING_MODEL_PATH` 指定。
- 产品和知识数据位于 `data/`。
- 一个兼容 OpenAI API 的聊天模型服务及其 API Key，才能使用完整的 LLM 对话能力。可选集成（例如 Tavily 搜索和智谱视觉）需各自配置密钥。

配置模块在加载时会检查本地嵌入模型；若目录或模型权重不完整，应用启动会报错。模型文件不通过本项目自动下载，请在启动前准备好模型。

## 本地启动

以下命令在仓库根目录执行。

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`，至少填入聊天模型 API Key，并确认本地向量模型路径正确。之后启动：

```powershell
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env` 后启动：

```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

服务默认监听 `http://localhost:8000`。交互式 API 文档位于 [`/docs`](http://localhost:8000/docs)，ReDoc 位于 [`/redoc`](http://localhost:8000/redoc)；健康检查为 [`/health`](http://localhost:8000/health)。

## 配置说明

应用从项目根目录的 `.env` 读取配置。`.env.example` 是可用配置项的示例；不要把真实密钥提交到版本库。

| 配置项 | 用途 | 默认值 / 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | 聊天模型密钥（当前配置示例使用 DeepSeek） | 必须配置有效密钥才能调用模型 |
| `MODEL_NAME` | 聊天模型名称 | `deepseek-chat` |
| `VECTORSTORE_DIR` | 主向量库目录 | `./vectorstore` |
| `SALES_VECTORSTORE_DIR` | 销售数据向量库目录 | `./sales_vectorstore` |
| `DATA_DIR` | 产品及知识数据目录 | `./data` |
| `EMBEDDING_MODEL_PATH` | 本地嵌入模型目录 | 留空时尝试项目内置候选路径 |
| `TOP_K` | 默认检索数量 | `5` |
| `LED_API_KEY` | HTTP API 鉴权密钥 | 留空时不启用鉴权，仅适合本地开发 |
| `RESPONSE_LANGUAGE_POLICY` | 回复语言策略：`en` 固定英文，`auto` 跟随客户语言 | `en` |
| `TAVILY_API_KEY` | 可选 Web 搜索 | 未配置时不使用该集成 |
| `GLM_API_KEY`、`VISION_ENABLED` | 可选图片需求提取 | `VISION_ENABLED=true`；需配置有效密钥才能调用 |
| `VISION_MAX_IMAGES`、`VISION_MAX_IMAGE_MB` | 单次图片数量和单张大小限制 | `3` 张、`5` MB |
| `LLM_TIMEOUT_SECS`、`LLM_MAX_RETRIES` | LLM 超时与重试 | `30` 秒、`2` 次 |
| `LED_RAG_HISTORY_LIMIT`、`LED_RAG_HISTORY_TOTAL_CHARS` | 对话历史窗口限制 | `50` 条、`6000` 字符预算 |

当前应用中的聊天模型客户端实际读取 `DEEPSEEK_API_KEY` 和 `MODEL_NAME`。`.env.example` 中保留的 `OPENAI_API_KEY`、`OPENAI_API_BASE` 是示例兼容配置项，当前代码没有读取它们；切换模型服务时应同时确认 `src/core/llm.py` 及各调用点的配置方式。

认证通过请求头 `X-API-Key` 传入。生产环境应设置 `LED_API_KEY`，并将服务放在适当的网络访问控制之后；不要将未鉴权服务直接暴露到不受信任的网络。

## HTTP API

| 方法 | 路径 | 用途 | 鉴权 |
|---|---|---|---|
| `GET` | `/` | 服务基本信息 | 否 |
| `GET` | `/health` | 应用及向量库健康状态 | 否 |
| `GET` | `/diagnostics/retrieval` | 向量库及产品元数据统计 | 否 |
| `POST` | `/chat` | 普通或 SSE 流式对话 | 是（仅当配置 `LED_API_KEY`） |
| `GET` | `/memory/{session_id}` | 获取会话历史和需求 | 是（仅当配置 `LED_API_KEY`） |
| `POST` | `/memory/clear` | 清除指定会话记忆 | 是（仅当配置 `LED_API_KEY`） |
| `POST` | `/rebuild` | 启动知识库重建任务 | 是（仅当配置 `LED_API_KEY`） |
| `GET` | `/rebuild`、`/rebuild/{task_id}` | 查看重建任务列表或状态 | 是（仅当配置 `LED_API_KEY`） |
| `POST` | `/rebuild/{task_id}/cancel` | 取消重建任务 | 是（仅当配置 `LED_API_KEY`） |
| `GET` | `/circuit-breaker/status` | 查看熔断器状态 | 否 |
| `POST` | `/circuit-breaker/reset` | 重置熔断器 | 是（仅当配置 `LED_API_KEY`） |

### 对话示例

没有设置 `LED_API_KEY` 时，本地开发请求不需要认证头：

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-001","question":"I need an indoor LED display for a meeting room."}'
```

启用鉴权后，增加 `-H "X-API-Key: <your-api-key>"`。Windows PowerShell 可使用 `curl.exe` 执行相同请求。

`session_id` 用于隔离不同客户的会话。常用请求字段：

```json
{
  "session_id": "demo-001",
  "question": "I need an indoor LED display.",
  "images": [],
  "turn_id": "optional-idempotency-key",
  "source": "web"
}
```

也可用 `messages` 一次提交同一 turn 中按顺序到达的多条消息；图片可通过 `images` 中的 URL 或 Base64 数据提供。图片请求受视觉配置和大小/数量上限约束。响应包含 `answer`、`turn_id`、`action`、`question_slot` 等字段；响应结构和所有可选字段可在 `/docs` 或 `src/api.py` 中查看。重复提交相同 `turn_id` 可用于幂等重放。

流式模式通过请求头 `Accept: text/event-stream` 启用。清除会话时调用 `POST /memory/clear`，请求体为 `{"session_id":"demo-001"}`。

## 测试与离线评估

### 自动化测试

```bash
python -m pytest
```

可先运行定向测试，例如：

```bash
python -m pytest tests/test_vectorstore_service.py tests/test_readiness_measurements.py
```

### 离线评估

评估脚本、用例和指标定义见 [`eval/README.md`](eval/README.md)。常用命令：

```bash
python -m eval.recommendation_eval
python -m eval.calculator_eval
python -m eval.retrieval_eval
python -m eval.dialogue.run_dialogue_eval
```

推荐与计算评估不需要在线模型；检索评估需要本地 BGE-M3 模型和向量库。`--agent` 等端到端选项还需要可用的模型服务。评估报告写入 `eval/reports/`（默认不纳入版本控制）。

### 最近一次验收快照

以下数字是本次重构验收时的实测结果，不是持续集成承诺；修改评估用例、数据或依赖后请重新运行对应命令。

| 检查 | 结果 |
|---|---|
| 全量自动化测试 | `695 passed, 1 warning` |
| 推荐评估 | 需求槽位准确率 `0.9914`；硬约束捕获率 `0.9896`；派生需求捕获率 `0.4003`；产品类型准确率 `1.0` |
| 计算评估 | `14/14` |
| 检索评估 | 82 cases；过滤后 Recall@10 `0.9511`、MRR `0.9267`、Model Recall@10 `0.9286`；硬约束违规 `0` |
| 对话链路评估 | 12 cases；一轮一问、action 准确率、问题槽位准确率均为 `1.0` |

## 维护提示

- 修改对话轮次、需求抽取或推荐规则后，运行全量测试，并重跑对应离线评估；对话行为变更还应运行对话评估。
- 修改产品数据或向量索引流程后，检查 `/health` 与 `/diagnostics/retrieval`，并运行检索评估。重建任务通过 `/rebuild` API 管理。
- `src/vision/` 属于视觉提取链路；修改图片功能时，应同时验证文字需求流程仍可独立工作。
- 当前部分 LangChain/Chroma 依赖仍使用弃用接口。全量测试中的 warning 是 LangGraph 关于未来 `allowed_objects` 默认值的提示，应在相关依赖升级时单独评估迁移。

## 最近一次架构整理记录

项目曾按职责拆分会话状态、销售需求节点、查询理解、回复合成、API 服务和就绪度测量等模块，并保留旧导入/数据形态的兼容入口。兼容入口的目的在于降低迁移风险，不代表旧模块仍拥有独立的状态来源。架构决策入口约束由 `tests/architecture/test_decision_entrypoints.py` 覆盖。