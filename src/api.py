"""
FastAPI application for LED RAG System.
Provides REST API endpoints for product selection using LangGraph Agent.

安全加固（Phase 4）：
  - 全局 API Key 鉴权
  - SSE 流式输出
  - LLM 降级机制
"""
from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn
import logging
import os
import asyncio
import time
from pathlib import Path

from src.config import config
from src.rag.corpus import build_retrieval_corpus, corpus_as_dicts, corpus_summary
from src.rag.loader import ENVIRONMENT_METADATA_VERSION
from src.core.embeddings import load_vectorstore, recreate_vectorstore
from src.memory.store import memory
from src.orchestrator import DualAgentOrchestrator
from src.rag.rerank import sanitize_customer_response
from src.tasks import task_manager, TaskStatus
from src.core.fallback import fallback_manager, LLMFallbackManager, CircuitBreaker, CircuitBreakerConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LED Product RAG API",
    description="LED Product Selection Assistant with LangGraph Agent",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
static_dir = os.path.join(project_root, "static")
data_dir = os.path.join(project_root, "data")
logger.info(f"Static directory: {static_dir}")
logger.info(f"Data directory: {data_dir}")

# Mount first_contact assets (videos, PDFs) from data directory FIRST (more specific path)
first_contact_dir = os.path.join(data_dir, "first_contact")
if os.path.exists(first_contact_dir):
    app.mount("/static/first_contact", StaticFiles(directory=first_contact_dir, html=False), name="first_contact")
else:
    logger.warning(f"First contact directory not found: {first_contact_dir}")

# Mount business_card assets (images) from data directory
buisness_card_dir = os.path.join(data_dir, "buisness_card")
if os.path.exists(buisness_card_dir):
    app.mount("/static/buisness_card", StaticFiles(directory=buisness_card_dir, html=False), name="buisness_card")
else:
    logger.warning(f"Business card directory not found: {buisness_card_dir}")

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")
else:
    logger.warning(f"Static directory not found: {static_dir}")

# Pydantic models
class ImageInput(BaseModel):
    """客户随消息发送的图片（《智谱视觉需求提取接入实施计划》第十五阶段）。

    三种传法任选其一：
      url         http(s) 图片地址，或 data:image/...;base64,xxx
      data        裸 base64（需带 mime_type）
      mime_type   仅 data 方式需要，默认 image/jpeg
    """

    url: Optional[str] = None
    data: Optional[str] = None
    mime_type: Optional[str] = "image/jpeg"


class ChatRequest(BaseModel):
    session_id: str
    # 允许纯图片消息（不带文字）
    question: str = ""
    images: Optional[List[ImageInput]] = None
    # ── v2.5：客户一次连续发多条消息 → 前端把它们作为一个 turn 一起送来 ──
    # 结构：[{"text": "...", "images": [...], "message_id": "..."}]（按到达顺序）
    messages: Optional[List[dict]] = None
    client_message_ids: Optional[List[str]] = None

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    requirement: Optional[dict] = None
    reflection_score: Optional[float] = None
    products: Optional[List[dict]] = None
    route: Optional[str] = None  # "fast" | "agent"
    complexity: Optional[str] = None
    first_contact_intro: Optional[str] = None  # First contact self-introduction
    first_contact_messages: Optional[List[dict]] = None  # First contact generated messages (intro + assets)
    # v2.6 §4/§24：一个 turn 只有一条客户可见回复 —— 追加内容已并入 answer，
    # 这个字段仅为老客户端兼容而保留（正常恒为空列表）。
    extra_messages: Optional[List[str]] = None
    vision: Optional[dict] = None  # 视觉需求提取指标（计划第二十二阶段）
    # ── v2.6 §24：对外只暴露"一个 turn 的最终决定"────────────────────────
    turn_id: Optional[str] = None
    action: Optional[str] = None
    question_slot: Optional[str] = None
    question_count: Optional[int] = None
    response_count: Optional[int] = None
    # debug_context 只给内部排查用，**前端不得**当成第二条消息渲染
    debug_context: Optional[dict] = None

class ClearMemoryRequest(BaseModel):
    session_id: str

class RebuildResponse(BaseModel):
    """POST /rebuild 的响应"""
    task_id: str
    message: str
    status: str

class TaskStatusResponse(BaseModel):
    """GET /rebuild/{task_id} 的响应"""
    task_id: str
    name: str
    status: str
    progress: int
    message: str
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None


# ── API 鉴权 ────────────────────────────────────────────────────────────────
def _get_api_key() -> str:
    """获取配置的 API Key（支持 .env 中的 LED_API_KEY）。"""
    return os.environ.get("LED_API_KEY", "")


def _normalize_images(images: Optional[List[ImageInput]]) -> List[str]:
    """把请求里的图片统一成"URL 或 data URL"字符串（Vision 模块只认这两种）。

    计划第二十一阶段：这里做数量与格式的第一层校验，超限直接忽略多余图片，
    不让异常图片把主流程打断。
    """
    if not images:
        return []
    limit = int(getattr(config, "VISION_MAX_IMAGES", 3) or 3)
    normalized: List[str] = []
    for item in list(images)[:limit]:
        try:
            if item.url:
                normalized.append(item.url.strip())
            elif item.data:
                mime = (item.mime_type or "image/jpeg").split(";")[0].strip()
                normalized.append(f"data:{mime};base64,{item.data.strip()}")
            else:
                logger.warning("Image payload without url/data — ignored")
        except Exception as error:  # pragma: no cover - 防御式
            logger.warning("Bad image payload ignored: %s", error)
    if len(images) > limit:
        logger.warning("Too many images (%d), only first %d used", len(images), limit)
    return normalized


def _merge_turn_request(request: "ChatRequest") -> tuple[str, List[str]]:
    """v2.5：把"一次请求里的多条消息"合并成一个 turn（并做 message_id 幂等）。

    · 前端把客户连续发的多条消息放进 `messages`（按顺序）；
    · 老客户端仍然只发 `question` / `images`，两条路径都支持；
    · 同一个 `message_id` 重复提交（重试 / 网络重发）只处理一次。
    """
    try:
        from .input import merge_request_payload

        payload = merge_request_payload(
            str(getattr(request, "session_id", "") or ""),
            question=str(getattr(request, "question", "") or ""),
            images=[_normalize_images(getattr(request, "images", None))] if getattr(request, "images", None) else None,
            messages=getattr(request, "messages", None),
            message_ids=getattr(request, "client_message_ids", None),
        )
        text = payload.text or str(getattr(request, "question", "") or "")
        images = payload.images
        if text != str(getattr(request, "question", "") or "") or images:
            logger.info(
                "[TurnAggregate] parts=%d → text_len=%d images=%d ids=%s",
                len(getattr(request, "messages", None) or []) or 1,
                len(text), len(images), payload.message_ids,
            )
        return text, images
    except Exception as error:  # pragma: no cover - 防御式
        logger.warning("Turn aggregation failed, falling back: %s", error)
        return str(getattr(request, "question", "") or ""), _normalize_images(
            getattr(request, "images", None)
        )


def _verify_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> str:
    """验证 API Key，失败返回 401。"""
    configured_key = _get_api_key()
    # 如果未配置 API Key，跳过验证（开发模式）
    if not configured_key:
        return ""
    if not x_api_key or x_api_key != configured_key:
        raise HTTPException(status_code=401, detail="未授权：请提供有效的 API Key")
    return x_api_key

# Global variables
orchestrator: DualAgentOrchestrator = None
vectorstore = None


_REQUIRED_METADATA = {
    "indoor", "outdoor", "display_type",
    "environment_metadata_version", "product_category",
    # Phase 1/v2.0 新增的功能性字段：缺失说明语料 schema 变过，需要重建
    "gob", "flexible", "modules_per_cabinet",
}


def _validate_vectorstore(store_dir: Path, expected_count: int):
    """加载并校验一个向量库目录，返回 ``(vectorstore, count, reasons)``。

    reasons 为空表示这个目录是"当前语料版本"的有效向量库，可以直接用。
    """
    try:
        loaded = load_vectorstore(persist_dir=str(store_dir))
    except Exception as error:
        return None, 0, [f"加载失败: {error}"]

    try:
        collection = loaded._collection
        count = collection.count()
        metadatas = collection.get(include=["metadatas"]).get("metadatas", [])
    except Exception as error:
        return loaded, 0, [f"读取失败: {error}"]

    reasons: list = []
    if count == 0:
        reasons.append("没有任何记录")
    if len(metadatas) != count:
        reasons.append(f"metadata 数量({len(metadatas)})与记录数({count})不一致")
    if count != expected_count:
        reasons.append(f"记录数 {count} != 期望 {expected_count}")
    invalid = [
        metadata for metadata in metadatas
        if not metadata or not _REQUIRED_METADATA.issubset(metadata)
        or not isinstance(metadata.get("indoor"), bool)
        or not isinstance(metadata.get("outdoor"), bool)
        or metadata.get("display_type") not in {"LED", "LCD", "IFP"}
        or metadata.get("environment_metadata_version") != ENVIRONMENT_METADATA_VERSION
        or metadata.get("level") != "model"
    ]
    if invalid:
        reasons.append(f"{len(invalid)} 条记录的 metadata 是旧版本")
    return loaded, count, reasons


def get_documents_for_retrieval():
    """构建 Model 级检索语料（Phase 2 起为唯一产品语料来源）。

    返回 ``[{"id", "text", "metadata"}]``，供 Vector / Sparse / BM25 三路共用。
    """
    try:
        documents = build_retrieval_corpus(config.DATA_DIR)
        logger.info("Retrieval corpus: %s", corpus_summary(documents))
        return corpus_as_dicts(documents)
    except Exception as e:
        logger.error("Failed to build retrieval corpus: %s", e)
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    global orchestrator, vectorstore
    
    logger.info("=" * 50)
    logger.info("Initializing LED RAG System v2.0 (Sales + Solution Agents)...")
    logger.info("=" * 50)
    
    try:
        # Load source documents（Phase 2：Model 级语料，一个实际销售型号一个 Document）
        corpus_documents = build_retrieval_corpus(config.DATA_DIR)
        documents = corpus_as_dicts(corpus_documents)
        if not documents:
            raise RuntimeError("No product documents found; cannot initialize retrieval")
        logger.info("Retrieval corpus: %s", corpus_summary(corpus_documents))

        # Check if vector store needs rebuild.
        # 依次尝试「主目录 → 稳定回退目录（<dir>_rebuilt）」：Windows 上主目录可能被
        # 别的进程锁住，重建只能落到回退目录；下次启动若回退目录已经是**当前版本**的
        # 有效向量库，就直接复用它，绝不再重新嵌入一遍（否则每次启动都要等一次全量重建）。
        fresh_documents = corpus_documents
        expected_count = len(fresh_documents)
        primary_dir = Path(config.VECTORSTORE_DIR)
        candidate_dirs = [primary_dir, Path(str(primary_dir) + "_rebuilt")]

        collection_count = 0
        vectorstore = None
        for candidate in candidate_dirs:
            sqlite_path = candidate / "chroma.sqlite3"
            if not sqlite_path.exists() or os.path.getsize(sqlite_path) == 0:
                continue
            loaded, collection_count, reasons = _validate_vectorstore(candidate, expected_count)
            if loaded is not None and not reasons:
                vectorstore = loaded
                collection = vectorstore._collection
                if str(candidate) != str(primary_dir):
                    logger.warning(
                        "主向量库不可用 → 直接复用已重建好的向量库：%s（%d 条记录）",
                        candidate, collection_count,
                    )
                    config.VECTORSTORE_DIR = str(candidate)
                else:
                    logger.info(
                        "Existing vector store has valid v%s metadata: %d records",
                        ENVIRONMENT_METADATA_VERSION, collection_count,
                    )
                break
            logger.warning(
                "向量库 %s 需要重建：%s", candidate, "; ".join(reasons) or "未知原因"
            )
            # 释放进程内句柄，否则文件被自己锁住、删不掉（Windows）
            from src.core.embeddings import _release_vectorstore_handles

            _release_vectorstore_handles()

        if vectorstore is None:
            logger.warning(
                "Vector store needs rebuild: expected=%d chunks", expected_count
            )
            vectorstore = recreate_vectorstore(fresh_documents)
            active_dir = getattr(vectorstore, "_active_persist_dir", config.VECTORSTORE_DIR)
            logger.info("Vector store rebuilt and persisted at %s", active_dir)
            config.VECTORSTORE_DIR = active_dir
            try:
                vectorstore = load_vectorstore(persist_dir=active_dir)
            except Exception as error:
                logger.warning("Could not reload vectorstore after rebuild: %s", error)
            collection = vectorstore._collection

        logger.info(
            "Vector store ready: path=%s collection=%s records=%s",
            config.VECTORSTORE_DIR,
            collection.name,
            collection.count(),
        )
        
        # Initialize agents
        from src.agents.sales.runner import SalesAgentRunner
        from src.agents.solution.runner import SolutionAgentRunner
        from src.rag.sparse import get_sparse_search
        from src.rag.bm25 import BM25Search
        from src.rag.fusion import HybridSearch

        sparse = get_sparse_search(documents, config.VECTORSTORE_DIR) if documents else None
        bm25 = BM25Search(documents) if documents else None

        # Initialize Sales Agent
        sales_agent = SalesAgentRunner(sales_search=vectorstore)
        sales_agent.setup()
        logger.info("Sales Agent initialized")

        # Initialize Solution Agent
        solution_agent = SolutionAgentRunner(vectorstore, documents)
        logger.info("Solution Agent initialized")
        
        # Initialize orchestrator
        orchestrator = DualAgentOrchestrator(sales_agent, solution_agent)
        logger.info("Orchestrator initialized with Sales Agent + Solution Agent")

        # Smoke test
        display_types_present = set(
            doc.get("metadata", {}).get("display_type")
            for doc in documents
            if doc.get("metadata", {}).get("display_type")
        )
        smoke_queries = {
            "unfiltered": ("display", None),
            "indoor": ("indoor display", {"indoor": True}),
            "outdoor": ("outdoor display", {"outdoor": True}),
        }
        for dt in display_types_present:
            smoke_queries[dt] = (f"{dt} display", {"display_type": dt})
        smoke_results = {
            name: len(
                sales_agent.sales_search.similarity_search(
                    query, k=1, filter=filters
                )
            )
            for name, (query, filters) in smoke_queries.items()
        }
        logger.info("Retrieval smoke test: %s", smoke_results)
        failed_smoke_tests = [name for name, count in smoke_results.items() if count == 0]
        if failed_smoke_tests:
            raise RuntimeError(
                f"Vector retrieval smoke test failed for {failed_smoke_tests}; "
                f"path={config.VECTORSTORE_DIR}, records={collection.count()}"
            )
        
        logger.info("=" * 50)
        logger.info("System ready!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "LED Product RAG API v2.0",
        "status": "running",
        "version": "2.0.0",
        "features": ["LangGraph Agent", "Hybrid Search (Vector + BM25)", "Reflection"]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    collection = vectorstore._collection if vectorstore else None
    return {
        "status": "healthy",
        "orchestrator_ready": orchestrator is not None,
        "vectorstore_ready": vectorstore is not None,
        "vectorstore_path": config.VECTORSTORE_DIR,
        "collection": collection.name if collection else None,
        "record_count": collection.count() if collection else 0,
    }


@app.get("/diagnostics/retrieval")
async def retrieval_diagnostics():
    """Report vector-store metadata coverage."""
    if not vectorstore:
        raise HTTPException(status_code=503, detail="Vector store is not initialized")

    collection = vectorstore._collection
    metadatas = collection.get(include=["metadatas"]).get("metadatas", [])
    return {
        "path": config.VECTORSTORE_DIR,
        "collection": collection.name,
        "record_count": collection.count(),
        "metadata_count": len(metadatas),
        "indoor_records": sum(metadata.get("indoor") is True for metadata in metadatas if metadata),
        "outdoor_records": sum(metadata.get("outdoor") is True for metadata in metadatas if metadata),
        "led_records": sum(metadata.get("display_type") == "LED" for metadata in metadatas if metadata),
        "lcd_records": sum(metadata.get("display_type") == "LCD" for metadata in metadatas if metadata),
        "ifp_records": sum(metadata.get("display_type") == "IFP" for metadata in metadatas if metadata),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request, api_key: str = Depends(_verify_api_key)):
    """聊天接口，支持常规和 SSE 流式两种模式。
    
    请求头：
      X-API-Key: API 密钥（必填，除非 .env 中未配置 LED_API_KEY）
      Accept: text/event-stream → 启用 SSE 流式输出
      
    请求体：
      session_id: 会话 ID
      question: 用户问题
    """
    if not orchestrator:
        raise HTTPException(status_code=500, detail="System not initialized")

    # ── SSE 流式检测 ─────────────────────────────────────────────────
    accept_header = http_request.headers.get("Accept", "")
    use_stream = "text/event-stream" in accept_header

    if use_stream:
        return StreamingResponse(
            _stream_chat(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── 普通模式 ────────────────────────────────────────────────────
    return await _chat_sync(request)


@app.post("/memory/clear")
async def clear_memory(request: ClearMemoryRequest, api_key: str = Depends(_verify_api_key)):
    """清除指定会话的内存。"""
    memory.clear(request.session_id)
    # 语义缓存是按会话隔离的，清会话时必须一起清（否则同名消息会复用旧上下文）
    try:
        from src.core.requirement_extractor import get_requirement_extractor

        get_requirement_extractor().clear_session_semantics(request.session_id)
    except Exception as error:  # pragma: no cover - 防御式
        logger.warning("Clear semantic cache failed for %s: %s", request.session_id, error)
    return {
        "message": "Memory cleared",
        "session_id": request.session_id
    }


@app.get("/memory/{session_id}")
async def get_memory(session_id: str, api_key: str = Depends(_verify_api_key)):
    """获取指定会话的聊天历史和需求。"""
    messages = memory.get_history(session_id)
    requirements = memory.get_requirements(session_id)
    return {
        "session_id": session_id,
        "count": len(messages),
        "messages": messages,
        "requirements": requirements,
    }


# ── LLM Fallback ──────────────────────────────────────────────────────────
_LLM_FALLBACK_RESPONSES = {
    "greeting": "Hello! I'm your LED display advisor. How can I help you today?",
    "warranty": "Our products come with a 1-year warranty by default, and the warranty can be extended for an additional fee. What scenario are you looking to use it for?",
    "product_question": "Thanks for your question! We carry the full range: LED, LCD, and IFP displays. Is there anything specific you'd like to know more about?",
    "default": "Sorry, the system is busy right now. Please try again shortly, or reach out to our support team.",
}


def _get_fallback_response(question: str) -> str:
    """Return fallback response based on question type."""
    q_lower = question.lower()
    if any(kw in q_lower for kw in ("hello", "hi", "你好", "您好")):
        return _LLM_FALLBACK_RESPONSES["greeting"]
    if any(kw in q_lower for kw in ("warranty", "guarantee", "质保", "保修")):
        return _LLM_FALLBACK_RESPONSES["warranty"]
    if any(kw in q_lower for kw in ("brightness", "pixel pitch", "resolution", "size", "spec", "亮度", "点间距", "分辨率", "尺寸")):
        return _LLM_FALLBACK_RESPONSES["product_question"]
    return _LLM_FALLBACK_RESPONSES["default"]


async def _chat_sync(request: ChatRequest) -> ChatResponse:
    """同步聊天处理，包含 LLM 降级逻辑。"""
    logger.info("Chat sync from session: %s", request.session_id)
    
    # 检查 Circuit Breaker
    cb = fallback_manager.get_circuit_breaker("chat")
    if not cb.is_available():
        logger.warning("CircuitBreaker OPEN, returning fallback response")
        fallback_answer = fallback_manager.get_fallback(
            fallback_manager.classify_for_fallback(request.question)
        )
        return ChatResponse(
            session_id=request.session_id,
            answer=fallback_answer,
            requirement={},
            reflection_score=0,
            products=[],
            route="circuit_breaker",
            complexity="fallback",
        )
    
    try:
        # v2.5：多条消息合成一个 turn（含 message_id 幂等）
        merged_text, request_images = _merge_turn_request(request)
        _parts = list(getattr(request, "messages", None) or [])
        result = orchestrator.process_message(
            message=merged_text,
            session_id=request.session_id,
            images=request_images or None,
            # v2.5++++（计划 §15）：把"这一轮由几条消息聚合而来"告诉编排器，
            # 日志里就能看到 messages=3 aggregated=true
            message_count=len(_parts) or 1,
            aggregated=len(_parts) > 1,
        )
        # 记录成功
        fallback_manager.record_success("chat")
        
    except Exception as llm_error:
        # 记录失败
        fallback_manager.record_failure("chat")
        logger.warning("LLM error in orchestrator, using fallback: %s", llm_error)
        
        # 检查是否应该人工接管
        if fallback_manager.should_handover():
            handover_message = (
                "Sorry, we're having some trouble on our side right now, and our team has been "
                "notified to follow up with you. You can also reach our support line: 400-xxx-xxxx."
            )
            fallback_manager.reset_handover()
            return ChatResponse(
                session_id=request.session_id,
                answer=handover_message,
                requirement={},
                reflection_score=0,
                products=[],
                route="human_handover",
                complexity="handover",
            )
        
        # 使用降级响应
        fallback_answer = fallback_manager.get_fallback(
            fallback_manager.classify_for_fallback(request.question)
        )
        return ChatResponse(
            session_id=request.session_id,
            answer=fallback_answer,
            requirement={},
            reflection_score=0,
            products=[],
            route="fallback",
            complexity="fallback",
        )

    requirements = result.get("requirements") or {}
    location_type = str(requirements.get("location_type", ""))
    is_outdoor = bool(requirements.get("outdoor")) or location_type in (
        "户外", "室外", "外面", "露天", "全户外", "半户外", "户外使用", "室外使用",
    )
    # 多屏回复（一个项目多块屏）里每块屏各自有环境：室内那块的型号本来就是
    # 室内型号，不能拿"当前这块屏是室外"去整段过滤 —— 否则会出现
    # "只推荐了一块屏"（实测 bug：Screen 1 的整段回复被删掉，只剩 Screen 2）。
    response_text = sanitize_customer_response(
        result.get("response", ""),
        outdoor=is_outdoor and not result.get("multi_screen"),
    )
    # 语言护栏：策略=en 时绝不让中文发给客户（命中就用一次 LLM 重写成英文；
    # 重写不了就退回英文兜底话术，见下面的 not response_text 分支）
    from src.rag.reply_composer import enforce_english

    response_text = enforce_english(response_text, message=request.question)

    if not response_text:
        # 【客户口径】没有匹配结果时不说"找不到"，改成邀请客户放宽某个条件
        from src.rag.reply_composer import relaxation_answer, reply_language

        if result.get("products"):
            # 有产品但清洗后为空：说明回复里全是"不该出现的内容"（例如给室外推荐了
            # 室内型号被过滤掉）。这种情况要把日志留清楚，不要静默换成兜底话术。
            logger.warning(
                "Response was emptied by sanitizer while %d products were returned; "
                "falling back to the relaxation request (session=%s)",
                len(result.get("products") or []),
                request.session_id,
            )
        response_text = relaxation_answer(reply_language(request.question))

    logger.info("Response: %s...", response_text[:100])
    # v2.6 §24：对外只暴露"一个 turn → 一个 action → 一条回复"；
    # 其余内部信息（候选动作 / 会话状态 / 被丢弃的问题）只在 debug_context 里。
    final_response = result.get("final_response") or {}
    debug_context = {
        "conversation_state": result.get("conversation_state") or {},
        "decision_audit": result.get("decision_audit") or {},
        "final_response": final_response,
    }
    return ChatResponse(
        session_id=request.session_id,
        answer=response_text,
        requirement=requirements,
        reflection_score=None,
        products=result.get("products", []),
        route=result.get("route"),
        complexity=result.get("complexity"),
        first_contact_intro=result.get("_perf", {}).get("first_contact_intro"),
        first_contact_messages=result.get("_perf", {}).get("first_contact_messages"),
        # 计划 §4.2：追加气泡已并入 answer，这里不再单独返回
        extra_messages=[],
        vision=result.get("vision"),
        turn_id=result.get("turn_id"),
        action=result.get("action"),
        question_slot=result.get("question_slot"),
        question_count=result.get("question_count"),
        response_count=result.get("response_count", 1),
        debug_context=debug_context,
    )


async def _stream_chat(request: ChatRequest):
    """SSE 流式聊天处理。

    与普通 ``/chat`` 走**同一条完整管线**（首轮接待 → Sales Agent 需求采集 →
    Recommendation Ready Gate → 选型/计算/表达），只是把最终文本拆块输出。

    旧实现直接调 ``solution_agent.run_stream()``，会绕过 Sales Agent 与需求采集，
    导致"只知道室内外 + 场景"就推荐。这里修正为统一入口。
    """
    import json as _json

    session_id = request.session_id
    question = request.question
    stream_images = _normalize_images(request.images) or None

    yield f"data: {_json.dumps({'type': 'start', 'session_id': session_id})}\n\n"

    start_time = time.time()
    try:
        result = orchestrator.process_message(
            message=question,
            session_id=session_id,
            images=stream_images,
        )
        answer = str(result.get("response") or "")
        if not answer:
            answer = _get_fallback_response(question)
        # 语言护栏：策略=en 时流式输出同样不能出现中文
        from src.rag.reply_composer import enforce_english

        answer = enforce_english(answer, message=question) or _get_fallback_response(question)
        # v2.6 §4：追加气泡已经在收口层并入 `response`；这里若还有（老链路兜底）
        # 也只是拼进**同一条**消息，绝不再产生第二个气泡。
        extras = [str(item) for item in (result.get("extra_messages") or []) if item]
        if extras:
            answer = answer + "\n\n" + "\n\n".join(extras)

        ttft = (time.time() - start_time) * 1000
        yield f"data: {_json.dumps({'type': 'ttft', 'ms': round(ttft, 1)})}\n\n"

        # 按句/词组切块，保持 SSE 语义但不依赖真实 token 流
        import re as _re

        chunks = _re.findall(r"\S+\s*", answer)
        for chunk in chunks:
            yield f"data: {_json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
    except Exception as exc:
        logger.error("Stream error: %s", exc)
        yield f"data: {_json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        yield f"data: {_json.dumps({'type': 'chunk', 'content': _get_fallback_response(question)})}\n\n"

    total_time = (time.time() - start_time) * 1000
    yield f"data: {_json.dumps({'type': 'done', 'total_ms': round(total_time, 1)})}\n\n"


# ── rebuild ────────────────────────────────────────────────────────────────────────
# Phase 4: 异步 rebuild 后台任务

async def _create_rebuild_coro(task_id_ref: list) -> dict:
    """后台重建协程（接收 task_id 引用）"""
    # 注意：必须用 recreate_vectorstore（会先清空旧 collection）。
    # create_vectorstore 是"追加"语义，直接用它会把语料重复写入（49 → 98）。
    from src.core.embeddings import recreate_vectorstore
    from src.agents.sales.runner import SalesAgentRunner
    from src.agents.solution.runner import SolutionAgentRunner

    task_id = task_id_ref[0]
    
    task_manager.update_progress(task_id, 10, "加载 Model 级产品数据...")
    corpus_documents = build_retrieval_corpus(config.DATA_DIR)
    documents = corpus_as_dicts(corpus_documents)
    
    task_manager.update_progress(task_id, 30, "创建向量库...")
    new_vectorstore = recreate_vectorstore(corpus_documents)
    active_dir = getattr(new_vectorstore, "_active_persist_dir", config.VECTORSTORE_DIR)
    config.VECTORSTORE_DIR = active_dir
    task_manager.update_progress(task_id, 40, f"向量库已重建: {active_dir}")
    
    task_manager.update_progress(task_id, 50, "初始化 Sales Agent...")
    new_sales_agent = SalesAgentRunner(sales_search=new_vectorstore)
    
    task_manager.update_progress(task_id, 70, "初始化 Solution Agent...")
    new_solution_agent = SolutionAgentRunner(new_vectorstore, documents)
    
    task_manager.update_progress(task_id, 85, "初始化编排器...")
    new_orchestrator = DualAgentOrchestrator(new_sales_agent, new_solution_agent)
    
    # 原子更新全局变量
    task_manager.update_progress(task_id, 95, "更新全局状态...")
    global vectorstore, orchestrator
    vectorstore = new_vectorstore
    orchestrator = new_orchestrator
    
    task_manager.update_progress(task_id, 100, "完成")
    
    return {
        "message": f"向量库重建成功，文档数: {len(documents)}",
        "document_count": len(documents),
    }


@app.post("/rebuild", response_model=RebuildResponse)
async def rebuild_vectorstore(api_key: str = Depends(_verify_api_key)):
    """
    异步重建向量库（管理员接口）
    
    POST /rebuild
        - 立即返回 task_id
        - 后台异步执行 rebuild
        - 可通过 GET /rebuild/{task_id} 查询进度
        
    返回：
        {
            "task_id": "abc123",
            "message": "重建任务已提交",
            "status": "pending"
        }
    """
    # 先创建占位 task_id
    import uuid
    task_id = str(uuid.uuid4())[:8]
    task_id_holder = [task_id]
    
    # 创建后台任务（把 task_id 显式传给任务管理器，
    # 否则任务管理器会另生成一个 id，导致 GET /rebuild/{task_id} 永远 404）
    await task_manager.create_task(
        name="rebuild_vectorstore",
        coro=_create_rebuild_coro(task_id_holder),
        task_id=task_id,
    )
    
    logger.info(f"Rebuild task submitted: {task_id}")
    
    return RebuildResponse(
        task_id=task_id,
        message="重建任务已提交，请通过 GET /rebuild/{task_id} 查询进度",
        status="pending"
    )


@app.get("/rebuild/{task_id}", response_model=TaskStatusResponse)
async def get_rebuild_status(task_id: str, api_key: str = Depends(_verify_api_key)):
    """
    查询 rebuild 任务状态
    
    GET /rebuild/{task_id}
    
    返回：
        {
            "task_id": "abc123",
            "name": "rebuild_vectorstore",
            "status": "running",
            "progress": 50,
            "message": "正在创建向量库...",
            "created_at": 1699999999.0,
            "started_at": 1699999999.1,
            "completed_at": null,
            "result": null,
            "error": null
        }
    """
    task_info = task_manager.get_task(task_id)
    
    if not task_info:
        raise HTTPException(
            status_code=404,
            detail=f"任务 {task_id} 不存在或已过期"
        )
    
    return TaskStatusResponse(
        task_id=task_info.task_id,
        name=task_info.name,
        status=task_info.status.value,
        progress=task_info.progress,
        message=task_info.message,
        created_at=task_info.created_at,
        started_at=task_info.started_at,
        completed_at=task_info.completed_at,
        result=task_info.result,
        error=task_info.error,
    )


@app.get("/rebuild", response_model=list[dict])
async def list_rebuild_tasks(api_key: str = Depends(_verify_api_key)):
    """列出所有 rebuild 任务"""
    return task_manager.list_tasks()


@app.post("/rebuild/{task_id}/cancel")
async def cancel_rebuild(task_id: str, api_key: str = Depends(_verify_api_key)):
    """取消正在执行的 rebuild 任务"""
    success = await task_manager.cancel_task(task_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"任务 {task_id} 无法取消（可能已完成或不存在）"
        )
    return {"message": f"任务 {task_id} 已请求取消", "task_id": task_id}


def run_server():
    """Run the FastAPI server."""
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info"
    )


# ── Circuit Breaker 状态 ─────────────────────────────────────────────────────

@app.get("/circuit-breaker/status")
async def get_circuit_breaker_status():
    """查询熔断器状态"""
    return {
        "state": fallback_manager.get_circuit_breaker("chat").state.value,
        "consecutive_failures": fallback_manager._consecutive_failures,
        "should_handover": fallback_manager.should_handover(),
    }

@app.post("/circuit-breaker/reset")
async def reset_circuit_breaker(api_key: str = Depends(_verify_api_key)):
    """重置熔断器（管理员接口）"""
    fallback_manager.get_circuit_breaker("chat").reset()
    fallback_manager.reset_handover()
    return {"message": "Circuit breaker reset successfully"}


if __name__ == "__main__":
    run_server()
