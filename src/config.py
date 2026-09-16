"""
Configuration module for LED RAG System.
Loads settings from .env file.
"""
import os
from dotenv import load_dotenv

# Resolve project-relative paths from this config file, not the process cwd.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def project_path(value: str) -> str:
    """Resolve a relative setting against the project root, normalized."""
    if os.path.isabs(value):
        return os.path.normpath(value)
    return os.path.normpath(os.path.join(PROJECT_ROOT, value))


class Config:
    """Application configuration"""
    
    # API Configuration
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")
    
    # Server Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    
    # Vector Store Configuration
    VECTORSTORE_DIR = project_path(os.getenv("VECTORSTORE_DIR", "./vectorstore"))
    SALES_VECTORSTORE_DIR = project_path(os.getenv("SALES_VECTORSTORE_DIR", "./sales_vectorstore"))

    # Data Files
    DATA_DIR = project_path(os.getenv("DATA_DIR", "./data"))

    # Local embedding model (never download from HuggingFace Hub)
    EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "")
    
    # Retrieval Configuration
    TOP_K = int(os.getenv("TOP_K", "5"))

    # Tavily Search Configuration
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    # Text Splitting Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    # ── Phase 2: Reflection 预算控制 ──────────────────────────────────
    REFLECTION_MAX_ROUNDS       = int(os.getenv("REFLECTION_MAX_ROUNDS", "3"))
    REFLECTION_MAX_TOKENS       = int(os.getenv("REFLECTION_MAX_TOKENS", "2000"))
    REFLECTION_MAX_TIME_MS      = int(os.getenv("REFLECTION_MAX_TIME_MS", "8000"))
    REFLECTION_NO_IMPROVE_STOP  = int(os.getenv("REFLECTION_NO_IMPROVE_STOP", "2"))

    # ── Phase 4: 安全与可观测性 ──────────────────────────────────────
    LED_API_KEY       = os.getenv("LED_API_KEY", "")  # 空值 = 开发模式（不鉴权）
    LLM_TIMEOUT_SECS  = int(os.getenv("LLM_TIMEOUT_SECS", "30"))
    LLM_MAX_RETRIES  = int(os.getenv("LLM_MAX_RETRIES", "2"))

    # ── v2.0 Phase 14：回复语言策略 ──────────────────────────────────
    #   en   = 始终英语回复（系统既有策略，默认值，保持行为不变）
    #   auto = 跟随客户语言回复（v2.0 文档描述的"Original Language Response"）
    RESPONSE_LANGUAGE_POLICY = os.getenv("RESPONSE_LANGUAGE_POLICY", "en").strip().lower() or "en"

    # ── 智谱视觉需求提取（《智谱视觉需求提取接入实施计划》）──────────────
    # 图片 → 智谱视觉模型 → 结构化需求 → 合并进 RequirementProfile
    GLM_API_KEY       = os.getenv("GLM_API_KEY", "")
    GLM_VISION_MODEL  = os.getenv("GLM_VISION_MODEL", "glm-4v-plus")
    GLM_API_BASE      = os.getenv("GLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4")
    VISION_ENABLED    = os.getenv("VISION_ENABLED", "true").strip().lower() not in ("0", "false", "no")
    VISION_TIMEOUT_SECS = int(os.getenv("VISION_TIMEOUT_SECS", "40"))
    VISION_MAX_RETRIES  = int(os.getenv("VISION_MAX_RETRIES", "1"))
    VISION_MAX_IMAGES   = int(os.getenv("VISION_MAX_IMAGES", "3"))
    VISION_MAX_IMAGE_MB = float(os.getenv("VISION_MAX_IMAGE_MB", "5"))


def resolve_embedding_model_path() -> str:
    """Return the on-disk BGE-M3 directory. Raises if the local model is missing."""
    candidates = []
    env_path = os.getenv("EMBEDDING_MODEL_PATH", "").strip()
    if env_path:
        candidates.append(project_path(env_path))
    candidates.extend([
        os.path.join(PROJECT_ROOT, "models", "BAAI--bge-m3", "snapshots", "master"),
        os.path.join(PROJECT_ROOT, "models", "BAAI", "bge-m3"),
        os.path.join(PROJECT_ROOT, "models", "bge-m3"),
    ])
    weight_names = ("pytorch_model.bin", "model.safetensors", "model.onnx")
    for path in candidates:
        if not os.path.isdir(path):
            continue
        if os.path.isfile(os.path.join(path, "config.json")) and any(
            os.path.isfile(os.path.join(path, name)) for name in weight_names
        ):
            return os.path.normpath(path)
    raise FileNotFoundError(
        "Local BGE-M3 model not found. Put weights under "
        "led-rag-system/models/BAAI--bge-m3/snapshots/master "
        "or set EMBEDDING_MODEL_PATH."
    )


# Global config instance
config = Config()
config.EMBEDDING_MODEL_PATH = resolve_embedding_model_path()

# Force HuggingFace / Transformers to stay offline once the local model is resolved.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
