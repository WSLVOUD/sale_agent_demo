"""
数据加载模块
处理从文本文件加载产品数据
"""
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from typing import List, Optional, Dict, Any
import logging
import re
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

ENVIRONMENT_METADATA_VERSION = 4

KNOWLEDGE_SOURCES_FILENAME_RE = re.compile(r"\.txt$|\.md$|\.yaml$|\.yml$", re.IGNORECASE)

# ── Product category classification ──────────────────────────────────────────
_DISPLAY_PATTERNS = (
    re.compile(r"display|screen|panel|wall|拼接|屏|电视|signage", re.IGNORECASE),
    re.compile(r"oled|led\s|led$|lcd\s|lcd$|cob\s|all-in-one|ultra.?hd", re.IGNORECASE),
    re.compile(r"广告机", re.IGNORECASE),
)

_PIXEL_PITCH_RE = re.compile(r"pixel\s*pitch\s*(?P<pitch>\d+(?:\.\d+)?)\s*mm", re.IGNORECASE)
_BRIGHTNESS_RE = re.compile(r"brightness\s*(?:=\s*)?(?P<value>\d{2,6})\s*(?:nit|cd|nits|cds)?", re.IGNORECASE)
_SERIES_HEADER_RE = re.compile(r"\b(TW\d{2})\s*[-‐‑‒–—]\s*(OD|IR(?:HD)?|HOD|COB|3216)\b", re.IGNORECASE)

_NON_DISPLAY_PATTERNS = (
    re.compile(r"mount|bracket|stand|支架|吊架|落地架|机架|mounting|壁挂|立式", re.IGNORECASE),
    re.compile(r"module|模块|ops\b|扩展|插卡", re.IGNORECASE),
    re.compile(r"cms|content management|management system|platform|software|软件|系统\b", re.IGNORECASE),
    re.compile(r"pop-out|hydraulic|配件|accessory", re.IGNORECASE),
    re.compile(r"cleaner|维护|inspection", re.IGNORECASE),
)


def _classify_product_category(product_name: str) -> str:
    """Classify a product by its name into 'display' | 'mount' | 'module' | 'software'."""
    name = product_name.lower()

    mount_patterns = [
        re.compile(r"\b(floor stand|moving bracket|mobile bracket|support stand|standing bracket|hanging bracket|bracket|stand support|wall mount|wall-mounted|wall mountings)\b", re.IGNORECASE),
        re.compile(r"\b(支架|吊架|落地架|机架|底座|支撑|挂架|壁挂)\b", re.IGNORECASE),
    ]
    if any(p.search(name) for p in mount_patterns):
        if re.search(r"\b(digital display|interactive flat panel|led display|lcd display|screen|monitor|panel|video wall|display wall)\b", name):
            if not re.search(r"\b(floor stand|moving bracket|mobile bracket|support stand|standing bracket|hanging bracket|bracket|支架|吊架|落地架|机架|底座|支撑|挂架)\b", name):
                return "display"
        return "mount"

    display_patterns = [
        re.compile(r"\b(display|screen|panel|video wall|digital signage|signage|monitor|lcd|led|oled|tv|拼接|屏|广告机|电视|液晶|显示器)\b", re.IGNORECASE),
        re.compile(r"\b(all-in-one|ultra.?hd|cob|interactive flat panel|会议一体机|交互平板)\b", re.IGNORECASE),
        re.compile(r"\b(TW\d+|T\d{2}Omni|HG\d+)\b", re.IGNORECASE),
    ]
    if any(p.search(name) for p in display_patterns):
        return "display"

    return "display"


def _extract_product_name(text: str) -> str:
    """Pull the product name from the first 'Product：' / 'Product:' line."""
    normalised = text.replace("\uff1a", ":")
    m = re.search(r"(?mi)^Product:\s*(.+?)$", normalised, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _extract_numeric_specs(chunk_text: str) -> Dict[str, Any]:
    """Pull brightness / pitch / rental flags out of a chunk's text."""
    first_header = None
    for m in _SERIES_HEADER_RE.finditer(chunk_text):
        if first_header is None:
            first_header = m.group(2).upper()
        else:
            break

    is_rental: Optional[bool] = None
    if first_header:
        if "IR" in first_header:
            is_rental = True
        else:
            is_rental = False
    elif re.search(r"\brental\b|租赁|快装|快拆", chunk_text, re.IGNORECASE):
        is_rental = True
    elif re.search(r"\bfixed\b|固定|户外固定|outdoor fixed", chunk_text, re.IGNORECASE):
        is_rental = False

    brightness_values: List[int] = []
    pitches: List[float] = []
    MODEL_CODE_RE = re.compile(r"(TW\d{2})[\W]+?(OD|IR(?:HD)?|HOD|COB|3216)", re.IGNORECASE)

    for line in chunk_text.splitlines():
        line = line.strip()
        model_match = MODEL_CODE_RE.search(line)
        if first_header and model_match:
            model_series = model_match.group(2).upper()
            if first_header.startswith("IR"):
                if not model_series.startswith("IR"):
                    continue
            elif model_series != first_header:
                continue

        for pm in _PIXEL_PITCH_RE.finditer(line):
            try:
                p = float(pm.group("pitch"))
                if 0.1 <= p <= 50.0:
                    pitches.append(p)
            except ValueError:
                pass

        for bm in _BRIGHTNESS_RE.finditer(line):
            try:
                v = int(bm.group("value"))
                if 50 <= v <= 30000:
                    brightness_values.append(v)
            except ValueError:
                pass

    specs: Dict[str, Any] = {}
    if brightness_values:
        specs["brightness_min_cd"] = min(brightness_values)
        specs["brightness_max_cd"] = max(brightness_values)
    if pitches:
        specs["pixel_pitch_min_mm"] = min(pitches)
        specs["pixel_pitch_max_mm"] = max(pitches)
    if is_rental is not None:
        specs["is_rental"] = is_rental

    return specs


def apply_environment_metadata(chunk: Document, product_name: str = "") -> None:
    """Infer a chunk's supported environment and product category."""
    chunk_text = chunk.page_content.lower()

    has_outdoor = bool(re.search(r"\boutdoor\b|户外", chunk_text))
    has_indoor = bool(re.search(r"\bindoor\b|室内", chunk_text))

    parent_outdoor = chunk.metadata.get("outdoor")
    parent_indoor = chunk.metadata.get("indoor")
    if parent_outdoor is True and has_outdoor is False and has_indoor is False:
        has_outdoor = True
    if parent_indoor is True and has_outdoor is False and has_indoor is False:
        has_indoor = True

    if has_outdoor and not has_indoor:
        chunk.metadata["indoor"] = False
        chunk.metadata["outdoor"] = True
    elif has_indoor and not has_outdoor:
        chunk.metadata["indoor"] = True
        chunk.metadata["outdoor"] = False
    elif parent_outdoor is True and parent_indoor is not True:
        chunk.metadata["indoor"] = False
        chunk.metadata["outdoor"] = True
    elif parent_indoor is True and parent_outdoor is not True:
        chunk.metadata["indoor"] = True
        chunk.metadata["outdoor"] = False

    numeric_specs = _extract_numeric_specs(chunk.page_content)
    if numeric_specs:
        chunk.metadata.update(numeric_specs)

    if product_name:
        category = _classify_product_category(product_name)
    else:
        category = _classify_product_category(chunk.page_content[:300])

    chunk.metadata.update({
        "environment_metadata_version": ENVIRONMENT_METADATA_VERSION,
        "product_category": category,
    })


def load_products(file_path: str) -> List[Document]:
    """Load product data from a text file."""
    logger.info(f"Loading products from {file_path}")
    try:
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        raise


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents by 'Product Name:' header — one chunk per product, no fixed-size split."""
    logger.info(f"Splitting {len(documents)} documents by Product Name")

    PRODUCT_HEADER_RE = re.compile(r"(?=^Product\s+Name[：:])", re.MULTILINE)

    all_chunks: List[Document] = []
    for doc in documents:
        # Split into blocks at each "Product Name:" line
        blocks = PRODUCT_HEADER_RE.split(doc.page_content)
        if len(blocks) == 1:
            # No Product Name header found — keep whole doc as one chunk
            product_name = _extract_product_name(doc.page_content)
            chunk = Document(page_content=doc.page_content.strip(), metadata=dict(doc.metadata))
            apply_environment_metadata(chunk, product_name=product_name)
            all_chunks.append(chunk)
            continue

        for block in blocks:
            if not block.strip():
                continue
            product_name = _extract_product_name(block)
            chunk = Document(page_content=block.strip(), metadata=dict(doc.metadata))
            apply_environment_metadata(chunk, product_name=product_name)
            all_chunks.append(chunk)

    logger.info(f"Created {len(all_chunks)} product chunks")
    return all_chunks


def load_and_process_products(file_path: str = None, use_sample: bool = False) -> List[Document]:
    """Load and process product documents."""
    if use_sample:
        from .loader import create_sample_products
        documents = create_sample_products()
    elif file_path:
        documents = load_products(file_path)
        documents = split_documents(documents)
    else:
        raise ValueError("Either file_path or use_sample must be provided")
    return documents


def _infer_display_type_from_filename(filename: str) -> str:
    """Infer display_type from the filename."""
    name = filename.lower()
    if "ifp" in name:
        return "IFP"
    if "lcd" in name:
        return "LCD"
    if "led" in name:
        return "LED"
    return "LED"


def load_all_product_files(data_dir: str) -> List[Document]:
    """Automatically scan ``data_dir`` for all ``.txt`` product files."""
    root = Path(data_dir)
    if not root.exists():
        logger.warning("Product data directory does not exist: %s", root)
        return []

    all_documents: List[Document] = []
    seen: set[str] = set()

    for path in sorted(root.iterdir()):
        if not path.is_file() or not path.suffix.lower() == ".txt":
            continue
        if path.name in seen:
            continue
        seen.add(path.name)

        display_type = _infer_display_type_from_filename(path.name)
        logger.info("Loading product file: %s (display_type=%s)", path.name, display_type)

        try:
            docs = load_products(str(path))
            for doc in docs:
                doc.metadata["display_type"] = display_type
                if display_type == "IFP":
                    doc.metadata["indoor"] = True
                    doc.metadata["outdoor"] = False
                elif display_type == "LCD":
                    doc.metadata["indoor"] = True
                    doc.metadata["outdoor"] = False
                else:
                    doc.metadata["indoor"] = True
                    doc.metadata["outdoor"] = True

            chunks = split_documents(docs)
            for chunk in chunks:
                chunk.metadata["chunk_id"] = str(uuid.uuid4())
            all_documents.extend(chunks)
            logger.info("Loaded %d chunks from %s", len(chunks), path.name)
        except Exception as exc:
            logger.error("Failed to load product file %s: %s", path, exc)

    logger.info("Auto-loaded %d total chunks from %d product files", len(all_documents), len(seen))
    return all_documents


def _knowledge_topic_from_path(path: Path) -> str:
    """Derive a knowledge topic from the file path."""
    parts = [p.strip().lower() for p in path.parts if p.strip()]
    for index, part in enumerate(parts):
        if part == "knowledge" and index + 1 < len(parts):
            next_part = parts[index + 1]
            if next_part and next_part != path.name.lower():
                return next_part

    stem = path.stem.lower()
    stem = re.sub(r"[\s_\-]+", "_", stem)
    return stem


def load_knowledge_base(knowledge_dir: str) -> List[Document]:
    """Load the domain knowledge base."""
    root = Path(knowledge_dir)
    if not root.exists():
        logger.warning("Knowledge directory does not exist: %s", root)
        return []

    documents: List[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not KNOWLEDGE_SOURCES_FILENAME_RE.search(path.name):
            continue
        try:
            loader = TextLoader(str(path), encoding="utf-8")
            loaded = loader.load()
        except Exception as exc:
            logger.error("Failed to load knowledge file %s: %s", path, exc)
            continue

        topic = _knowledge_topic_from_path(path)
        for doc in loaded:
            doc.metadata = {
                **doc.metadata,
                "source": "knowledge",
                "topic": topic,
                "category": "selection_guide",
                "filename": path.name,
                "knowledge_path": str(path),
            }
            documents.append(doc)
        logger.info("Loaded knowledge file: %s topic=%s", path.name, topic)

    logger.info("Loaded %d knowledge documents", len(documents))
    return documents
