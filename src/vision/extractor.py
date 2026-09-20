"""
Vision Extractor（计划「第六阶段：实现 Vision Extractor」）。

    智谱原始响应
          ↓
    JSON 提取（容忍 ```json 代码块 / 前后废话）
          ↓
    Schema 校验（枚举、数值范围）
          ↓
    字段标准化（场景 token / 环境枚举）
          ↓
    单位标准化（mm / cm / ft / inch → m；mm 保持 mm）
          ↓
    VisionRequirement

额外职责（计划第二十阶段：性能控制）：
    同一 session + 同一 image hash 只调用一次视觉模型。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional

from src.config import config
from src.vision.client import VisionError, ZhipuVisionClient
from src.vision.prompts import VISION_SYSTEM_PROMPT, build_user_prompt
from src.vision.schema import (
    EXTRA_FIELDS,
    VISION_EXPLICIT,
    VISION_INFERRED,
    VisionField,
    VisionRequirement,
)

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_NUMBER_RE = re.compile(r"(-?\d+(?:[.,]\d+)?)")

_ENVIRONMENT_ALIASES = {
    "indoor": "indoor", "室内": "indoor", "户内": "indoor", "inside": "indoor",
    "outdoor": "outdoor", "室外": "outdoor", "户外": "outdoor", "露天": "outdoor",
    "outside": "outdoor",
    "semi_outdoor": "semi_outdoor", "semi-outdoor": "semi_outdoor",
    "半户外": "semi_outdoor", "半室外": "semi_outdoor",
}

_INSTALLATION_ALIASES = {
    # ── rental：租赁 / 快拆快装（客户口径）──────────────────────────────
    "quick-lock": "rental", "quick lock": "rental", "quicklock": "rental",
    "quick-release": "rental", "quick release": "rental",
    "quick install": "rental", "quick installation": "rental",
    "quick assemble": "rental", "quick-assemble": "rental", "fast install": "rental",
    "fly case": "rental", "flight case": "rental", "road case": "rental",
    "truss": "rental", "on truss": "rental", "truss mount": "rental",
    "portable": "rental", "detachable": "rental", "removable": "rental",
    "temporary": "rental", "temporarily": "rental", "rental": "rental", "rent": "rental",
    "event": "rental", "touring": "rental", "tour rig": "rental",
    "快装": "rental", "快拆": "rental", "快锁": "rental", "快装快拆": "rental",
    "租赁": "rental", "租用": "rental", "临时": "rental", "便携": "rental",
    "桁架": "rental", "航空箱": "rental", "运输箱": "rental",
    "巡演": "rental", "可拆装": "rental", "可移动": "rental",
    # ── fixed：固定安装 / 长期不动 ─────────────────────────────────────
    "fixed installation": "fixed", "permanently installed": "fixed",
    "wall-mounted": "fixed", "wall mounted": "fixed", "wall mount": "fixed",
    "embedded": "fixed", "built-in": "fixed", "built in": "fixed",
    "recessed": "fixed", "flush mounted": "fixed", "long-term": "fixed",
    "steel structure": "fixed", "steel frame": "fixed", "steel beam": "fixed",
    "metal frame": "fixed", "column mounted": "fixed", "pole mounted": "fixed",
    "mast mounted": "fixed", "ground mounted": "fixed", "bolted": "fixed",
    "anchored": "fixed", "screwed to the wall": "fixed",
    "fixed": "fixed", "permanent": "fixed",
    "固定安装": "fixed", "永久固定": "fixed", "固定": "fixed", "固装": "fixed",
    "永久": "fixed", "嵌入": "fixed", "嵌墙": "fixed", "钢结构": "fixed",
    "立柱": "fixed", "长期": "fixed", "壁挂": "fixed", "挂墙": "fixed",
}


def _alias_hit(text: str, key: str) -> bool:
    """别名命中判断：英文按**整词**匹配（允许 "wall-mounted"/"wall mounted"），

    中文按子串匹配。整词匹配可以避免 "installation" 里的 "install"、
    "parent" 里的 "rent" 这类误判。
    """
    if key and all(ord(ch) < 128 for ch in key) and any(ch.isalpha() for ch in key):
        parts = [re.escape(part) for part in re.split(r"[\s\-]+", key) if part]
        pattern = r"(?<![a-z])" + r"[-\s]*".join(parts) + r"(?![a-z])"
        return bool(re.search(pattern, text))
    return bool(key) and key in text

_SPECIAL_ALIASES = {
    "waterproof": "waterproof", "防水": "waterproof", "ip65": "waterproof", "ip66": "waterproof",
    "cob": "cob", "hdr": "hdr", "gob": "gob",
    "flexible": "flexible", "柔性": "flexible", "curved": "flexible", "弧形": "flexible",
    "interaction": "interaction", "touch": "interaction", "触控": "interaction",
}


def _parse_number(text: Any) -> Optional[float]:
    match = _NUMBER_RE.search(str(text or ""))
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", "."))
    except ValueError:  # pragma: no cover - 防御式
        return None


def parse_meters(value: Any) -> Optional[float]:
    """把 "5000 mm" / "5 m" / "16 ft" / "5米" / 5 统一成米。"""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return round(number / 1000, 4) if number > 100 else round(number, 4)
    text = str(value).strip().lower()
    number = _parse_number(text)
    if number is None:
        return None
    # 单位优先（比"数值大小猜单位"可靠）：500 cm 必须按厘米算，不能按 >100 当毫米
    if "ft" in text or "feet" in text or "foot" in text or "英尺" in text:
        number *= 0.3048
    elif "inch" in text or '"' in text or "英寸" in text:
        number *= 0.0254
    elif "cm" in text or "厘米" in text:
        number /= 100
    elif "mm" in text or "毫米" in text:
        number /= 1000
    elif "m" in text or "米" in text:
        pass
    elif number > 100:
        # 没写单位 → 按大小猜：5000 基本是毫米，5 基本是米
        number /= 1000
    return round(number, 4) if number > 0 else None


def parse_mm(value: Any) -> Optional[float]:
    """点间距：统一成毫米。"""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().lower()
        number = _parse_number(text)
        if number is None:
            return None
        if "m" in text.replace("mm", "") and "mm" not in text:
            number *= 1000
    # 真实产品点间距在 0.3~20mm 之间；超出范围说明模型在编数字
    return round(number, 3) if 0.3 <= number <= 20 else None


def parse_nit(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    number = value if isinstance(value, (int, float)) else _parse_number(value)
    if number is None:
        return None
    number = int(float(number))
    return number if 1 <= number <= 20000 else None


def _strip_noise(raw: str) -> str:
    """去掉 markdown 代码块与前后废话，只留 JSON 主体。"""
    text = str(raw or "").strip()
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return text.strip()


def parse_vision_json(raw: str) -> Dict[str, Any]:
    """把模型输出解析成 dict；解析失败抛 VisionError（调用方降级）。"""
    text = _strip_noise(raw)
    if not text:
        raise VisionError("empty vision response", kind="parse")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:  # 容忍结尾多逗号等小毛病
            data = json.loads(re.sub(r",\s*([}\]])", r"\1", text))
        except json.JSONDecodeError as error:
            raise VisionError(f"vision json parse failed: {error}", kind="parse") from error
    if not isinstance(data, dict):
        raise VisionError("vision json is not an object", kind="parse")
    return data


class VisionExtractor:
    """视觉需求提取器：图片 → VisionRequirement。"""

    _cache: Dict[str, VisionRequirement] = {}
    _CACHE_LIMIT = 128

    def __init__(self, client: Optional[ZhipuVisionClient] = None) -> None:
        self.client = client or ZhipuVisionClient()

    # ── 图片哈希 / 缓存（计划第二十阶段）────────────────────────────────
    @staticmethod
    def image_hash(image: Any) -> str:
        if isinstance(image, bytes):
            payload = image
        elif isinstance(image, str) and not image.startswith(("http://", "https://")):
            import base64
            import os

            if os.path.exists(image):
                with open(image, "rb") as handle:
                    payload = handle.read()
            elif image.startswith("data:"):
                payload = image.split(",", 1)[-1].encode("utf-8")
            else:
                payload = base64.b64encode(image.encode("utf-8"))
        else:
            payload = str(image or "").encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def _cache_key(cls, session_id: str, image_hash: str) -> str:
        session = str(session_id or "").strip()
        return f"{session}::{image_hash}" if session and image_hash else ""

    @classmethod
    def cache_lookup(cls, session_id: str, image_hash: str) -> Optional[VisionRequirement]:
        key = cls._cache_key(session_id, image_hash)
        return cls._cache.get(key) if key else None

    @classmethod
    def cache_store(cls, session_id: str, image_hash: str, result: VisionRequirement) -> None:
        key = cls._cache_key(session_id, image_hash)
        if not key:
            return
        cls._cache[key] = result
        while len(cls._cache) > cls._CACHE_LIMIT:
            cls._cache.pop(next(iter(cls._cache)))

    @classmethod
    def clear_session_cache(cls, session_id: str) -> None:
        prefix = f"{str(session_id or '').strip()}::"
        if prefix == "::":
            return
        for key in [k for k in cls._cache if k.startswith(prefix)]:
            cls._cache.pop(key, None)

    # ── 主入口 ──────────────────────────────────────────────────────────
    def extract(
        self,
        image: Any,
        *,
        session_id: str = "",
        customer_text: str = "",
        mime_type: str = "",
        use_cache: bool = True,
    ) -> VisionRequirement:
        """单张图片 → VisionRequirement（失败抛 VisionError）。"""
        digest = self.image_hash(image)
        if use_cache:
            cached = self.cache_lookup(session_id, digest)
            if cached is not None:
                logger.info("Vision cache hit: session=%s hash=%s", session_id, digest[:12])
                return cached

        raw = self.client.analyze_image(
            image,
            build_user_prompt(customer_text),
            system_prompt=VISION_SYSTEM_PROMPT,
            mime_type=mime_type,
        )
        result = self.from_payload(parse_vision_json(raw))
        result.image_hash = digest
        result.model = getattr(self.client, "model", "")
        if use_cache:
            self.cache_store(session_id, digest, result)
        return result

    def extract_many(
        self,
        images: Iterable[Any],
        *,
        session_id: str = "",
        customer_text: str = "",
        max_images: Optional[int] = None,
    ) -> List[VisionRequirement]:
        """多张图片：逐张提取（单张失败不影响其它图片）。"""
        limit = max_images or getattr(config, "VISION_MAX_IMAGES", 3)
        results: List[VisionRequirement] = []
        for index, image in enumerate(list(images or [])[:limit]):
            try:
                results.append(
                    self.extract(image, session_id=session_id, customer_text=customer_text)
                )
            except VisionError as error:
                logger.warning("Vision extract failed (image #%d): %s", index + 1, error)
            except Exception as error:  # pragma: no cover - 防御式
                logger.warning("Vision extract crashed (image #%d): %s", index + 1, error)
        return results

    # ── 解析 / 标准化 ───────────────────────────────────────────────────
    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> VisionRequirement:
        """模型 JSON → VisionRequirement（无效字段直接丢弃，不猜）。"""
        core = {
            "display_type": cls._display_type,
            "environment": cls._environment,
            "purpose": cls._purpose,
            "installation": cls._installation,
            "target_width_m": parse_meters,
            "target_height_m": parse_meters,
            "viewing_distance_m": parse_meters,
            "pixel_pitch_mm": parse_mm,
            "brightness_min_nit": parse_nit,
            "brightness_max_nit": parse_nit,
        }
        data: Dict[str, Any] = {}
        for name, normalizer in core.items():
            if name == "installation":
                field = cls._coerce_installation(
                    payload.get(name), notes=payload.get("notes")
                )
            else:
                field = cls._coerce_field(payload.get(name), normalizer)
            if field is not None:
                data[name] = field

        for name in EXTRA_FIELDS:
            field = cls._coerce_field(payload.get(name), lambda value: str(value).strip())
            if field is not None:
                data[name] = field

        specials = cls._special_requirements(payload.get("special_requirements"))
        if specials:
            data["special_requirements"] = specials

        notes = payload.get("notes")
        if isinstance(notes, str) and notes.strip():
            data["notes"] = notes.strip()[:300]
        return VisionRequirement(**data)

    @staticmethod
    def _coerce_field(raw: Any, normalizer) -> Optional[VisionField]:
        """把模型给的一个字段标准化；无效则返回 None（丢弃，不猜）。"""
        if raw in (None, "", [], {}):
            return None

        if isinstance(raw, dict):
            value = raw.get("value")
            source = str(raw.get("source") or "").strip().lower()
            confidence = raw.get("confidence")
            evidence = str(raw.get("evidence") or "").strip()[:200]
        else:
            # 模型没按格式返回（裸值）→ 一律按"推断"处理：
            # 不能因为格式问题就把"猜的"当成"看到的"。
            value = raw
            source = VISION_INFERRED
            confidence = 0.5
            evidence = ""

        normalized = normalizer(value)
        if normalized in (None, "", [], {}):
            return None

        if source not in (VISION_EXPLICIT, VISION_INFERRED):
            # 来源缺失 / 写错 → 保守按推断处理（宁可再问一次，也不把猜测当确认）
            source = VISION_INFERRED
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            conf = 0.0
        return VisionField(
            value=normalized,
            confidence=max(0.0, min(1.0, conf)),
            source=source,  # type: ignore[arg-type]
            evidence=evidence,
        )

    @classmethod
    def _coerce_installation(cls, raw: Any, notes: Any = None) -> Optional[VisionField]:
        """installation 专用：模型把线索原文当值返回时，再从 evidence 里兜底解析一次。

        实测风险：模型可能返回
            {"value": "steel structure below the screen", "evidence": "steel structure"}
        或者 value 写成 "快装快拆" 这类线索词 —— 只要 evidence / value 里能识别出
        fixed / rental 线索，就正常落库；两边都认不出才丢弃（交给客户确认）。

        另外两条保护：
          - 模型的 notes（一句话描述）里写了安装结构时，也拿来判断；
          - confidence 很低的 "vision_explicit" 降级为 vision_inferred ——
            图片判断本来就可能看错，低置信时让 Gate 再问客户一次更安全。
        """
        field = cls._coerce_field(raw, cls._installation)
        if field is not None:
            if field.source == VISION_EXPLICIT and field.confidence < 0.6:
                logger.info(
                    "Vision installation downgraded to inferred (confidence=%.2f): %s",
                    field.confidence, field.value,
                )
                return VisionField(
                    value=field.value,
                    confidence=field.confidence,
                    source=VISION_INFERRED,  # type: ignore[arg-type]
                    evidence=field.evidence,
                )
            return field

        candidates: List[Any] = []
        if isinstance(raw, dict):
            candidates.extend([raw.get("evidence"), raw.get("value")])
        candidates.append(notes)
        for candidate in candidates:
            canonical = cls._installation(candidate)
            if canonical:
                logger.info(
                    "Vision installation resolved from evidence: %r -> %s",
                    candidate, canonical,
                )
                base: Dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
                base["value"] = canonical
                if candidate is notes:
                    # 从"整体描述"里推出来的 → 只能算推断，绝不能当成图片明确可见
                    base["source"] = VISION_INFERRED
                    base["confidence"] = 0.5
                    base["evidence"] = str(notes)[:200]
                return cls._coerce_field(base, cls._installation)
        return None

    @staticmethod
    def _display_type(value: Any) -> Optional[str]:
        text = str(value or "").strip().upper()
        for candidate in ("LED", "LCD", "IFP"):
            if candidate in text:
                return candidate
        if "一体机" in text or "交互平板" in text:
            return "IFP"
        return None

    @staticmethod
    def _environment(value: Any) -> Optional[str]:
        text = str(value or "").strip().lower()
        if text in _ENVIRONMENT_ALIASES:
            return _ENVIRONMENT_ALIASES[text]
        for key, canonical in _ENVIRONMENT_ALIASES.items():
            if key and key in text:
                return canonical
        return None

    @staticmethod
    def _installation(value: Any) -> Optional[str]:
        """安装方式标准化：**最长的键优先**，避免 "installation" 里的 "install" 之类误判。

        客户口径：fixed = 永久固定不动；rental = 快装快拆（可整体拆走）。
        """
        text = str(value or "").strip().lower()
        if not text:
            return None
        for key in sorted(_INSTALLATION_ALIASES, key=len, reverse=True):
            if _alias_hit(text, key):
                return _INSTALLATION_ALIASES[key]
        return None

    @staticmethod
    def _purpose(value: Any) -> Optional[str]:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            from src.core.purpose_normalizer import PurposeNormalizer

            canonical, _ = PurposeNormalizer().normalize(text)
            if canonical:
                return canonical.value
        except Exception:  # pragma: no cover - 防御式
            pass
        try:
            from src.rag.query_understanding import _detect_purpose

            detected = _detect_purpose(text.lower())
            if detected:
                return detected
        except Exception:  # pragma: no cover - 防御式
            pass
        return None

    @staticmethod
    def _special_requirements(raw: Any) -> List[str]:
        if not raw:
            return []
        values = raw if isinstance(raw, list) else [raw]
        result: List[str] = []
        for item in values:
            text = str(item or "").strip().lower()
            if not text:
                continue
            canonical = _SPECIAL_ALIASES.get(text)
            if not canonical:
                for key, token in _SPECIAL_ALIASES.items():
                    if key in text:
                        canonical = token
                        break
            if canonical and canonical not in result:
                result.append(canonical)
        return result


def merge_vision_results(results: Iterable[VisionRequirement]) -> VisionRequirement:
    """多张图片的结果合并：显式证据优先于推断；先出现的显式值优先。"""
    merged = VisionRequirement()
    for result in results or []:
        if result is None:
            continue
        for name in list(VisionRequirement.model_fields):
            field = getattr(result, name, None)
            if not isinstance(field, VisionField) or field.value in (None, "", [], {}):
                continue
            current = getattr(merged, name, None)
            if current is None or current.value in (None, "", [], {}):
                setattr(merged, name, field)
            elif current.is_inferred and field.is_explicit:
                setattr(merged, name, field)
        for token in result.special_requirements:
            if token not in merged.special_requirements:
                merged.special_requirements.append(token)
        if result.notes and result.notes not in merged.notes:
            merged.notes = (merged.notes + " " + result.notes).strip()[:300]
    return merged


_extractor: Optional[VisionExtractor] = None


def get_vision_extractor() -> VisionExtractor:
    """进程内复用同一个 Vision Extractor（含图片缓存）。"""
    global _extractor
    if _extractor is None:
        _extractor = VisionExtractor()
    return _extractor


__all__ = [
    "VisionExtractor",
    "get_vision_extractor",
    "merge_vision_results",
    "parse_meters",
    "parse_mm",
    "parse_nit",
    "parse_vision_json",
]
