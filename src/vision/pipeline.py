"""v2.3.1 Phase 2：Vision 接入链路的两个纯函数（从 orchestrator 迁出）。

    extract_vision_for_turn()（vision 包内） → _merge_vision_into_stored_profile()

只负责"把图片结果并进已存的需求档案"，不含任何推荐/回复逻辑。逻辑一行未改。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..config import config

logger = logging.getLogger(__name__)


def _vision_enabled() -> bool:
    """视觉开关（默认开；配置里 VISION_ENABLED=false 可整体关掉）。"""
    try:
        return bool(getattr(_config, "VISION_ENABLED", True))
    except Exception:  # pragma: no cover - 防御式
        return True


def _merge_vision_into_stored_profile(
    memory_store: Any,
    session_id: str,
    vision_results: List[Any],
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """把图片识别出的需求并进"memory 里的需求档案"。

    为什么放在 Orchestrator 而不是 Sales Agent：
      - 需求档案是 Sales Agent 的输入，图片在 Sales 之前就要合并好；
      - Sales Agent 保持不动（计划第十六阶段：不新建 Vision Sales Agent）。

    合并结果写回 memory；下一轮 Sales Agent 从 memory 读取时即可看到图片信息。
    任何异常都不往上抛（图片不能拖垮主流程）。
    """
    if not vision_results or not memory_store:
        return
    try:
        from ..models.requirement import RequirementProfile
        from . import apply_vision_to_profile

        stored = None
        if hasattr(memory_store, "get_requirement_profile"):
            stored = memory_store.get_requirement_profile(session_id)
        profile = None
        if isinstance(stored, RequirementProfile):
            profile = stored
        elif isinstance(stored, dict) and stored:
            try:
                profile = RequirementProfile.model_validate(stored)
            except Exception:  # pragma: no cover - 防御式
                profile = None
        if profile is None:
            profile = RequirementProfile()

        for result in vision_results:
            profile, stats = apply_vision_to_profile(profile, result)
            logger.info(
                "[%s] Vision merged into profile: fields=%s conflicts=%s size_hint=%s",
                session_id, stats.get("merged_fields"), stats.get("conflict_count"),
                stats.get("size_hint"),
            )

        if hasattr(memory_store, "set_requirement_profile"):
            memory_store.set_requirement_profile(session_id, profile)
        if isinstance(metrics, dict):
            metrics["merge_success"] = True
        # 旧的 requirements 视图由 Sales Agent 下一轮从 Profile 重建
    except Exception as exc:  # pragma: no cover - 防御式
        logger.warning("[%s] Vision merge failed: %s", session_id, exc)
