"""结构化需求判定（旧路由机已删除 —— 整改计划 §六/§十七）。

原来这个模块里还有一整套"查询复杂度路由"：
``QueryRoute`` / ``RoutingDecision`` / ``classify_complexity`` 以及它的一堆
关键词辅助函数（``_has_scene_intent`` / ``_is_param_only`` /
``_extract_parameter_constraints`` …）。那套路由**在生产路径上零调用**
（旧的三层 fast/normal/agent 路由早已删除），里面却还有"用关键词把
display_type 判成 LCD / IFP"的第二套类型判定 —— 与整改计划 §八
"IFP 判断必须统一（只由 LCD Decision Center 决定）"直接冲突，已整段删除。

现在只保留一个纯辅助：``has_structured_requirement`` —— 判断需求里是否已经
带有场景级上下文（purpose / environment / installation / usage）。
生产调用方：``src/agents/solution/runner.py``（决定是否走结构化产品查询）。
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def has_structured_requirement(requirements: dict | None) -> bool:
    """结构化需求里是否已经带有"场景级"上下文（Phase 6）。

    这是 Router 的第一判据：只要上游（RequirementExtractor / RequirementProfile）
    已经识别出 purpose / environment / installation / usage，就说明这是"有场景的
    推荐类查询"，不必再靠关键词表去猜 —— 关键词表降级为 fallback。
    """
    if not requirements:
        return False
    for key in ("purpose", "usage", "environment", "location_type", "installation"):
        value = requirements.get(key)
        if value not in (None, "", [], {}):
            return True
    return False



__all__ = ["has_structured_requirement"]
