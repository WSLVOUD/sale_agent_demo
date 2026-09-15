"""
Unified Requirement Extractor - 统一需求提取器

职责：
1. 从客户自然语言提取结构化需求
2. 集成 LLM 语义理解 + 规则确定性解析 + 数字单位转换
3. 返回规范化的 RequirementProfile
4. 记录每个字段的来源和置信度

设计原则：
- LLM 负责语义理解（purpose, environment 的语义映射）
- Regex/Parser 负责数字确定性解析（size, distance, pixel_pitch）
- Rule Engine 负责验证和约束
- 禁止 LLM 直接决定客户没有明确说过的参数
"""

import logging
from typing import Any, Dict, Optional, List
from src.models.requirement import RequirementProfile
from src.rag.query_understanding import (
    extract_slots,
    detect_language,
    language_name,
)
from src.core.llm import get_llm
from src.core.purpose_normalizer import PurposeNormalizer, CanonicalPurpose
from src.core.environment_installation_resolver import (
    EnvironmentResolver,
    InstallationResolver,
    ConflictDetector,
)

logger = logging.getLogger(__name__)


class RequirementExtractor:
    """统一需求提取器 - 单入口处理所有客户需求理解"""

    def __init__(self):
        self.llm = get_llm()
        self.language = "en"
        self.purpose_normalizer = PurposeNormalizer()

    def extract(
        self,
        message: str,
        previous_profile: Optional[RequirementProfile] = None,
        semantic_override: Optional[Dict[str, Any]] = None,
        use_llm: bool = True,
    ) -> RequirementProfile:
        """
        从客户消息中提取需求。

        Args:
            message: 客户的自然语言消息
            previous_profile: 前一轮的需求档案（用于继承和合并）
            semantic_override: 上游（Sales Agent 的同一次 LLM 调用）已经产出的
                语义结果；给了它就不再单独调 LLM（Phase 13：同一轮只理解一次）
            use_llm: 是否允许调用 LLM 做语义补充

        Returns:
            RequirementProfile - 结构化需求档案
        """
        if not message or not str(message).strip():
            return previous_profile or RequirementProfile()

        # Step 1: 检测语言
        self.language = detect_language(message)
        logger.info(f"Detected language: {self.language}")

        # Step 2: 规则确定性解析（高确定性，无需 LLM）
        rule_slots = extract_slots(message)
        logger.debug(f"Rule extraction: {rule_slots}")

        # Step 2b: canonical phrase fast path（计划 Phase 6.2 / 22）
        # 关键词表没覆盖的说法（shopping center / commercial complex / football venue…）
        # 也先在短语表里找一次 —— 命中就不需要 LLM，这就是"关键词表降级为快速路径"。
        if not rule_slots.get("purpose"):
            canonical, confidence = self.purpose_normalizer.normalize(message)
            if canonical:
                rule_slots["purpose"] = canonical.value
                rule_slots.setdefault("_explicit_keys", set()).add("purpose")
                logger.info(
                    "Purpose fast path: %r → %s (confidence %.2f)",
                    message[:60], canonical.value, confidence,
                )

        # Step 3: LLM 语义理解（低确定性场景、歧义解析）
        if semantic_override:
            llm_result = {k: v for k, v in semantic_override.items() if v is not None}
            self._cache_semantic(message, llm_result)
            logger.info("RequirementExtractor: 复用上游语义结果（不再调用 LLM）")
        elif not use_llm:
            llm_result = self._cached_semantic(message) or {}
        else:
            llm_result = self._llm_semantic_extract(message, rule_slots)
        logger.debug(f"LLM extraction: {llm_result}")

        # Step 4: 合并结果（规则优先，LLM 作补充）
        merged_slots = self._merge_extractions(rule_slots, llm_result, message)
        logger.debug(f"Merged slots: {merged_slots}")

        # Step 5: Purpose 标准化
        if "purpose" in merged_slots:
            canonical_purpose, confidence = self.purpose_normalizer.normalize(
                merged_slots["purpose"]
            )
            if canonical_purpose:
                merged_slots["purpose"] = canonical_purpose.value
                logger.info(f"Normalized purpose: {canonical_purpose.value} (confidence: {confidence})")

        # Step 6: Environment 统一解析
        explicit_env = merged_slots.get("environment")
        purpose_enum = None
        if "purpose" in merged_slots:
            try:
                purpose_enum = CanonicalPurpose(merged_slots["purpose"])
            except ValueError:
                pass

        resolved_env, env_priority = EnvironmentResolver.resolve(
            explicit_environment=explicit_env,
            purpose=purpose_enum,
        )
        if resolved_env:
            merged_slots["environment"] = resolved_env
            # 标记来源
            if env_priority == EnvironmentResolver.PRIORITY_EXPLICIT:
                merged_slots.setdefault("_explicit_keys", set()).add("environment")
            elif env_priority == EnvironmentResolver.PRIORITY_SCENARIO:
                merged_slots.setdefault("_scenario_derived", []).append("environment")
            elif env_priority == EnvironmentResolver.PRIORITY_DEFAULT:
                merged_slots.setdefault("_default_slots", []).append("environment")

        # Step 7: Installation 统一解析
        explicit_inst = merged_slots.get("installation")
        resolved_inst, inst_priority = InstallationResolver.resolve(
            explicit_installation=explicit_inst,
            purpose=purpose_enum,
        )
        if resolved_inst:
            merged_slots["installation"] = resolved_inst
            # 标记来源
            if inst_priority == InstallationResolver.PRIORITY_EXPLICIT:
                merged_slots.setdefault("_explicit_keys", set()).add("installation")
            elif inst_priority == InstallationResolver.PRIORITY_SCENARIO:
                merged_slots.setdefault("_default_slots", []).append("installation")

        # Step 8: 构建 RequirementProfile（确定来源标记）
        # 关键：规则解析器从**客户原话**里读出来的字段（尺寸/视距/点间距/明确关键词）
        # 本身就是客户说过的，必须算 explicit；只有它自己标了 inferred/default/
        # scenario_derived 的才算"系统推断"。
        rule_explicit = {k for k in rule_slots if not str(k).startswith("_")}
        for marker in ("_inferred_slots", "_default_slots", "_scenario_derived"):
            rule_explicit -= {str(x) for x in (rule_slots.get(marker) or [])}
        explicit_keys = set(merged_slots.pop("_explicit_keys", set())) | rule_explicit
        semantic_conflicts = merged_slots.pop("_semantic_conflicts", [])
        profile = RequirementProfile.from_slots(
            merged_slots,
            explicit_keys=explicit_keys,
        )

        # Step 9: 与前一轮合并（保持跨轮一致性）
        if previous_profile:
            profile = previous_profile.merge(profile)
            logger.info(f"Merged with previous profile")

        # Step 10: 冲突检测
        conflicts = self._detect_conflicts(
            profile, purpose_enum, semantic_conflicts=semantic_conflicts
        )
        if conflicts:
            logger.warning(f"Conflicts detected: {conflicts}")
        # 冲突以「本轮重新检测」为准，消解后自动清空
        profile.conflicts = conflicts

        return profile

    # ── Phase 13：同一轮对话内语义结果缓存（避免 Sales/Solution 重复调 LLM）──
    _semantic_cache: Dict[str, Dict[str, Any]] = {}
    _SEMANTIC_CACHE_LIMIT = 32

    @staticmethod
    def _cache_key(message: str) -> str:
        return " ".join(str(message or "").lower().split())

    def _cache_semantic(self, message: str, semantic: Dict[str, Any]) -> None:
        key = self._cache_key(message)
        if not key:
            return
        self._semantic_cache[key] = dict(semantic or {})
        while len(self._semantic_cache) > self._SEMANTIC_CACHE_LIMIT:
            self._semantic_cache.pop(next(iter(self._semantic_cache)))

    def _cached_semantic(self, message: str) -> Optional[Dict[str, Any]]:
        return self._semantic_cache.get(self._cache_key(message))

    def _llm_semantic_extract(
        self,
        message: str,
        rule_slots: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        使用 LLM 进行语义理解。

        仅当规则无法确定时才调用 LLM，以避免过度依赖。
        """
        # 检查规则是否已经提取了关键信息
        has_purpose = "purpose" in rule_slots
        has_environment = "environment" in rule_slots
        has_display_type = "display_type" in rule_slots

        # 如果关键字段都已通过规则提取，跳过 LLM
        if has_purpose and has_environment and has_display_type:
            logger.debug("Rule extraction complete, skipping LLM")
            return {}

        # Phase 13：同一轮已经理解过这条消息 → 直接复用，不再调 LLM
        cached = self._cached_semantic(message)
        if cached is not None:
            logger.debug("Semantic extraction cache hit, skipping LLM")
            return dict(cached)

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            prompt = SystemMessage(
                content="""你是需求理解助手。从客户消息中提取关键信息。

返回 JSON（只返回有信息的字段）：
{
  "purpose": "场景类别（conference/retail/stadium/advertising/concert/stage/wedding/church 等，没有返回 null）",
  "environment": "indoor/outdoor/semi_outdoor（如果客户没明确说，返回 null）",
  "display_type": "LED/LCD/IFP（优先从客户原话提取）",
  "installation": "fixed/rental（如果客户没明确说，返回 null）"
}

重要规则：
1. 只返回客户**明确说过**的信息
2. 不要自动推断客户没说的参数
3. 例如：说"会议室"不能自动推断"室内"，除非明确说了
4. purpose 必须映射到标准 token
5. 只返回 JSON，不要任何解释"""
            )

            human = HumanMessage(
                content=f"客户消息：{message}\n\n已有规则提取：{rule_slots}"
            )

            response = self.llm.invoke([prompt, human])
            import json

            try:
                result = json.loads(response.content.strip())
                # 过滤 null 值
                semantic = {k: v for k, v in result.items() if v is not None}
                self._cache_semantic(message, semantic)
                return semantic
            except json.JSONDecodeError:
                logger.warning(f"LLM response not valid JSON: {response.content}")
                return {}

        except Exception as e:
            logger.error(f"LLM semantic extraction failed: {e}")
            return {}

    def _merge_extractions(
        self,
        rule_slots: Dict[str, Any],
        llm_result: Dict[str, Any],
        message: str = "",
    ) -> Dict[str, Any]:
        """
        合并规则提取和 LLM 提取的结果。

        优先级：
        1. 规则高确定性字段（数字、单位、明确关键词）
        2. LLM 补充的字段
        3. 冲突时规则优先

        【防幻觉】LLM 给出的 environment / installation 必须带"客户原话证据"，
        证据必须是客户消息里真实存在的片段，否则丢弃 —— 避免 LLM 替客户猜参数。
        """
        merged = dict(rule_slots)
        semantic_conflicts: List[str] = []

        for key, value in llm_result.items():
            if key.endswith("_evidence"):
                continue
            if key in merged:
                # 规则已提取 → 规则优先；但如果两边都明确且互相矛盾，记录冲突（Phase 18）
                if (
                    key in ("environment", "installation")
                    and merged.get(key)
                    and value
                    and merged.get(key) != value
                ):
                    semantic_conflicts.append(
                        f"{key}_conflict: rule={merged.get(key)} vs semantic={value}"
                    )
                    logger.warning("Semantic conflict on %s: %s vs %s", key, merged.get(key), value)
                continue
            if key in ("environment", "installation"):
                evidence = llm_result.get(f"{key}_evidence")
                if not self._evidence_ok(message, evidence):
                    logger.info(
                        "Dropping LLM %s=%r — 缺少客户原话证据", key, value
                    )
                    continue
            merged[key] = value
            if key in ("purpose", "environment", "installation", "display_type"):
                merged.setdefault("_explicit_keys", set()).add(key)

        if semantic_conflicts:
            merged.setdefault("_semantic_conflicts", []).extend(semantic_conflicts)
        return merged

    @staticmethod
    def _evidence_ok(message: str, evidence: Any) -> bool:
        """证据片段必须真实出现在客户消息里（大小写不敏感）。"""
        text = str(message or "").lower()
        fragment = str(evidence or "").strip().lower()
        return bool(fragment) and fragment in text

    def _detect_conflicts(
        self,
        profile: RequirementProfile,
        purpose: Optional[CanonicalPurpose],
        semantic_conflicts: Optional[List[str]] = None,
    ) -> List[str]:
        """
        检测需求中的冲突。

        例如：
        - 同一轮里规则解析与语义提取给出矛盾的 environment / installation
        - display_type 为 IFP 但 environment 为 outdoor
        """
        conflicts = list(semantic_conflicts or [])

        # 检查 IFP 与 outdoor 冲突
        if profile.display_type == "IFP" and profile.environment == "outdoor":
            conflicts.append("IFP_outdoor_conflict")
            logger.warning("IFP cannot be used outdoors")

        return conflicts


# 全局单例
_extractor = None


def get_requirement_extractor() -> RequirementExtractor:
    """获取全局 RequirementExtractor 实例"""
    global _extractor
    if _extractor is None:
        _extractor = RequirementExtractor()
    return _extractor
