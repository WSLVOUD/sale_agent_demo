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
    ) -> RequirementProfile:
        """
        从客户消息中提取需求。

        Args:
            message: 客户的自然语言消息
            previous_profile: 前一轮的需求档案（用于继承和合并）

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

        # Step 3: LLM 语义理解（低确定性场景、歧义解析）
        llm_result = self._llm_semantic_extract(message, rule_slots)
        logger.debug(f"LLM extraction: {llm_result}")

        # Step 4: 合并结果（规则优先，LLM 作补充）
        merged_slots = self._merge_extractions(rule_slots, llm_result)
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
        explicit_keys = merged_slots.pop("_explicit_keys", set())
        profile = RequirementProfile.from_slots(
            merged_slots,
            explicit_keys=explicit_keys,
        )

        # Step 9: 与前一轮合并（保持跨轮一致性）
        if previous_profile:
            profile = previous_profile.merge(profile)
            logger.info(f"Merged with previous profile")

        # Step 10: 冲突检测
        conflicts = self._detect_conflicts(profile, purpose_enum)
        if conflicts:
            logger.warning(f"Conflicts detected: {conflicts}")
            # TODO: 将冲突信息记录到 profile 中

        return profile

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
                return {k: v for k, v in result.items() if v is not None}
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
    ) -> Dict[str, Any]:
        """
        合并规则提取和 LLM 提取的结果。

        优先级：
        1. 规则高确定性字段（数字、单位、明确关键词）
        2. LLM 补充的字段
        3. 冲突时规则优先
        """
        merged = dict(rule_slots)

        for key, value in llm_result.items():
            if key not in merged:
                # LLM 提供的新字段
                merged[key] = value
            # 如果规则已提取该字段，保持规则结果（规则优先）

        return merged

    def _detect_conflicts(
        self,
        profile: RequirementProfile,
        purpose: Optional[CanonicalPurpose],
    ) -> List[str]:
        """
        检测需求中的冲突。

        例如：
        - environment 同时为 indoor 和 outdoor
        - display_type 为 IFP 但 environment 为 outdoor
        """
        conflicts = []

        # 检查 environment 冲突
        env_conflicts = ConflictDetector.detect_environment_conflicts(
            explicit=profile.environment,
            purpose=purpose,
        )
        conflicts.extend(env_conflicts)

        # 检查 installation 冲突
        inst_conflicts = ConflictDetector.detect_installation_conflicts(
            explicit=profile.installation,
            purpose=purpose,
        )
        conflicts.extend(inst_conflicts)

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
