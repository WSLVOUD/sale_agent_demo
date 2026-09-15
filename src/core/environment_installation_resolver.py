"""
Environment & Installation Resolver - 环境和安装方式解析器

职责：
1. 统一 environment 判断（indoor/outdoor/semi_outdoor）
2. 统一 installation 判断（fixed/rental）
3. 明确优先级和禁止推断规则
4. 实现冲突检测
"""

import logging
from typing import Optional, Tuple, Literal
from src.core.purpose_normalizer import CanonicalPurpose

logger = logging.getLogger(__name__)

Environment = Literal["indoor", "outdoor", "semi_outdoor"]
Installation = Literal["fixed", "rental"]


# 场景默认环境（仅在无法从客户原话判定时使用）
# 重要：这些都是"默认值"，不能覆盖客户的明确说法
SCENE_DEFAULT_ENVIRONMENT: dict[CanonicalPurpose, Optional[Environment]] = {
    # 明确室内
    CanonicalPurpose.CONFERENCE: "indoor",
    CanonicalPurpose.CLASSROOM: "indoor",
    CanonicalPurpose.CHURCH: "indoor",
    CanonicalPurpose.MUSEUM: "indoor",
    CanonicalPurpose.SHOWROOM: "indoor",
    CanonicalPurpose.AIRPORT: "indoor",
    CanonicalPurpose.BANK: "indoor",
    CanonicalPurpose.HOTEL: "indoor",
    CanonicalPurpose.RESTAURANT: "indoor",
    CanonicalPurpose.OFFICE: "indoor",
    CanonicalPurpose.HOSPITAL: "indoor",
    CanonicalPurpose.EXHIBITION: "indoor",
    CanonicalPurpose.CONTROL_ROOM: "indoor",
    CanonicalPurpose.HALL: "indoor",
    # 商场 / 零售：门店内的屏按室内处理（同一批关键词在 query_understanding 里
    # 也是"一眼室内"）。商场外立面属于 advertising，不走这条。
    CanonicalPurpose.RETAIL: "indoor",

    # 明确室外
    CanonicalPurpose.ADVERTISING: "outdoor",
    CanonicalPurpose.STADIUM: "outdoor",

    # 模糊（既可室内也可室外）
    CanonicalPurpose.CONCERT: None,
    CanonicalPurpose.STAGE: None,
    CanonicalPurpose.WEDDING: None,
    CanonicalPurpose.RENTAL: None,
    CanonicalPurpose.OTHER: None,
}

# 场景默认安装方式
SCENE_DEFAULT_INSTALLATION: dict[CanonicalPurpose, Optional[Installation]] = {
    # 大多数室内场景默认固装
    CanonicalPurpose.CONFERENCE: "fixed",
    CanonicalPurpose.CLASSROOM: "fixed",
    CanonicalPurpose.CHURCH: "fixed",
    CanonicalPurpose.MUSEUM: "fixed",
    CanonicalPurpose.SHOWROOM: "fixed",
    CanonicalPurpose.AIRPORT: "fixed",
    CanonicalPurpose.BANK: "fixed",
    CanonicalPurpose.HOTEL: "fixed",
    CanonicalPurpose.RESTAURANT: "fixed",
    CanonicalPurpose.OFFICE: "fixed",
    CanonicalPurpose.HOSPITAL: "fixed",
    CanonicalPurpose.EXHIBITION: "fixed",
    CanonicalPurpose.CONTROL_ROOM: "fixed",
    CanonicalPurpose.HALL: "fixed",
    CanonicalPurpose.RETAIL: "fixed",

    # 室外广告通常固装
    CanonicalPurpose.ADVERTISING: "fixed",
    CanonicalPurpose.STADIUM: "fixed",

    # 演出/活动通常租赁
    CanonicalPurpose.CONCERT: "rental",
    CanonicalPurpose.STAGE: "rental",
    CanonicalPurpose.WEDDING: "rental",
    CanonicalPurpose.RENTAL: "rental",

    # 其他未指定
    CanonicalPurpose.OTHER: None,
}


class EnvironmentResolver:
    """环境解析器"""

    PRIORITY_EXPLICIT = 4      # 客户明确说
    PRIORITY_SCENARIO = 3      # 场景直接判定
    PRIORITY_INFERRED = 2      # 规则推断
    PRIORITY_DEFAULT = 1       # 业务默认

    @staticmethod
    def resolve(
        explicit_environment: Optional[Environment] = None,
        purpose: Optional[CanonicalPurpose] = None,
    ) -> Tuple[Optional[Environment], int]:
        """
        解析环境。

        Priority:
        1. 客户明确说 (explicit_environment)
        2. 场景能直接判定 (SCENE_DEFAULT_ENVIRONMENT)
        3. None (继续询问)

        Args:
            explicit_environment: 客户明确说的环境（从 extract_slots 提取）
            purpose: 已识别的场景

        Returns:
            (resolved_environment, priority_level)
        """
        # Priority 1: 客户明确说
        if explicit_environment:
            logger.info(f"Environment: {explicit_environment} (explicit)")
            return explicit_environment, EnvironmentResolver.PRIORITY_EXPLICIT

        # Priority 2: 场景可以判定
        if purpose and purpose in SCENE_DEFAULT_ENVIRONMENT:
            default = SCENE_DEFAULT_ENVIRONMENT[purpose]
            if default:
                logger.info(f"Environment: {default} (from scene: {purpose.value})")
                return default, EnvironmentResolver.PRIORITY_SCENARIO

        # Priority 3: 无法判定，继续询问
        logger.info(f"Environment: unknown (need to ask)")
        return None, 0

    @staticmethod
    def is_ambiguous_scene(purpose: Optional[CanonicalPurpose]) -> bool:
        """
        判断是否为模糊场景（既可室内也可室外）。

        这些场景不能通过场景直接推断环境。
        """
        if not purpose:
            return False

        default = SCENE_DEFAULT_ENVIRONMENT.get(purpose)
        return default is None and purpose not in (
            CanonicalPurpose.OTHER,
        )


class InstallationResolver:
    """安装方式解析器"""

    PRIORITY_EXPLICIT = 4
    PRIORITY_SCENARIO = 3
    PRIORITY_DEFAULT = 1

    @staticmethod
    def resolve(
        explicit_installation: Optional[Installation] = None,
        purpose: Optional[CanonicalPurpose] = None,
    ) -> Tuple[Optional[Installation], int]:
        """
        解析安装方式。

        Priority:
        1. 客户明确说 (explicit_installation)
        2. 场景默认值 (SCENE_DEFAULT_INSTALLATION)
        3. None (继续询问 或 使用业务规则)

        Args:
            explicit_installation: 客户明确说的安装方式
            purpose: 已识别的场景

        Returns:
            (resolved_installation, priority_level)
        """
        # Priority 1: 客户明确说
        if explicit_installation:
            logger.info(f"Installation: {explicit_installation} (explicit)")
            return explicit_installation, InstallationResolver.PRIORITY_EXPLICIT

        # Priority 2: 场景默认值
        if purpose and purpose in SCENE_DEFAULT_INSTALLATION:
            default = SCENE_DEFAULT_INSTALLATION[purpose]
            if default:
                logger.info(f"Installation: {default} (from scene: {purpose.value})")
                return default, InstallationResolver.PRIORITY_SCENARIO

        # Priority 3: 无法判定
        logger.info(f"Installation: unknown (need to ask or apply business rule)")
        return None, InstallationResolver.PRIORITY_DEFAULT

    @staticmethod
    def is_likely_rental(purpose: Optional[CanonicalPurpose]) -> bool:
        """
        快速检查：该场景是否倾向于租赁？

        用于在无法从客户原话判定时的业务规则决策。
        """
        if not purpose:
            return False

        return purpose in (
            CanonicalPurpose.CONCERT,
            CanonicalPurpose.STAGE,
            CanonicalPurpose.WEDDING,
            CanonicalPurpose.RENTAL,
        )

    @staticmethod
    def is_likely_fixed(purpose: Optional[CanonicalPurpose]) -> bool:
        """
        快速检查：该场景是否倾向于固装？
        """
        if not purpose:
            return False

        return purpose in (
            CanonicalPurpose.CONFERENCE,
            CanonicalPurpose.CLASSROOM,
            CanonicalPurpose.RETAIL,
            CanonicalPurpose.ADVERTISING,
            CanonicalPurpose.STADIUM,
        )


class ConflictDetector:
    """冲突检测器"""

    @staticmethod
    def detect_environment_conflicts(
        explicit: Optional[Environment] = None,
        inferred: Optional[Environment] = None,
        purpose: Optional[CanonicalPurpose] = None,
    ) -> list[str]:
        """
        检测环境相关的**真正冲突**：同一轮里出现两个互相矛盾的明确信号。

        例如：
        - 规则解析说 indoor，同轮的语义提取却说 outdoor → 需要澄清

        注意：**"客户明确说 outdoor，但该场景通常 indoor"不算冲突** ——
        按优先级，客户明确事实优先，场景默认值让位（例如零售店门口的户外屏）。
        """
        conflicts = []

        if explicit and inferred and explicit != inferred:
            if explicit != inferred:
                logger.warning(
                    "Environment conflict: explicit=%s vs semantic=%s (purpose=%s)",
                    explicit, inferred, purpose.value if purpose else None,
                )
                conflicts.append(
                    f"environment_conflict: {explicit} vs {inferred}"
                )

        return conflicts

    @staticmethod
    def detect_installation_conflicts(
        explicit: Optional[Installation] = None,
        inferred: Optional[Installation] = None,
        purpose: Optional[CanonicalPurpose] = None,
    ) -> list[str]:
        """
        检测安装方式相关的**真正冲突**：同一轮里两个明确信号互相矛盾。

        注意："租赁屏用于会议室"是正常业务（例如租一块屏开一天会），不算冲突。
        """
        conflicts = []

        if explicit and inferred and explicit != inferred:
            if explicit != inferred:
                logger.warning(
                    "Installation conflict: explicit=%s vs semantic=%s (purpose=%s)",
                    explicit, inferred, purpose.value if purpose else None,
                )
                conflicts.append(
                    f"installation_conflict: {explicit} vs {inferred}"
                )

        return conflicts
