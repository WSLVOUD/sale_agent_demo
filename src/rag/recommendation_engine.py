"""
Phase 8：Recommendation Engine（确定性产品选型）。

流程（计划文档「十二、Phase 8」）::

    RequirementProfile
        ↓ Hard Filter        （环境 / 固装租赁 / 产品类型 / 明确参数必须满足）
    Candidate Models         （真实存在的 Model，不含 Series）
        ↓ Soft Score         （场景契合 / 视距点间距 / 尺寸箱体 / 预算 / 质量）
    Best Product             （Model 级，例如 TW21-3216-P2.5）

关键原则：
  - **RAG 不负责最终选型**：RAG 只提供产品事实，选型由本模块的确定性打分完成
  - **LLM 不参与选型**：输入是结构化档案，输出是确定性的排序结果
  - 推荐必须是 Model 级（``TW21-3216-P2.5``），不能只给 Series

评分权重（来自计划文档）：场景契合 30% / 视距点间距 25% / 尺寸箱体 20% /
预算 15% / 质量与特殊功能 10%。没有对应信息的分项会被剔除并重新归一化权重。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.models.product import CanonicalModel
from src.models.requirement import RequirementProfile
from src.rag.hard_filter import HardConstraints, build_hard_constraints
from src.rag.parameter_inference import infer_technical_parameters

logger = logging.getLogger(__name__)

# ── 权重（计划文档给定）────────────────────────────────────────────────────
WEIGHTS: Dict[str, float] = {
    "scene": 0.30,
    "pitch": 0.25,
    "size": 0.20,
    "budget": 0.15,
    "quality": 0.10,
}

# 场景 → 偏好点间距区间（用于"场景契合"打分）
# 业务规则：**室内默认首选 3~3.5mm**，所以室内场景的偏好区间统一在 3.0~3.5mm；
# 远距离大屏（机场/户外广告/体育场）与演出类维持更粗的口径。
# 客户明确点名点间距时不看这张表（走 explicit 分支）。
SCENE_PITCH_PREFERENCE: Dict[str, Tuple[float, float]] = {
    "conference": (3.0, 3.5),
    "classroom": (3.0, 3.5),
    "church": (3.0, 3.5),
    "retail": (3.0, 3.5),
    "showroom": (3.0, 3.5),
    "museum": (3.0, 3.5),
    "hall": (3.0, 4.0),
    "control_room": (3.0, 3.5),
    "office": (3.0, 3.5),
    "hotel": (3.0, 3.5),
    "bank": (3.0, 3.5),
    "restaurant": (3.0, 3.5),
    "airport": (3.0, 5.0),
    "hospital": (3.0, 3.5),
    "advertising": (4.0, 10.0),
    "stadium": (4.0, 10.0),
    "concert": (2.6, 4.8),
    "stage": (2.6, 4.8),
    "exhibition": (3.0, 3.9),
    "rental": (2.6, 4.8),
}

# 场景隐含的预算档位（客户没说预算时使用，避免给普通会议室推旗舰 COB）
SCENE_IMPLIED_BUDGET: Dict[str, str] = {
    "showroom": "high",
    "museum": "medium",
    "advertising": "medium",
    "stadium": "medium",
    "concert": "medium",
    "stage": "medium",
    "exhibition": "medium",
}
DEFAULT_IMPLIED_BUDGET = "medium"

# 场景 → 需要的特殊能力（缺失则扣分）
SCENE_FEATURES: Dict[str, Dict[str, float]] = {
    "showroom": {"hdr": 0.6, "cob": 0.4},
    "museum": {"hdr": 0.4, "cob": 0.3},
    "advertising": {"waterproof": 1.0},
    "stadium": {"waterproof": 1.0},
    "concert": {"rental": 1.0},
    "stage": {"rental": 1.0},
}

SCENE_HIGH_BRIGHTNESS_SCENES = {"advertising", "stadium", "concert"}

# 价格档位排序值（用于"最便宜优先"排序）
PRICE_TIER_ORDER: Dict[str, int] = {
    "low": 0,      # 最便宜
    "medium": 1,   # 中等
    "high": 2,     # 最贵
}


@dataclass
class ScoredModel:
    """候选型号及其打分明细。"""

    model: CanonicalModel
    score: float
    breakdown: Dict[str, Optional[float]] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        m = self.model
        return {
            "model": m.model,
            "series_id": m.series_id,
            "score": round(self.score, 2),
            "breakdown": {
                k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in self.breakdown.items()
            },
            "reasons": self.reasons,
            "pixel_pitch_mm": m.pixel_pitch_mm,
            "brightness_nit": m.brightness_nit,
            "installation": m.installation,
            "indoor": m.indoor,
            "outdoor": m.outdoor,
            "price_tier": m.price_tier,
            "warranty_years": m.warranty_years,
            "module_size_mm": m.module_size_text,
            "cabinet_size_mm": m.cabinet_size_text,
            "modules_per_cabinet": m.modules_per_cabinet,
            "cabinet_resolution": m.cabinet_resolution,
            "features": list(m.features),
        }


class RecommendationEngine:
    """确定性的 Model 级选型引擎。"""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        models: Optional[Sequence[CanonicalModel]] = None,
    ):
        if models is None:
            from src.config import config
            from src.rag.json_loader import load_canonical_models

            models = load_canonical_models(data_dir or config.DATA_DIR)
        self.models: List[CanonicalModel] = list(models)
        logger.info("RecommendationEngine ready with %d canonical models", len(self.models))

    # ── 主入口 ──────────────────────────────────────────────────────────
    def recommend(
        self,
        message: str = "",
        profile: Optional[RequirementProfile] = None,
        history: Optional[Sequence[Dict[str, Any]]] = None,
        top_k: int = 3,
        require_ready: bool = False,
    ) -> Dict[str, Any]:
        """产出 Model 级推荐（确定性，无 LLM）。"""
        if profile is None:
            from src.rag.query_understanding import understand_query

            profile = understand_query(message, history=history).profile

        # 第二道保险：调用方显式要求时，必须先过 Recommendation Ready Gate。
        # 统一入口 RecommendationService 会传 require_ready=True。
        gate = None
        if require_ready:
            from src.rag.readiness import check_recommendation_ready

            gate = check_recommendation_ready(profile)
            if not gate.ready:
                return {
                    "recommendation_status": "NEED_CLARIFICATION",
                    "missing_fields": gate.missing,
                    "next_question": gate.next_question,
                    "gate": gate.to_dict(),
                    "recommendations": [],
                    "candidate_count": 0,
                    "rejected_count": 0,
                    "violations": [],
                }

        technical = infer_technical_parameters(profile.to_facts())
        # 【客户口径 2026-09-21】客户明确要了分辨率（4K…）又没锁死点间距 →
        # 点间距以"能不能拼到目标分辨率"为准：视距推出来的区间只是倾向，
        # 不能把更细的点间距一票否决（见 _violation 里的 resolution_driven_pitch）。
        resolution_target = _resolution_target(profile)
        if resolution_target and profile.pixel_pitch_mm is None:
            technical["resolution_driven_pitch"] = True
            technical["resolution_target"] = list(resolution_target)
            logger.info(
                "[Resolution] 分辨率优先：目标 %sx%s → 不限制点间距下限（视距区间仅作倾向）",
                resolution_target[0], resolution_target[1],
            )
        # v2.1 Phase 7：客户授权 AI 决定点间距 → 由环境 + 观看距离 / 场景偏好确定性推导
        # （绝不让 LLM 猜一个 P 值当事实）
        if profile.is_delegated("pixel_pitch"):
            logger.info(
                "[Delegated] pixel_pitch 由环境(%s)+距离(%s)/场景(%s)推导：band=%s~%s target=%s",
                technical.get("environment"), technical.get("viewing_distance_m"),
                profile.purpose, technical.get("pixel_pitch_min_mm"),
                technical.get("pixel_pitch_max_mm"), technical.get("pitch_target_mm"),
            )
        constraints = build_hard_constraints(profile)

        candidates, rejected = self._hard_filter(constraints, technical)
        relaxed_environment = False
        if not candidates and not constraints.model:
            # 客户口径：点间距按"环境 + 距离"算出来（例如 >30m → P8~P10），
            # 但库里可能没有该环境的对应型号（例如 P10 只有户外系列）——
            # 这时放宽"环境"限制再筛一次，让最接近目标的型号能被选出来，
            # 而不是直接"没有匹配产品"。
            #
            # 客户口径（2026-09-18）：**不换环境**。同一环境里没有"距离对应的点间距"
            # （典型：室内 100m → 规则给出 P8~P10，而室内最大只有 P4.81）时，
            # 就在同一环境（+ 固装/租赁等其他硬条件）里取**点间距最大**的那款，
            # 而不是改推室外型号。
            # 只有"该环境/安装方式在库里本来就一片空白"（例如户外租赁）时，
            # 仍然按"没有匹配产品"处理。
            probe_technical = {
                key: value for key, value in technical.items()
                if key not in ("pixel_pitch_min_mm", "pixel_pitch_max_mm")
            }
            probe_candidates, _ = self._hard_filter(constraints, probe_technical)
            if probe_candidates:
                largest = max(probe_candidates, key=lambda m: m.pixel_pitch_mm)
                candidates = [largest]
                rejected = [
                    {"model": m.model, "reason": "点间距不符合该距离区间"}
                    for m in probe_candidates if m is not largest
                ]
                relaxed_environment = False
                logger.info(
                    "环境/距离对应点间距（%s~%smm）在 %s 环境无匹配 → 取该环境最大点间距 %s（P%s）",
                    technical.get("pixel_pitch_min_mm"), technical.get("pixel_pitch_max_mm"),
                    constraints.environment, largest.model, largest.pixel_pitch_mm,
                )
        relaxed = None
        if not candidates and constraints.model:
            # 客户点名的型号在库里不存在时，退化为"同系列 + 同量级点间距"，
            # 而不是直接返回空 —— 但仍必须满足其他硬约束。
            relaxed = self._relax_model_constraint(constraints, technical)
            candidates, rejected = self._hard_filter(relaxed, technical)
            if candidates:
                logger.info(
                    "型号 %s 不在产品库，已放宽为 系列=%s / 点间距≈%s",
                    constraints.model, relaxed.series_id, profile.pixel_pitch_mm,
                )
        # 【客户口径 2026-09-21】客户明确要了分辨率 → 先只留"尺寸上真能拼到目标分辨率"
        # 的型号（否则更粗点间距的型号会靠打分排到前面，把能达标的型号挤出 Top-K）。
        if technical.get("resolution_driven_pitch") and candidates:
            capable = []
            for model in candidates:
                try:
                    from src.engineering import check_model_feasibility

                    check = check_model_feasibility(profile, model)
                except Exception:  # pragma: no cover - 防御式
                    check = {"applicable": False}
                if not check.get("applicable") or check.get("acceptable"):
                    capable.append(model)
            if capable and len(capable) < len(candidates):
                logger.info(
                    "[Resolution] 分辨率优先：%d/%d 个型号在目标尺寸上能拼到目标分辨率",
                    len(capable), len(candidates),
                )
            if capable:
                candidates = capable

        scored = sorted(
            (self._score(model, profile, technical) for model in candidates),
            key=lambda item: (
                -item.score,                                                    # 1. 分数高的优先
                PRICE_TIER_ORDER.get(item.model.price_tier or "medium", 1),   # 2. 最便宜的优先
                self._pitch_tiebreak(item.model, technical),                     # 3. 点间距兜底
                item.model.model,                                               # 4. 型号名排序
            ),
        )
        top = scored[:top_k]

        violations = constraints.violations([item.model for item in top])
        # ── v2.5+++（计划 §4）：每个候选都带上 requested → resolved 的显式说明 ──
        # 客户说 P3、系统只能给 P2.9 时，话术里必须能解释这层关系（不是无声替换）。
        from src.rag.pitch_resolution import resolve_pitch

        # "现成可选的点间距"= 这一套配置下真正过滤出来的候选（不是整个目录）
        available_pitches = [m.pixel_pitch_mm for m in (candidates or self.models)]
        recommendations = []
        for item in top:
            payload = item.to_dict()
            pitch_resolution = resolve_pitch(
                profile,
                item.model,
                available_pitches=available_pitches,
                band_min=technical.get("pixel_pitch_min_mm"),
                band_max=technical.get("pixel_pitch_max_mm"),
            )
            payload["pitch_resolution"] = pitch_resolution.to_dict()
            if pitch_resolution.needs_explanation:
                explanation = pitch_resolution.explain()
                if explanation and explanation not in payload.get("reasons", []):
                    payload.setdefault("reasons", []).insert(0, explanation)
            recommendations.append(payload)

        result = {
            "profile": profile.model_dump(),
            "technical_parameters": technical,
            "hard_constraints": constraints.to_dict(),
            "candidate_count": len(candidates),
            "rejected_count": len(rejected),
            "recommendations": recommendations,
            "pitch_resolution": recommendations[0]["pitch_resolution"] if recommendations else None,
            "violations": violations,
            "relaxed_model": relaxed.model if relaxed else None,
            "original_model": constraints.model,
            # Phase 11/14：READY 或 DEGRADED_READY（部分字段客户不知道）
            "recommendation_status": (
                "DEGRADED" if gate is not None and gate.status == "DEGRADED_READY"
                else "RECOMMENDED"
            ),
            "unknown_requirements": list(gate.unknown_slots) if gate is not None else [],
        }
        logger.info(
            "RecommendationEngine: candidates=%d top=%s violations=%d",
            len(candidates), [item.model.model for item in top], len(violations),
        )
        return result

    @staticmethod
    def _pitch_tiebreak(
        model: CanonicalModel,
        technical: Dict[str, Any],
    ) -> float:
        """同分时的点间距兜底（客户口径：宁可偏粗，绝不无依据地取最细）。

        以前这里是"点间距小的优先"，于是在"没有观看距离 → 点间距维度不参与
        打分 → 同系列所有型号完全平分"的情况下，直接挑中最细最贵的型号
        （实测：10x5m 室内墙屏推出 P1.2，实际上 P3 就够）。

        现在：有窗口首选值 → 取最接近首选的；没有 → 取偏粗的一端（便宜、
        不会白花钱）。
        """
        pitch = float(getattr(model, "pixel_pitch_mm", 0.0) or 0.0)
        target = technical.get("pitch_target_mm")
        if target:
            return abs(pitch - float(target))
        return -pitch

    def _relax_model_constraint(
        self,
        constraints: HardConstraints,
        technical: Dict[str, Any],
    ) -> HardConstraints:
        """把"精确型号"松绑为"同系列 + 同量级点间距"。"""
        from dataclasses import replace

        series_id = None
        for model in self.models:
            if model.model == constraints.model:
                return constraints
        # 型号不存在：从型号名解析系列
        import re

        match = re.match(r"(TW\d{2}-(?:IRHD|HOD|COB|3216|IR|OD))", constraints.model or "")
        if match:
            series_id = match.group(1)
        pitch = constraints.pixel_pitch_min
        tolerance = None
        if pitch is not None and constraints.pixel_pitch_max is not None:
            tolerance = (constraints.pixel_pitch_max - constraints.pixel_pitch_min) / 2
        return replace(
            constraints,
            model=None,
            series_id=series_id or constraints.series_id,
            pixel_pitch_min=pitch,
            pixel_pitch_max=constraints.pixel_pitch_max,
        )

    # ── Hard Filter ─────────────────────────────────────────────────────
    def _hard_filter(
        self,
        constraints: HardConstraints,
        technical: Dict[str, Any],
    ) -> Tuple[List[CanonicalModel], List[Dict[str, str]]]:
        """排除任何违反硬约束的型号。"""
        kept: List[CanonicalModel] = []
        rejected: List[Dict[str, str]] = []
        for model in self.models:
            reason = self._violation(model, constraints, technical)
            if reason:
                rejected.append({"model": model.model, "reason": reason})
            else:
                kept.append(model)
        return kept, rejected

    def _violation(
        self,
        model: CanonicalModel,
        c: HardConstraints,
        technical: Dict[str, Any],
    ) -> Optional[str]:
        # 客户直接点名型号：该型号就是答案，不再用从型号名里解析出的
        # 点间距/亮度去否决它（例如 "TW11-OD-P6" 的实际点间距是 6.67mm）
        if c.model and model.model == c.model:
            return None
        if c.display_type and model.display_type != c.display_type:
            return f"display_type={model.display_type} != {c.display_type}"
        if c.exclude_ifp and model.display_type == "IFP":
            return "客户明确排除 IFP"
        if c.environment == "indoor" and not model.indoor:
            return "非室内产品"
        if c.environment == "outdoor" and not model.outdoor:
            return "非户外产品"
        if c.installation and model.installation != c.installation:
            return f"installation={model.installation} != {c.installation}"
        if c.waterproof and not model.waterproof:
            return "不防水"
        if c.cob and not model.cob:
            return "非 COB"
        if c.hdr and not model.hdr:
            return "不支持 HDR"
        if c.gob and not model.gob:
            return "非 GOB 防护"
        if c.flexible and not model.flexible:
            return "非柔性屏"
        if c.price_tier and model.price_tier != c.price_tier:
            return f"价格档位 {model.price_tier} != {c.price_tier}"
        if c.model and model.model != c.model:
            return "型号不匹配"
        if c.series_id and model.series_id != c.series_id:
            return "系列不匹配"
        pitch = model.pixel_pitch_mm
        if c.pixel_pitch_min is not None and pitch < c.pixel_pitch_min - 1e-6:
            return f"点间距 {pitch}mm 小于客户要求 {c.pixel_pitch_min}mm"
        if c.pixel_pitch_max is not None and pitch > c.pixel_pitch_max + 1e-6:
            return f"点间距 {pitch}mm 大于客户要求 {c.pixel_pitch_max}mm"
        # 客户没点名点间距时，"环境 + 观看距离"的业务区间同样按硬约束处理
        # （室外 4m → P4 就不再推 P2.5；室内 >3m → P3 及以上就不再推 P2.5）
        if c.pixel_pitch_min is None and c.pixel_pitch_max is None:
            band_min = technical.get("pixel_pitch_min_mm")
            band_max = technical.get("pixel_pitch_max_mm")
            # 【客户口径 2026-09-21】客户明确要了分辨率（4K…）时，点间距必须先满足
            # 分辨率：视距/环境推出来的区间只是"倾向"，不能把"能拼到 4K 的更细点间距"
            # 一票否决掉（实测 bug：3×5m 要 4K，P0.7 的型号被"室内 5m → P3 以上"挡掉）。
            if technical.get("resolution_driven_pitch") and band_min is not None:
                band_min = None
            if band_min is not None and pitch < float(band_min) - 1e-6:
                return f"点间距 {pitch}mm 低于该距离/环境的推荐下限 {band_min}mm"
            if band_max is not None and pitch > float(band_max) + 1e-6:
                return f"点间距 {pitch}mm 高于该距离/环境的推荐上限 {band_max}mm"
        if c.brightness_min is not None and model.brightness_nit < c.brightness_min:
            return f"亮度 {model.brightness_nit}nit 低于客户要求 {c.brightness_min}nit"
        if c.brightness_max is not None and model.brightness_nit > c.brightness_max:
            return f"亮度 {model.brightness_nit}nit 高于客户要求 {c.brightness_max}nit"
        return None

    # ── Soft Score ──────────────────────────────────────────────────────
    def _score(
        self,
        model: CanonicalModel,
        profile: RequirementProfile,
        technical: Dict[str, Any],
    ) -> ScoredModel:
        breakdown: Dict[str, Optional[float]] = {
            "scene": self._scene_fit(model, profile),
            "pitch": self._pitch_fit(model, profile, technical),
            "size": self._size_fit(model, profile),
            "budget": self._budget_fit(model, profile),
            "quality": self._quality_score(model),
        }
        applicable = {k: v for k, v in breakdown.items() if v is not None}
        # 业务规则给的"首选点间距"（室内 ≤3m→P2.5 / >3m→P3；室外 4m→P4、5m→P4-P5、
        # 6~20m→P5、>30m→P10）必须**稳定压过**场景/预算等软偏好，
        # 否则出现"规则说 P5、却被场景偏好推到 P6"的情况 → 提高 pitch 权重。
        weights = WEIGHTS
        if (
            technical.get("pitch_target_mm") is not None
            and profile.pixel_pitch_mm is None
        ):
            weights = {**WEIGHTS, "pitch": WEIGHTS["pitch"] * 2.5}
        total_weight = sum(weights[k] for k in applicable) or 1.0
        score = sum(weights[k] * v for k, v in applicable.items()) / total_weight * 100
        reasons = self._reasons(model, profile, breakdown)
        return ScoredModel(model=model, score=score, breakdown=breakdown, reasons=reasons)

    def _scene_fit(self, model: CanonicalModel, profile: RequirementProfile) -> Optional[float]:
        purpose = profile.purpose
        scores: List[float] = []

        # 客户明确点名了点间距 → 场景偏好不再参与（一律听客户的），
        # 否则"客户要 P2.0、场景偏好 P3"会互相打架，出现推荐 P2.5 的结果。
        if (
            purpose
            and purpose in SCENE_PITCH_PREFERENCE
            and profile.pixel_pitch_mm is None
        ):
            low, high = SCENE_PITCH_PREFERENCE[purpose]
            scores.append(_band_fit(model.pixel_pitch_mm, low, high))

        features = SCENE_FEATURES.get(purpose or "", {})
        if features:
            feature_score = 0.0
            for name, weight in features.items():
                if name == "rental":
                    ok = model.installation == "rental"
                elif name == "brightness":
                    ok = model.brightness_nit >= 4500
                else:
                    ok = bool(getattr(model, name, False))
                feature_score += weight * (1.0 if ok else 0.0)
            scores.append(min(1.0, feature_score / max(0.001, sum(features.values()))))

        if purpose in SCENE_HIGH_BRIGHTNESS_SCENES:
            scores.append(min(1.0, model.brightness_nit / 6000))

        if not scores:
            return None
        return sum(scores) / len(scores)

    def _pitch_fit(
        self,
        model: CanonicalModel,
        profile: RequirementProfile,
        technical: Dict[str, Any],
    ) -> Optional[float]:
        if profile.pixel_pitch_mm is not None:
            # 客户明确指定 → 完全匹配 1.0，误差按比例扣分
            delta = abs(model.pixel_pitch_mm - profile.pixel_pitch_mm)
            return max(0.0, 1.0 - delta / max(profile.pixel_pitch_mm, 0.1))
        low = technical.get("pixel_pitch_min_mm")
        high = technical.get("pixel_pitch_max_mm")
        if low is None or high is None:
            return None
        # 业务规则给出的"首选点间距"（室内 ≤3m→2.5 / >3m→3.0；
        # 室外 4m→4.0 / 5m→4.5 / 6~20m→5.0 / 20~30m→6.7 / >30m→10.0）
        target = technical.get("pitch_target_mm")
        if target is not None:
            span = max(high - low, 0.1)
            ratio = (min(max(float(target), low), high) - low) / span
            return _graded_pitch_fit(model.pixel_pitch_mm, low, high, target_ratio=ratio)
        # 区间内按"性价比最优点"（区间 75% 位置）渐变打分：
        # 越靠近该点越高分，贴着区间边缘次之，越界快速衰减。
        return _graded_pitch_fit(model.pixel_pitch_mm, low, high)

    def _size_fit(self, model: CanonicalModel, profile: RequirementProfile) -> Optional[float]:
        # 只有宽高都齐才能算箱体排布（客户只报了一条边时由 Gate 继续追问，
        # 这里不能拿半条尺寸去调计算器）
        if not (profile.target_width_mm and profile.target_height_mm):
            return None
        try:
            from src.tools.screen_calculator import calculate_screen
        except Exception:  # pragma: no cover - Phase 9 之前该模块不存在
            return None

        try:
            calc = calculate_screen(
                model=model.model,
                target_width_mm=profile.target_width_mm,
                target_height_mm=profile.target_height_mm,
            )
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("箱体排布打分失败（%s）：%s", model.model, exc)
            return None
        target_area = (profile.target_width_mm or 0) * (profile.target_height_mm or 0)
        actual_area = calc["actual_width_mm"] * calc["actual_height_mm"]
        if target_area <= 0 or actual_area <= 0:
            return None
        return max(0.0, min(1.0, target_area / actual_area))

    def _budget_fit(self, model: CanonicalModel, profile: RequirementProfile) -> Optional[float]:
        """价格档位契合度。

        - 客户明确预算 → 按目标档位打分（明确"便宜/预算低"时低档得分最高）
        - 客户没说预算 → **默认推荐最便宜档位**（业务规则：首选最便宜的）
        """
        if profile.budget_level is None:
            # 未指定预算 → 最便宜优先（low 档得分最高）
            return {"low": 1.0, "medium": 0.5, "high": 0.1}.get(model.price_tier, 0.5)

        order = {"low": 0, "medium": 1, "high": 2}
        target = order.get(profile.budget_level, 1)
        actual = order.get(model.price_tier, 1)
        return {0: 1.0, 1: 0.5, 2: 0.1}[abs(target - actual)]

    def _quality_score(self, model: CanonicalModel) -> float:
        score = 0.55
        score += 0.15 if model.warranty_years >= 2 else 0.0
        score += 0.10 if (model.lamp_brand or "").lower() in ("kinglight", "cob") else 0.0
        score += 0.08 if model.cob else 0.0
        score += 0.06 if model.hdr else 0.0
        score += 0.06 if model.waterproof else 0.0
        return min(1.0, score)

    def _reasons(
        self,
        model: CanonicalModel,
        profile: RequirementProfile,
        breakdown: Dict[str, Optional[float]],
    ) -> List[str]:
        reasons: List[str] = []
        # 【客户口径 2026-09-21】客户明确要了分辨率 → 话术里说清"为什么用更细的点间距"。
        # 有尺寸时，达不到目标分辨率的型号在推荐出口已经被剔掉，所以这里可以据此说明。
        resolution_target = _resolution_target(profile)
        if (
            resolution_target
            and profile.pixel_pitch_mm is None
            and profile.target_width_mm
            and profile.target_height_m
        ):
            reasons.append(
                f"{model.pixel_pitch_mm}mm pitch so the screen reaches about "
                f"{resolution_target[0]}x{resolution_target[1]} pixels"
            )
        if (breakdown.get("pitch") or 0) >= 0.8:
            reasons.append(f"pixel pitch {model.pixel_pitch_mm}mm matches the viewing distance")
        if profile.environment == "outdoor" and model.waterproof:
            reasons.append("waterproof design for outdoor use")
        if model.hdr:
            reasons.append("HDR support")
        if model.cob:
            reasons.append("COB packaging for reliability")
        # 客户口径：质保只在客户主动问到时回答（默认 1 年、可付费延长），
        # 推荐话术里**不主动提**质保。
        if model.installation == "rental":
            reasons.append("quick-install rental design")
        reasons.append(
            f"{model.cabinet_size_text} cabinet with {model.modules_per_cabinet} modules per cabinet"
        )
        return reasons


# ── 打分辅助 ────────────────────────────────────────────────────────────────
def _resolution_target(profile: Any) -> Optional[Tuple[int, int]]:
    """客户的分辨率目标（有才返回）—— 用于"分辨率优先决定点间距"。"""
    requirement = getattr(profile, "resolution_requirement", None)
    if isinstance(requirement, dict) and requirement:
        width = requirement.get("target_width")
        height = requirement.get("target_height")
        if width and height:
            return int(width), int(height)
    raw = str(getattr(profile, "resolution_raw", "") or "")
    if raw:
        try:
            from src.engineering import parse_resolution

            parsed = parse_resolution(raw)
            if parsed and parsed.target:
                return parsed.target
        except Exception:  # pragma: no cover - 防御式
            return None
    return None


def _band_fit(value: float, low: Optional[float], high: Optional[float]) -> float:
    """值落在 [low, high] 内得 1.0，越界按相对距离线性衰减（最低 0）。"""
    if low is None or high is None:
        return 0.0
    if low <= value <= high:
        return 1.0
    span = max(high - low, 0.1)
    distance = low - value if value < low else value - high
    return max(0.0, 1.0 - distance / span)


def _graded_pitch_fit(value: float, low: float, high: float, target_ratio: float = 0.75) -> float:
    """区间内渐变打分：越接近"性价比最优点"越高分，越界快速衰减。"""
    span = max(high - low, 0.1)
    target = low + target_ratio * span
    if low <= value <= high:
        return max(0.5, 1.0 - 0.5 * abs(value - target) / span)
    distance = low - value if value < low else value - high
    return max(0.0, 0.8 - 0.8 * distance / span)


__all__ = [
    "RecommendationEngine",
    "SCENE_PITCH_PREFERENCE",
    "ScoredModel",
    "WEIGHTS",
]
