"""v2.5+：客户说过室内/户外，就绝不能再问一遍（实测 bug 回归）。

现象：客户回了"室内"，系统后面又问"这是室内安装还是室外设置？"，甚至连着问两次。
根因（两条）：
  1. `requirement_extractor` Step 6 只看规则层有没有命中关键词，语义模型读出来的
     环境一律降级成 inferred（哪怕带了客户原话证据、哪怕是客户在回答我们的提问）；
  2. `readiness._environment_settled()` 不认 inferred → 有值也照样再问一遍。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.requirement_extractor import RequirementExtractor  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.readiness import check_recommendation_ready  # noqa: E402


def _profile(**slots) -> RequirementProfile:
    return RequirementProfile.from_slots(slots, explicit_keys=set(slots))


class TestEnvironmentAnswerSticks:

    def test_keyword_answer_is_confirmed(self):
        profile = RequirementExtractor().extract("indoor", use_llm=False)
        assert profile.environment == "indoor"
        assert profile.sources.get("environment") == "explicit"

    def test_chinese_answer_is_confirmed(self):
        profile = RequirementExtractor().extract("室内", use_llm=False)
        assert profile.environment == "indoor"
        assert profile.sources.get("environment") == "explicit"

    def test_semantic_answer_with_evidence_is_confirmed(self):
        """规则没命中关键词、但语义模型读出环境且带客户原话证据 → 也算客户明说。"""
        profile = RequirementExtractor().extract(
            "we will put it up in our sanctuary",
            semantic_override={
                "environment": "indoor",
                "environment_evidence": "sanctuary",
            },
            use_llm=False,
        )
        assert profile.environment == "indoor"
        assert profile.sources.get("environment") == "explicit"

    def test_short_answer_to_the_question_is_accepted_without_evidence(self):
        """客户就是在回答"室内还是室外"这一问 → 短回答没有证据片段也采信。"""
        previous = RequirementProfile()
        previous.last_asked_slot = "environment"
        profile = RequirementExtractor().extract(
            "inside the building please",
            previous_profile=previous,
            semantic_override={"environment": "indoor"},
            use_llm=False,
        )
        assert profile.environment == "indoor"
        assert profile.sources.get("environment") == "explicit"

    def test_unrelated_answer_is_still_dropped(self):
        """不是回答这一项、又没有证据 → 仍然要丢掉（防幻觉不能被放开）。"""
        profile = RequirementExtractor().extract(
            "we will put it up in our sanctuary",
            semantic_override={"environment": "indoor"},   # 没有 evidence
            use_llm=False,
        )
        assert profile.environment is None


class TestGateDoesNotAskAgain:

    def test_confirmed_environment_is_never_asked_again(self):
        profile = _profile(environment="indoor", installation="fixed", purpose="stage")
        decision = check_recommendation_ready(profile)
        assert decision.next_slot != "environment"
        assert "environment" not in decision.missing

    def test_answered_environment_is_not_asked_again_even_if_source_is_weak(self):
        """历史档案来源不够硬（inferred），但客户已经答过这一项 → 不再问。"""
        profile = _profile(environment="indoor", installation="fixed", purpose="stage")
        profile.sources["environment"] = "inferred"
        profile.record_ask("environment")
        decision = check_recommendation_ready(profile)
        assert decision.next_slot != "environment"
        assert "environment" not in decision.missing

    def test_never_asked_inferred_environment_is_confirmed_once(self):
        """客户从没说过、系统只是猜的 → 还是要跟客户确认一次（这条口径不变）。"""
        profile = _profile(environment="indoor", installation="fixed", purpose="stage")
        profile.sources["environment"] = "inferred"
        decision = check_recommendation_ready(profile)
        assert decision.next_slot == "environment" or "environment" in decision.missing

    def test_explicit_answer_then_gate_does_not_ask(self):
        """整条链：客户用"室内"回答 → 档案是客户确认 → Gate 不再问室内外。"""
        profile = RequirementExtractor().extract("室内", use_llm=False)
        profile = profile.merge(_profile(
            purpose="stage", installation="fixed", viewing_distance_m=5,
        ))
        decision = check_recommendation_ready(profile)
        assert profile.sources.get("environment") == "explicit"
        assert decision.next_slot != "environment"
