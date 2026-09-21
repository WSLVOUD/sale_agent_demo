"""v2.7 §11/§12/§14/§15（Phase 10/11/13）：Momentum + 自然承接 + 回复密度。"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import (  # noqa: E402
    DETAILED,
    MINIMAL,
    NORMAL,
    build_natural_continuation,
    compute_momentum,
    continuation_candidates,
    decide_response_density,
    momentum_bonus,
    render_minimal,
)


class TestMomentum:

    def test_momentum_follows_the_latest_topic(self):
        momentum = compute_momentum(newly_filled_slots=["size"])
        assert momentum.slot == "size"
        assert "pixel_pitch" in momentum.follow_ups

    def test_momentum_prefers_newest(self):
        momentum = compute_momentum(
            answered_slots=["environment"], newly_filled_slots=["size"]
        )
        assert momentum.slot == "size"
        assert momentum.slots[0] == "size"

    def test_continuation_candidates_are_ranked(self):
        momentum = compute_momentum(newly_filled_slots=["size"])
        ranked = continuation_candidates(momentum, ["installation", "pixel_pitch", "size"])
        assert ranked[0] == "pixel_pitch"
        assert set(ranked) == {"installation", "pixel_pitch", "size"}

    def test_momentum_bonus_only_for_related_slots(self):
        momentum = compute_momentum(newly_filled_slots=["size"])
        assert momentum_bonus("pixel_pitch", momentum) > 0
        assert momentum_bonus("price_preference", momentum) == 0


class TestNaturalContinuation:

    def test_single_parameter_uses_minimal_density(self):
        continuation = build_natural_continuation(
            customer_message="8 meters",
            newly_filled_slots=["viewing_distance"],
            next_required_slot="environment",
            momentum=compute_momentum(newly_filled_slots=["viewing_distance"]),
        )
        assert continuation.density == MINIMAL
        assert continuation.mode == "ask"
        assert continuation.opening == "", "普通参数不要任何铺垫"
        assert continuation.avoid_ack is True

    def test_customer_question_is_answered(self):
        continuation = build_natural_continuation(
            customer_message="How long is delivery?",
            customer_question=True,
            question_kind="DELIVERY_QUESTION",
        )
        assert continuation.mode == "answer"

    def test_recommendation_is_detailed(self):
        continuation = build_natural_continuation(
            customer_message="ok",
            has_recommendation=True,
        )
        assert continuation.density == DETAILED
        assert continuation.mode == "recommend"

    def test_render_minimal_is_just_the_question(self):
        continuation = build_natural_continuation(
            newly_filled_slots=["size"], next_required_slot="pixel_pitch"
        )
        text = render_minimal(continuation, question="Do you know the pitch you want?")
        assert text == "Do you know the pitch you want?"


class TestResponseDensity:

    def test_one_parameter_is_minimal(self):
        assert decide_response_density(newly_filled_slots=["size"]) == MINIMAL

    def test_two_parameters_is_normal(self):
        assert decide_response_density(newly_filled_slots=["size", "environment"]) == NORMAL

    def test_technical_or_recommendation_is_detailed(self):
        assert decide_response_density(has_engineering=True) == DETAILED
        assert decide_response_density(has_recommendation=True) == DETAILED
        assert decide_response_density(customer_requested_detail=True) == DETAILED

    def test_delivery_question_is_short(self):
        assert decide_response_density(
            customer_question=True, question_kind="DELIVERY_QUESTION"
        ) == MINIMAL

    def test_conflict_is_normal(self):
        assert decide_response_density(conflicts=["size_conflict"]) == NORMAL
