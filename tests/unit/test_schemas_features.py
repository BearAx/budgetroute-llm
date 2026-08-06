from __future__ import annotations

import pytest
from pydantic import ValidationError

from budgetroute.features.request_features import RequestFeatureExtractor
from budgetroute.schemas import GenerationRequest, RouteDecision, RouteName


def test_prompt_validation() -> None:
    with pytest.raises(ValidationError, match="non-whitespace"):
        GenerationRequest(prompt="   ")


def test_feature_extraction_is_interpretable() -> None:
    features = RequestFeatureExtractor().extract(
        GenerationRequest(prompt="Calculate 12 + 7?\nReturn one number.")
    )
    assert features.category == "numeric_reasoning"
    assert features.has_math_symbols
    assert features.question_mark_count == 1
    assert features.line_count == 2
    assert features.digit_ratio > 0


def test_route_decision_round_trip() -> None:
    decision = RouteDecision(
        route=RouteName.SMALL,
        confidence=0.9,
        reason="test",
        policy="always_small",
    )
    assert RouteDecision.model_validate_json(decision.model_dump_json()) == decision
