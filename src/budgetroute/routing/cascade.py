"""Cascade policy; execution and escalation are owned by the inference engine."""

from __future__ import annotations

from budgetroute.config import RoutingConfig
from budgetroute.routing.base import feature_snapshot
from budgetroute.routing.heuristic import difficulty_score
from budgetroute.schemas import GenerationRequest, RequestFeatures, RouteDecision, RouteName


class CascadePolicy:
    name = "cascade"

    def __init__(self, config: RoutingConfig) -> None:
        self.config = config

    def decide(self, request: GenerationRequest, features: RequestFeatures) -> RouteDecision:
        difficulty = difficulty_score(features)
        return RouteDecision(
            route=RouteName.CASCADE,
            confidence=1.0 - difficulty / 2,
            difficulty_score=difficulty,
            reason="Run the small backend first and evaluate its confidence for escalation",
            policy=self.name,
            features=feature_snapshot(features),
            thresholds={"cascade_confidence": self.config.cascade_confidence_threshold},
        )


def escalation_reason(confidence: float | None, threshold: float) -> str | None:
    if confidence is None:
        return "Small-backend confidence was unavailable"
    if confidence >= threshold:
        return None
    return f"Small-backend confidence {confidence:.3f} was below {threshold:.3f}"
