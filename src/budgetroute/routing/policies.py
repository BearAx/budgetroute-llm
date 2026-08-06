"""Fixed and seeded-random routing baselines."""

from __future__ import annotations

import hashlib

from budgetroute.routing.base import feature_snapshot
from budgetroute.schemas import GenerationRequest, RequestFeatures, RouteDecision, RouteName


class AlwaysSmallPolicy:
    name = "always_small"

    def decide(self, request: GenerationRequest, features: RequestFeatures) -> RouteDecision:
        return RouteDecision(
            route=RouteName.SMALL,
            confidence=1.0,
            difficulty_score=None,
            reason="Fixed baseline always selects the small backend",
            policy=self.name,
            features=feature_snapshot(features),
        )


class AlwaysLargePolicy:
    name = "always_large"

    def decide(self, request: GenerationRequest, features: RequestFeatures) -> RouteDecision:
        return RouteDecision(
            route=RouteName.LARGE,
            confidence=1.0,
            difficulty_score=None,
            reason="Fixed baseline always selects the large backend",
            policy=self.name,
            features=feature_snapshot(features),
        )


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int, small_probability: float, large_probability: float) -> None:
        total = small_probability + large_probability
        self.seed = seed
        self.small_probability = small_probability / total

    def decide(self, request: GenerationRequest, features: RequestFeatures) -> RouteDecision:
        material = f"{self.seed}\0{request.request_id}\0{request.prompt}".encode()
        sample = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") / (2**64 - 1)
        route = RouteName.SMALL if sample < self.small_probability else RouteName.LARGE
        return RouteDecision(
            route=route,
            confidence=abs(sample - self.small_probability),
            difficulty_score=None,
            reason=f"Seeded deterministic sample {sample:.4f} selected {route.value}",
            policy=self.name,
            features=feature_snapshot(features),
            thresholds={"small_probability": self.small_probability},
        )
