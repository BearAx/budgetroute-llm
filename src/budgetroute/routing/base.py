"""Routing policy protocol and shared helpers."""

from __future__ import annotations

from typing import Protocol

from budgetroute.schemas import GenerationRequest, RequestFeatures, RouteDecision


class RoutingPolicy(Protocol):
    @property
    def name(self) -> str: ...

    def decide(self, request: GenerationRequest, features: RequestFeatures) -> RouteDecision: ...


def feature_snapshot(features: RequestFeatures) -> dict[str, object]:
    return features.model_dump(mode="json")
