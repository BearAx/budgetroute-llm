"""Load-aware and cost-aware policies with explicit operational trade-offs."""

from __future__ import annotations

from collections.abc import Callable

from budgetroute.config import BackendConfig, RoutingConfig
from budgetroute.inference.load import BackendLoadSnapshot
from budgetroute.routing.base import feature_snapshot
from budgetroute.routing.heuristic import difficulty_score
from budgetroute.schemas import (
    BackendName,
    GenerationRequest,
    RequestFeatures,
    RouteDecision,
    RouteName,
)

LoadProvider = Callable[[], dict[BackendName, BackendLoadSnapshot]]


def _expected_latency(snapshot: BackendLoadSnapshot, default_ms: float) -> float:
    observed = snapshot.ewma_latency_ms or default_ms
    return observed * (1.0 + snapshot.utilization)


def _trace_features(
    features: RequestFeatures, snapshots: dict[BackendName, BackendLoadSnapshot]
) -> dict[str, object]:
    return {
        **feature_snapshot(features),
        "trusted_backend_load": {
            name.value: snapshot.model_dump() for name, snapshot in snapshots.items()
        },
    }


class LoadAwarePolicy:
    name = "load_aware"

    def __init__(self, config: RoutingConfig, load_provider: LoadProvider) -> None:
        self.config = config
        self.load_provider = load_provider

    def decide(self, request: GenerationRequest, features: RequestFeatures) -> RouteDecision:
        difficulty = difficulty_score(features)
        snapshots = self.load_provider()
        small = snapshots[BackendName.SMALL]
        large = snapshots[BackendName.LARGE]
        small_latency = _expected_latency(small, self.config.target_latency_ms * 0.45)
        large_latency = _expected_latency(large, self.config.target_latency_ms * 0.9)
        overloaded = (
            small.utilization >= self.config.overload_threshold
            and large.utilization >= self.config.overload_threshold
        )
        if (
            overloaded
            and self.config.human_review_enabled
            and difficulty >= self.config.human_review_threshold
        ):
            route = RouteName.HUMAN_REVIEW
            reason = "Both backends are overloaded and the request exceeds the review threshold"
            confidence = min(1.0, difficulty)
            latency = 0.0
        elif request.requires_retrieval and features.relevant_context_found:
            route = RouteName.SMALL_WITH_RETRIEVAL
            reason = "Relevant context lowers generation difficulty for the small backend"
            confidence = min(1.0, max(0.5, features.retrieval_similarity))
            latency = small_latency
        elif difficulty >= self.config.difficulty_threshold:
            if large.utilization >= self.config.overload_threshold and small.utilization < 1.0:
                route = RouteName.CASCADE
                reason = "Large backend is under pressure; using a guarded small-first cascade"
                latency = small_latency
            else:
                route = RouteName.LARGE
                reason = "High request difficulty takes precedence over current backend load"
                latency = large_latency
            confidence = min(1.0, 0.5 + abs(difficulty - self.config.difficulty_threshold))
        elif small.utilization <= large.utilization or small_latency <= large_latency:
            route = RouteName.SMALL
            reason = (
                "Small backend satisfies the difficulty threshold with lower load-adjusted cost"
            )
            confidence = min(1.0, 0.5 + self.config.difficulty_threshold - difficulty)
            latency = small_latency
        else:
            route = RouteName.LARGE
            reason = "Large backend has materially lower load-adjusted latency"
            confidence = min(1.0, 0.5 + abs(small.utilization - large.utilization))
            latency = large_latency
        return RouteDecision(
            route=route,
            confidence=confidence,
            difficulty_score=difficulty,
            reason=reason,
            policy=self.name,
            features=_trace_features(features, snapshots),
            thresholds={
                "difficulty": self.config.difficulty_threshold,
                "overload": self.config.overload_threshold,
                "target_latency_ms": self.config.target_latency_ms,
            },
            estimated_latency_ms=latency,
        )


class BudgetAwarePolicy:
    name = "budget_aware"

    def __init__(
        self,
        config: RoutingConfig,
        small_config: BackendConfig,
        large_config: BackendConfig,
        load_provider: LoadProvider,
    ) -> None:
        self.config = config
        self.backend_configs = {
            BackendName.SMALL: small_config,
            BackendName.LARGE: large_config,
        }
        self.load_provider = load_provider

    @staticmethod
    def _cost(config: BackendConfig, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * config.input_cost_units_per_1k_tokens
            + output_tokens * config.output_cost_units_per_1k_tokens
        ) / 1000.0

    def decide(self, request: GenerationRequest, features: RequestFeatures) -> RouteDecision:
        difficulty = difficulty_score(features)
        snapshots = self.load_provider()
        output_tokens = request.max_new_tokens or max(
            self.backend_configs[BackendName.SMALL].generation.max_new_tokens,
            self.backend_configs[BackendName.LARGE].generation.max_new_tokens,
        )
        estimates: dict[BackendName, tuple[float, float, float]] = {}
        for backend, default_latency, quality_bonus in (
            (BackendName.SMALL, self.config.target_latency_ms * 0.45, 0.0),
            (BackendName.LARGE, self.config.target_latency_ms * 0.9, 0.2),
        ):
            cost = self._cost(self.backend_configs[backend], features.token_count, output_tokens)
            latency = _expected_latency(snapshots[backend], default_latency)
            quality = min(0.99, max(0.01, 1.0 - difficulty * 0.7 + quality_bonus))
            max_cost = self.config.max_estimated_cost_units
            cost_scale = max(max_cost or max(cost, 1.0), 1e-12)
            utility = (
                self.config.quality_weight * quality
                - self.config.latency_weight * (latency / self.config.target_latency_ms)
                - self.config.cost_weight * (cost / cost_scale)
            )
            if max_cost is not None and cost > max_cost:
                utility = float("-inf")
            estimates[backend] = (utility, cost, latency)
        selected = max(estimates, key=lambda name: estimates[name][0])
        utility, cost, latency = estimates[selected]
        if (
            utility == float("-inf")
            and self.config.human_review_enabled
            and difficulty >= self.config.human_review_threshold
        ):
            route = RouteName.HUMAN_REVIEW
            reason = "No backend satisfies the configured cost budget for this hard request"
            cost = 0.0
            latency = 0.0
        elif utility == float("-inf") and request.allow_abstention:
            route = RouteName.ABSTAIN
            reason = "No backend satisfies the configured cost budget"
            cost = 0.0
            latency = 0.0
        else:
            route = RouteName.SMALL if selected == BackendName.SMALL else RouteName.LARGE
            reason = (
                f"{selected.value} maximizes configured quality/cost/latency utility "
                f"({utility:.3f})"
            )
        return RouteDecision(
            route=route,
            confidence=min(1.0, max(0.0, 1.0 - difficulty / 2.0)),
            difficulty_score=difficulty,
            reason=reason,
            policy=self.name,
            features={
                **_trace_features(features, snapshots),
                "candidate_estimates": {
                    name.value: {
                        "utility": None if values[0] == float("-inf") else values[0],
                        "cost_units": values[1],
                        "latency_ms": values[2],
                    }
                    for name, values in estimates.items()
                },
            },
            thresholds={
                "target_latency_ms": self.config.target_latency_ms,
                "max_estimated_cost_units": self.config.max_estimated_cost_units or 0.0,
                "quality_weight": self.config.quality_weight,
                "latency_weight": self.config.latency_weight,
                "cost_weight": self.config.cost_weight,
            },
            estimated_cost_units=cost,
            estimated_latency_ms=latency,
        )
