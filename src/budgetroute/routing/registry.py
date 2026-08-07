"""Routing policy construction from validated configuration."""

from collections.abc import Callable

from budgetroute.config import BackendConfig, RoutingConfig
from budgetroute.inference.load import BackendLoadSnapshot
from budgetroute.routing.adaptive import BudgetAwarePolicy, LoadAwarePolicy
from budgetroute.routing.base import RoutingPolicy
from budgetroute.routing.cascade import CascadePolicy
from budgetroute.routing.heuristic import HeuristicPolicy, RetrievalFirstPolicy
from budgetroute.routing.learned import LearnedPolicy
from budgetroute.routing.policies import AlwaysLargePolicy, AlwaysSmallPolicy, RandomPolicy
from budgetroute.schemas import BackendName


def build_policy(
    config: RoutingConfig,
    *,
    load_provider: Callable[[], dict[BackendName, BackendLoadSnapshot]] | None = None,
    small_backend: BackendConfig | None = None,
    large_backend: BackendConfig | None = None,
) -> RoutingPolicy:
    if config.policy == "always_small":
        return AlwaysSmallPolicy()
    if config.policy == "always_large":
        return AlwaysLargePolicy()
    if config.policy == "random":
        return RandomPolicy(config.seed, config.small_probability, config.large_probability)
    if config.policy == "heuristic":
        return HeuristicPolicy(config)
    if config.policy == "retrieval_first":
        return RetrievalFirstPolicy(config)
    if config.policy == "cascade":
        return CascadePolicy(config)
    if config.policy == "learned":
        assert config.learned_model_path is not None
        return LearnedPolicy(config.learned_model_path, config.learned_success_threshold)
    if config.policy == "load_aware":
        if load_provider is None:
            raise ValueError("load-aware routing requires backend load telemetry")
        return LoadAwarePolicy(config, load_provider)
    if config.policy == "budget_aware":
        if load_provider is None or small_backend is None or large_backend is None:
            raise ValueError("budget-aware routing requires load telemetry and backend costs")
        return BudgetAwarePolicy(config, small_backend, large_backend, load_provider)
    raise ValueError(f"unknown routing policy: {config.policy}")
