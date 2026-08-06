"""Routing policy construction from validated configuration."""

from budgetroute.config import RoutingConfig
from budgetroute.routing.base import RoutingPolicy
from budgetroute.routing.cascade import CascadePolicy
from budgetroute.routing.heuristic import HeuristicPolicy, RetrievalFirstPolicy
from budgetroute.routing.learned import LearnedPolicy
from budgetroute.routing.policies import AlwaysLargePolicy, AlwaysSmallPolicy, RandomPolicy


def build_policy(config: RoutingConfig) -> RoutingPolicy:
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
        return LearnedPolicy(config.learned_model_path)
    raise ValueError(f"unknown routing policy: {config.policy}")
