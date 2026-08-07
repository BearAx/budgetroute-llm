from __future__ import annotations

from budgetroute.config import RoutingConfig
from budgetroute.features.request_features import RequestFeatureExtractor
from budgetroute.routing.calibration import fit_probability_calibrator
from budgetroute.routing.heuristic import HeuristicPolicy
from budgetroute.routing.policies import RandomPolicy
from budgetroute.schemas import GenerationRequest, RouteName


def test_random_policy_reproduces_decision() -> None:
    request = GenerationRequest(request_id="same", prompt="same prompt")
    features = RequestFeatureExtractor().extract(request)
    first = RandomPolicy(42, 0.5, 0.5).decide(request, features)
    second = RandomPolicy(42, 0.5, 0.5).decide(request, features)
    assert first == second


def test_heuristic_routes_easy_and_difficult() -> None:
    policy = HeuristicPolicy(RoutingConfig(policy="heuristic", difficulty_threshold=0.35))
    extractor = RequestFeatureExtractor()
    easy = GenerationRequest(prompt="What is the capital of France?")
    hard = GenerationRequest(prompt="Calculate the equation 17 * 6 + 5 step by step.")
    assert policy.decide(easy, extractor.extract(easy)).route == RouteName.SMALL
    assert policy.decide(hard, extractor.extract(hard)).route == RouteName.LARGE


def test_heuristic_abstains_when_allowed() -> None:
    policy = HeuristicPolicy(RoutingConfig(policy="heuristic"))
    request = GenerationRequest(prompt="This is ambiguous without any context.")
    decision = policy.decide(request, RequestFeatureExtractor().extract(request))
    assert decision.route == RouteName.ABSTAIN


def test_temperature_calibrator_is_bounded_and_deterministic() -> None:
    probabilities = [0.02, 0.1, 0.65, 0.8, 0.97, 0.99]
    outcomes = [0, 0, 0, 1, 1, 1]
    first = fit_probability_calibrator(probabilities, outcomes)
    second = fit_probability_calibrator(probabilities, outcomes)
    assert first == second
    assert first.method == "temperature"
    transformed = first.transform(probabilities)
    assert all(0.0 <= value <= 1.0 for value in transformed)
    assert transformed == sorted(transformed)
