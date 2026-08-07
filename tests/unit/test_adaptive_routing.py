from __future__ import annotations

from budgetroute.backends.fake import FakeBackend
from budgetroute.config import BackendConfig, RoutingConfig
from budgetroute.features.request_features import RequestFeatureExtractor
from budgetroute.inference.load import BackendLoadSnapshot, BackendLoadTracker, TrackedBackend
from budgetroute.routing.adaptive import BudgetAwarePolicy, LoadAwarePolicy
from budgetroute.schemas import BackendName, GenerationRequest, RetrievalHit, RouteName


def _snapshot(name: BackendName, utilization: float, latency: float = 10.0) -> BackendLoadSnapshot:
    return BackendLoadSnapshot(
        backend=name,
        inflight_requests=int(utilization * 10),
        capacity=10,
        utilization=utilization,
        ewma_latency_ms=latency,
        completed_requests=10,
        failed_requests=0,
        error_rate=0.0,
    )


def test_load_aware_policy_uses_guarded_cascade_under_large_backend_pressure() -> None:
    snapshots = {
        BackendName.SMALL: _snapshot(BackendName.SMALL, 0.1),
        BackendName.LARGE: _snapshot(BackendName.LARGE, 1.2),
    }
    config = RoutingConfig(policy="load_aware", difficulty_threshold=0.1)
    policy = LoadAwarePolicy(config, lambda: snapshots)
    request = GenerationRequest(prompt="Calculate 2 + 2 and explain the equation")
    features = RequestFeatureExtractor().extract(request)
    decision = policy.decide(request, features)
    assert decision.route == RouteName.CASCADE
    assert decision.features["trusted_backend_load"]["large"]["utilization"] == 1.2


def test_budget_aware_policy_emits_explicit_human_review_when_no_route_fits() -> None:
    snapshots = {
        BackendName.SMALL: _snapshot(BackendName.SMALL, 0.0),
        BackendName.LARGE: _snapshot(BackendName.LARGE, 0.0),
    }
    backend = BackendConfig(
        quality=0.8,
        input_cost_units_per_1k_tokens=10,
        output_cost_units_per_1k_tokens=10,
    )
    config = RoutingConfig(
        policy="budget_aware",
        max_estimated_cost_units=0.000001,
        human_review_enabled=True,
        human_review_threshold=0.1,
    )
    policy = BudgetAwarePolicy(config, backend, backend, lambda: snapshots)
    request = GenerationRequest(prompt="```code``` Calculate x = 3 + 4 in detail")
    decision = policy.decide(request, RequestFeatureExtractor().extract(request))
    assert decision.route == RouteName.HUMAN_REVIEW
    assert decision.estimated_cost_units == 0.0


def test_tracked_backend_records_success_and_failure_free_load() -> None:
    config = BackendConfig(type="fake", quality=1.0)
    delegate = FakeBackend(BackendName.SMALL, config)
    tracker = BackendLoadTracker(BackendName.SMALL, capacity=2)
    backend = TrackedBackend(delegate, tracker)
    backend.initialize()
    output = backend.generate(GenerationRequest(prompt="What is the capital of France?"))
    snapshot = tracker.snapshot()
    assert output.text == "Paris"
    assert snapshot.completed_requests == 1
    assert snapshot.inflight_requests == 0
    assert snapshot.ewma_latency_ms >= 0


def test_load_aware_policy_retrieval_easy_and_human_review_branches() -> None:
    extractor = RequestFeatureExtractor()
    overloaded = {
        BackendName.SMALL: _snapshot(BackendName.SMALL, 1.1),
        BackendName.LARGE: _snapshot(BackendName.LARGE, 1.2),
    }
    review_policy = LoadAwarePolicy(
        RoutingConfig(
            policy="load_aware",
            difficulty_threshold=0.1,
            human_review_enabled=True,
            human_review_threshold=0.1,
        ),
        lambda: overloaded,
    )
    hard = GenerationRequest(prompt="```code``` Calculate 2 + 2 in detail")
    assert review_policy.decide(hard, extractor.extract(hard)).route == RouteName.HUMAN_REVIEW

    available = {
        BackendName.SMALL: _snapshot(BackendName.SMALL, 0.1, latency=5),
        BackendName.LARGE: _snapshot(BackendName.LARGE, 0.5, latency=20),
    }
    policy = LoadAwarePolicy(RoutingConfig(policy="load_aware"), lambda: available)
    easy = GenerationRequest(prompt="What is the capital of France?")
    assert policy.decide(easy, extractor.extract(easy)).route == RouteName.SMALL
    retrieval = GenerationRequest(prompt="According to the document?", requires_retrieval=True)
    hits = [RetrievalHit(document_id="doc", chunk_id="chunk", text="context", score=0.9)]
    assert (
        policy.decide(retrieval, extractor.extract(retrieval, hits)).route
        == RouteName.SMALL_WITH_RETRIEVAL
    )


def test_budget_aware_policy_selects_an_eligible_backend_and_can_abstain() -> None:
    snapshots = {
        BackendName.SMALL: _snapshot(BackendName.SMALL, 0.0, latency=5),
        BackendName.LARGE: _snapshot(BackendName.LARGE, 0.0, latency=50),
    }
    small = BackendConfig(input_cost_units_per_1k_tokens=0.1)
    large = BackendConfig(input_cost_units_per_1k_tokens=10)
    request = GenerationRequest(prompt="A short factual question")
    features = RequestFeatureExtractor().extract(request)
    selected = BudgetAwarePolicy(
        RoutingConfig(
            policy="budget_aware",
            max_estimated_cost_units=1,
            quality_weight=0.1,
            latency_weight=0.45,
            cost_weight=0.45,
        ),
        small,
        large,
        lambda: snapshots,
    ).decide(request, features)
    assert selected.route == RouteName.SMALL
    assert selected.estimated_cost_units is not None

    rejected = BudgetAwarePolicy(
        RoutingConfig(policy="budget_aware", max_estimated_cost_units=0),
        small,
        large,
        lambda: snapshots,
    ).decide(request, features)
    assert rejected.route == RouteName.ABSTAIN
