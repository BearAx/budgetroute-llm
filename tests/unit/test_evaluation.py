from __future__ import annotations

import pytest

from budgetroute.evaluation.calibration_metrics import (
    brier_score,
    expected_calibration_error,
    reliability_diagram_data,
)
from budgetroute.evaluation.evaluators import exact_match, keyword, numeric, token_f1
from budgetroute.evaluation.routing_metrics import routing_quality_metrics, selective_metrics
from budgetroute.evaluation.uncertainty import grouped_bootstrap_mean


def test_deterministic_evaluators() -> None:
    assert exact_match("The Paris!", "the paris").score == 1
    assert token_f1("alpha beta", "alpha gamma").score == pytest.approx(0.5)
    assert numeric("The answer is 1,024.", "1024").score == 1
    assert numeric("First 2, then 3. Final answer: 3/4", "0.75").score == 1
    assert numeric("Reasoning 2 + 3 = 5, therefore 5", "5").score == 1
    assert keyword("A partial match", ["art"]).score == 0


def test_grouped_bootstrap_is_deterministic_and_group_aware() -> None:
    first = grouped_bootstrap_mean([0.0, 1.0, 0.5, 0.5], ["a", "a", "b", "b"], samples=100, seed=7)
    second = grouped_bootstrap_mean([0.0, 1.0, 0.5, 0.5], ["a", "a", "b", "b"], samples=100, seed=7)
    assert first == second
    assert first is not None
    assert first["group_count"] == 2


def test_calibration_handles_empty_and_constant_inputs() -> None:
    assert brier_score([], []) is None
    assert expected_calibration_error([0.5, 0.5], [0, 1]) == 0
    diagram = reliability_diagram_data([1.0], [1], bins=2)
    assert sum(item["count"] for item in diagram) == 1


def test_routing_metrics_handle_empty_classes() -> None:
    metrics = routing_quality_metrics([1, 1], [1, 1], [0.8, 0.9])
    assert metrics["routing_accuracy"] == 1
    assert metrics["escalation_precision"] is None
    assert selective_metrics([1.0, 0.0], [False, True]) == {
        "coverage": 0.5,
        "selective_accuracy": 1.0,
    }
