"""Routing, escalation, coverage, and selective-quality metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from budgetroute.evaluation.calibration_metrics import (
    brier_score,
    expected_calibration_error,
    reliability_diagram_data,
)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def routing_quality_metrics(
    predicted_small_success: list[int],
    actual_small_success: list[int],
    probabilities: list[float] | None = None,
) -> dict[str, Any]:
    if len(predicted_small_success) != len(actual_small_success):
        raise ValueError("predicted and actual routing labels must have equal length")
    if not actual_small_success:
        return {
            "routing_accuracy": None,
            "escalation_precision": None,
            "escalation_recall": None,
            "false_escalation_rate": None,
            "false_non_escalation_rate": None,
            "brier_score": None,
            "expected_calibration_error": None,
            "reliability": [],
        }
    predicted = np.asarray(predicted_small_success, dtype=int)
    actual = np.asarray(actual_small_success, dtype=int)
    predicted_escalate = predicted == 0
    actual_escalate = actual == 0
    true_escalation = int((predicted_escalate & actual_escalate).sum())
    false_escalation = int((predicted_escalate & ~actual_escalate).sum())
    missed_escalation = int((~predicted_escalate & actual_escalate).sum())
    probabilities = probabilities or [float(item) for item in predicted_small_success]
    true_positive = int(((predicted == 1) & (actual == 1)).sum())
    true_negative = int(((predicted == 0) & (actual == 0)).sum())
    false_positive = int(((predicted == 1) & (actual == 0)).sum())
    false_negative = int(((predicted == 0) & (actual == 1)).sum())
    return {
        "routing_accuracy": float((predicted == actual).mean()),
        "small_model_success_prediction_accuracy": float((predicted == actual).mean()),
        "escalation_precision": _safe_ratio(true_escalation, true_escalation + false_escalation),
        "escalation_recall": _safe_ratio(true_escalation, true_escalation + missed_escalation),
        "false_escalation_rate": _safe_ratio(false_escalation, int((actual == 1).sum())),
        "false_non_escalation_rate": _safe_ratio(missed_escalation, int((actual == 0).sum())),
        "confusion_matrix": [[true_negative, false_positive], [false_negative, true_positive]],
        "brier_score": brier_score(probabilities, actual_small_success),
        "expected_calibration_error": expected_calibration_error(
            probabilities, actual_small_success
        ),
        "reliability": reliability_diagram_data(probabilities, actual_small_success),
    }


def selective_metrics(quality: list[float], abstained: list[bool]) -> dict[str, float | None]:
    if len(quality) != len(abstained):
        raise ValueError("quality and abstention vectors must have equal length")
    if not quality:
        return {"coverage": None, "selective_accuracy": None}
    covered = [
        score for score, is_abstained in zip(quality, abstained, strict=True) if not is_abstained
    ]
    return {
        "coverage": len(covered) / len(quality),
        "selective_accuracy": sum(covered) / len(covered) if covered else None,
    }
