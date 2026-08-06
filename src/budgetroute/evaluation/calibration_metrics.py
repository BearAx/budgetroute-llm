"""Calibration metrics robust to empty and constant inputs."""

from __future__ import annotations

from typing import Any

import numpy as np


def brier_score(probabilities: list[float], outcomes: list[int]) -> float | None:
    if not probabilities or len(probabilities) != len(outcomes):
        return None
    prediction = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    target = np.asarray(outcomes, dtype=float)
    return float(np.mean((prediction - target) ** 2))


def reliability_diagram_data(
    probabilities: list[float], outcomes: list[int], bins: int = 10
) -> list[dict[str, Any]]:
    if not probabilities or len(probabilities) != len(outcomes) or bins <= 0:
        return []
    prediction = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    target = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result: list[dict[str, Any]] = []
    for index in range(bins):
        if index == bins - 1:
            mask = (prediction >= edges[index]) & (prediction <= edges[index + 1])
        else:
            mask = (prediction >= edges[index]) & (prediction < edges[index + 1])
        count = int(mask.sum())
        result.append(
            {
                "bin_lower": float(edges[index]),
                "bin_upper": float(edges[index + 1]),
                "count": count,
                "mean_confidence": float(prediction[mask].mean()) if count else None,
                "accuracy": float(target[mask].mean()) if count else None,
            }
        )
    return result


def expected_calibration_error(
    probabilities: list[float], outcomes: list[int], bins: int = 10
) -> float | None:
    diagram = reliability_diagram_data(probabilities, outcomes, bins)
    total = sum(int(item["count"]) for item in diagram)
    if total == 0:
        return None
    return float(
        sum(
            int(item["count"])
            / total
            * abs(float(item["mean_confidence"]) - float(item["accuracy"]))
            for item in diagram
            if item["count"]
        )
    )
