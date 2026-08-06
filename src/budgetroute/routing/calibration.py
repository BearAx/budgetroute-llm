"""Threshold selection for quality-aware routing policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    coverage: float
    selective_accuracy: float


def choose_threshold(
    probabilities: list[float], outcomes: list[int], target_accuracy: float
) -> ThresholdResult | None:
    if not probabilities or len(probabilities) != len(outcomes):
        return None
    best: ThresholdResult | None = None
    for threshold in sorted(set([0.0, *probabilities, 1.0])):
        mask = np.asarray(probabilities) >= threshold
        if not mask.any():
            continue
        accuracy = float(np.asarray(outcomes, dtype=float)[mask].mean())
        coverage = float(mask.mean())
        candidate = ThresholdResult(threshold, coverage, accuracy)
        if accuracy >= target_accuracy and (best is None or coverage > best.coverage):
            best = candidate
    return best
