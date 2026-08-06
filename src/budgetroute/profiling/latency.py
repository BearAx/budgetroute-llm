"""Latency summaries with explicit small-sample warnings."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

import numpy as np


@dataclass(frozen=True)
class LatencySummary:
    count: int
    mean_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    warning: str | None


def summarize_latencies(values_ms: list[float]) -> LatencySummary:
    if not values_ms:
        return LatencySummary(0, None, None, None, None, "no successful latency samples")
    values = np.asarray(values_ms, dtype=np.float64)
    warning = None
    if len(values_ms) < 20:
        warning = "percentiles are unstable with fewer than 20 samples"
    return LatencySummary(
        count=len(values_ms),
        mean_ms=float(mean(values_ms)),
        p50_ms=float(np.percentile(values, 50)),
        p95_ms=float(np.percentile(values, 95)),
        p99_ms=float(np.percentile(values, 99)) if len(values_ms) >= 100 else None,
        warning=warning,
    )
