"""Bounded metrics, feedback aggregates, and feature-distribution drift detection."""

from __future__ import annotations

import json
import threading
from collections import Counter, deque
from pathlib import Path
from typing import Any

from budgetroute.config import MonitoringConfig
from budgetroute.schemas import (
    FeedbackRecord,
    GenerationRequest,
    GenerationResponse,
    RequestFeatures,
)


def _summary(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    result: dict[str, dict[str, float]] = {}
    for name in rows[0]:
        values = [row[name] for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        result[name] = {"mean": mean, "standard_deviation": variance**0.5}
    return result


class RuntimeMonitor:
    """Retain aggregate numbers and numeric features, never prompts or generated text."""

    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._features: deque[dict[str, float]] = deque(maxlen=config.window_size)
        self._baseline = self._load_baseline(config.baseline_path)
        self._counters: Counter[str] = Counter()
        self._latency_sum_ms = 0.0
        self._queue_sum_ms = 0.0

    @staticmethod
    def _load_baseline(path: Path | None) -> dict[str, dict[str, float]] | None:
        if path is None:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        features = data.get("features")
        if not isinstance(features, dict):
            raise ValueError("monitoring baseline must contain a features mapping")
        return {
            str(name): {
                "mean": float(values["mean"]),
                "standard_deviation": float(values.get("standard_deviation", 0.0)),
            }
            for name, values in features.items()
        }

    def observe(
        self,
        request: GenerationRequest,
        features: RequestFeatures,
        response: GenerationResponse,
    ) -> None:
        del request  # Prompts and metadata are intentionally not retained.
        if not self.config.enabled:
            return
        with self._lock:
            self._features.append(features.numeric_snapshot())
            self._counters["requests_total"] += 1
            self._counters[f"route_{response.route.value}_total"] += 1
            self._counters["errors_total"] += 0
            self._counters["human_review_total"] += int(response.execution.human_review_required)
            self._counters["abstentions_total"] += int(response.execution.abstained)
            self._latency_sum_ms += response.timing.total_ms
            self._queue_sum_ms += response.timing.queue_ms
            if self._baseline is None and len(self._features) == self.config.minimum_samples:
                self._baseline = _summary(list(self._features))

    def record_error(self) -> None:
        with self._lock:
            self._counters["errors_total"] += 1

    def record_feedback(self, feedback: FeedbackRecord) -> None:
        correct = feedback.correct
        del feedback  # Identifiers and notes are intentionally not retained in-process.
        with self._lock:
            self._counters["feedback_total"] += 1
            self._counters["feedback_correct_total"] += int(correct)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            rows = list(self._features)
            baseline = dict(self._baseline) if self._baseline is not None else None
            counters = dict(self._counters)
            latency_sum = self._latency_sum_ms
            queue_sum = self._queue_sum_ms
        current = _summary(rows)
        per_feature: dict[str, float] = {}
        if baseline is not None and len(rows) >= self.config.minimum_samples:
            for name, reference in baseline.items():
                if name not in current:
                    continue
                scale = max(reference.get("standard_deviation", 0.0), 1.0)
                per_feature[name] = abs(current[name]["mean"] - reference["mean"]) / scale
        score = sum(per_feature.values()) / len(per_feature) if per_feature else 0.0
        return {
            "enabled": self.config.enabled,
            "sample_count": len(rows),
            "window_size": self.config.window_size,
            "baseline_ready": baseline is not None,
            "drift_method": "mean_shift_in_baseline_standard_deviations",
            "drift_score": score,
            "drift_threshold": self.config.drift_threshold,
            "drift_detected": score >= self.config.drift_threshold,
            "feature_drift": per_feature,
            "counters": counters,
            "latency_sum_ms": latency_sum,
            "queue_sum_ms": queue_sum,
        }

    def prometheus(self, scheduler: dict[str, Any] | None = None) -> str:
        snapshot = self.snapshot()
        counters = snapshot["counters"]
        lines = [
            "# HELP budgetroute_requests_total Completed inference requests.",
            "# TYPE budgetroute_requests_total counter",
            f"budgetroute_requests_total {counters.get('requests_total', 0)}",
            "# HELP budgetroute_errors_total Expected and unexpected request failures.",
            "# TYPE budgetroute_errors_total counter",
            f"budgetroute_errors_total {counters.get('errors_total', 0)}",
            "# HELP budgetroute_human_review_total Requests requiring external human review.",
            "# TYPE budgetroute_human_review_total counter",
            f"budgetroute_human_review_total {counters.get('human_review_total', 0)}",
            "# HELP budgetroute_latency_milliseconds_sum Sum of end-to-end request latency.",
            "# TYPE budgetroute_latency_milliseconds_sum counter",
            f"budgetroute_latency_milliseconds_sum {snapshot['latency_sum_ms']:.6f}",
            "# HELP budgetroute_drift_score Current numeric feature drift score.",
            "# TYPE budgetroute_drift_score gauge",
            f"budgetroute_drift_score {snapshot['drift_score']:.6f}",
            "# HELP budgetroute_drift_detected Whether drift exceeds the configured threshold.",
            "# TYPE budgetroute_drift_detected gauge",
            f"budgetroute_drift_detected {int(snapshot['drift_detected'])}",
        ]
        if scheduler is not None:
            lines.extend(
                [
                    "# HELP budgetroute_queue_depth Current scheduler queue depth.",
                    "# TYPE budgetroute_queue_depth gauge",
                    f"budgetroute_queue_depth {scheduler.get('queue_depth', 0)}",
                    "# HELP budgetroute_batches_total Executed inference batches.",
                    "# TYPE budgetroute_batches_total counter",
                    f"budgetroute_batches_total {scheduler.get('batches', 0)}",
                    "# HELP budgetroute_queue_rejections_total Rejected overload submissions.",
                    "# TYPE budgetroute_queue_rejections_total counter",
                    f"budgetroute_queue_rejections_total {scheduler.get('rejected', 0)}",
                ]
            )
        return "\n".join(lines) + "\n"
