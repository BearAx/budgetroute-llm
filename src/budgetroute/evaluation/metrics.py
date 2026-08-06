"""Aggregate quality, system, route, and calibration metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from budgetroute.evaluation.calibration_metrics import (
    brier_score,
    expected_calibration_error,
    reliability_diagram_data,
)
from budgetroute.evaluation.routing_metrics import selective_metrics
from budgetroute.profiling.latency import summarize_latencies
from budgetroute.schemas import PredictionRecord


def aggregate_metrics(predictions: list[PredictionRecord]) -> dict[str, Any]:
    successful = [item for item in predictions if item.error is None]
    qualities = [item.quality_score for item in successful]
    latencies = [item.latency_ms for item in successful]
    latency = summarize_latencies(latencies)
    generation_latency = summarize_latencies(
        [item.generation_ms for item in successful if item.generation_ms > 0]
    )
    retrieval_latency = summarize_latencies(
        [item.retrieval_ms for item in successful if item.retrieval_ms > 0]
    )
    ttft = summarize_latencies(
        [
            item.time_to_first_token_ms
            for item in successful
            if item.time_to_first_token_ms is not None
        ]
    )
    by_category: dict[str, list[float]] = defaultdict(list)
    for item in successful:
        by_category[item.category].append(item.quality_score)
    route_counts = Counter(item.route.value for item in successful)
    total_latency_seconds = sum(latencies) / 1000.0
    outcomes = [int(item.quality_score >= 0.8) for item in successful]
    confidences = [item.confidence for item in successful]
    selective = selective_metrics(qualities, [item.abstained for item in successful])
    return {
        "request_count": len(predictions),
        "successful_requests": len(successful),
        "failed_requests": len(predictions) - len(successful),
        "mean_quality": sum(qualities) / len(qualities) if qualities else None,
        "accuracy": sum(score == 1.0 for score in qualities) / len(qualities)
        if qualities
        else None,
        "quality_by_category": {
            category: sum(scores) / len(scores) for category, scores in sorted(by_category.items())
        },
        "route_distribution": {
            route: count / len(successful) for route, count in sorted(route_counts.items())
        }
        if successful
        else {},
        "route_counts": dict(sorted(route_counts.items())),
        "latency": {
            "count": latency.count,
            "mean_ms": latency.mean_ms,
            "p50_ms": latency.p50_ms,
            "p95_ms": latency.p95_ms,
            "p99_ms": latency.p99_ms,
            "warning": latency.warning,
        },
        "generation_latency": {
            "count": generation_latency.count,
            "mean_ms": generation_latency.mean_ms,
            "p50_ms": generation_latency.p50_ms,
            "p95_ms": generation_latency.p95_ms,
            "warning": generation_latency.warning,
        },
        "retrieval_latency": {
            "count": retrieval_latency.count,
            "mean_ms": retrieval_latency.mean_ms,
            "p50_ms": retrieval_latency.p50_ms,
            "p95_ms": retrieval_latency.p95_ms,
            "warning": retrieval_latency.warning,
        },
        "time_to_first_token": {
            "count": ttft.count,
            "mean_ms": ttft.mean_ms,
            "p50_ms": ttft.p50_ms,
            "p95_ms": ttft.p95_ms,
            "warning": ttft.warning,
        },
        "throughput_requests_per_second": (
            len(successful) / total_latency_seconds if total_latency_seconds else None
        ),
        "input_tokens": sum(item.input_tokens for item in successful),
        "output_tokens": sum(item.output_tokens for item in successful),
        "generated_tokens_per_second": (
            sum(item.output_tokens for item in successful) / total_latency_seconds
            if total_latency_seconds
            else None
        ),
        "process_rss_mb_peak": max(
            (item.process_rss_mb for item in successful if item.process_rss_mb is not None),
            default=None,
        ),
        "peak_cuda_mb": max(
            (item.peak_cuda_mb for item in successful if item.peak_cuda_mb is not None),
            default=None,
        ),
        "escalations": sum(item.escalated for item in successful),
        "abstentions": sum(item.abstained for item in successful),
        **selective,
        "brier_score": brier_score(confidences, outcomes),
        "expected_calibration_error": expected_calibration_error(confidences, outcomes),
        "reliability": reliability_diagram_data(confidences, outcomes),
    }
