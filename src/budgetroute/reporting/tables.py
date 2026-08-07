"""CSV policy summaries derived from metrics artifacts."""

from __future__ import annotations

import csv
import io
from typing import Any


def policy_csv(metrics: dict[str, Any]) -> str:
    buffer = io.StringIO(newline="")
    fields = [
        "policy",
        "requests",
        "mean_quality",
        "p50_latency_ms",
        "p95_latency_ms",
        "throughput_requests_per_second",
        "escalations",
        "abstentions",
        "human_review_requests",
        "estimated_cost_units",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for name, values in sorted(metrics.get("policies", {}).items()):
        latency = values.get("latency", {})
        writer.writerow(
            {
                "policy": name,
                "requests": values.get("request_count"),
                "mean_quality": values.get("mean_quality"),
                "p50_latency_ms": latency.get("p50_ms"),
                "p95_latency_ms": latency.get("p95_ms"),
                "throughput_requests_per_second": values.get("throughput_requests_per_second"),
                "escalations": values.get("escalations"),
                "abstentions": values.get("abstentions"),
                "human_review_requests": values.get("human_review_requests"),
                "estimated_cost_units": values.get("estimated_cost_units"),
            }
        )
    return buffer.getvalue()
