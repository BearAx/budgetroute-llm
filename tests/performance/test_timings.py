from __future__ import annotations

from budgetroute.config import AppConfig
from budgetroute.inference.service import build_service
from budgetroute.profiling.latency import summarize_latencies
from budgetroute.schemas import GenerationRequest


def test_fake_delay_populates_ordered_percentiles(fake_config: AppConfig) -> None:
    service = build_service(fake_config)
    try:
        timings = [
            service.generate(GenerationRequest(prompt=f"Simple request {index}")).timing.total_ms
            for index in range(5)
        ]
    finally:
        service.close()
    summary = summarize_latencies(timings)
    assert all(value > 0 for value in timings)
    assert summary.p50_ms is not None and summary.p95_ms is not None
    assert summary.p50_ms <= summary.p95_ms
