from __future__ import annotations

import json
from pathlib import Path

import pytest

from budgetroute.config import AppConfig, MonitoringConfig
from budgetroute.features.request_features import RequestFeatureExtractor
from budgetroute.inference.service import build_service
from budgetroute.monitoring.runtime import RuntimeMonitor
from budgetroute.schemas import FeedbackRecord, GenerationRequest


def test_runtime_monitor_detects_feature_shift_without_retaining_prompts(
    fake_config: AppConfig,
) -> None:
    monitor = RuntimeMonitor(
        MonitoringConfig(window_size=20, minimum_samples=10, drift_threshold=0.2)
    )
    service = build_service(fake_config)
    extractor = RequestFeatureExtractor()
    try:
        for index in range(10):
            request = GenerationRequest(request_id=f"base-{index}", prompt="Short factual question")
            monitor.observe(request, extractor.extract(request), service.generate(request))
        for index in range(10):
            prompt = "Explain in detail " + "very long shifted text " * 100
            request = GenerationRequest(request_id=f"shift-{index}", prompt=prompt)
            monitor.observe(request, extractor.extract(request), service.generate(request))
        monitor.record_feedback(FeedbackRecord(request_id="base-0", correct=True, notes="private"))
        snapshot = monitor.snapshot()
        assert snapshot["drift_detected"] is True
        assert snapshot["drift_method"] == ("mean_and_quantile_shift_plus_category_total_variation")
        assert snapshot["category_distribution"]
        assert snapshot["counters"]["feedback_total"] == 1
        assert "prompt" not in str(snapshot).lower()
        assert "private" not in str(snapshot)
        assert "budgetroute_drift_score" in monitor.prometheus()
    finally:
        service.close()


def test_runtime_monitor_loads_baseline_and_counts_errors(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "features": {
                    "word_count": {"mean": 2, "standard_deviation": 1},
                    "char_count": {"mean": 10, "standard_deviation": 2},
                }
            }
        ),
        encoding="utf-8",
    )
    monitor = RuntimeMonitor(
        MonitoringConfig(
            window_size=20,
            minimum_samples=10,
            drift_threshold=0.2,
            baseline_path=baseline,
        )
    )
    monitor.record_error()
    snapshot = monitor.snapshot()
    assert snapshot["baseline_ready"] is True
    assert snapshot["counters"]["errors_total"] == 1

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="features mapping"):
        RuntimeMonitor(MonitoringConfig(window_size=20, minimum_samples=10, baseline_path=invalid))
