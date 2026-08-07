from __future__ import annotations

import json
from pathlib import Path

import pytest

from budgetroute.adaptation.online import CalibrationRegistry, page_hinkley
from budgetroute.config import AdaptationConfig
from budgetroute.exceptions import RouterTrainingError
from budgetroute.operations.models import LabeledObservation


def _observations(count: int) -> list[LabeledObservation]:
    return [
        LabeledObservation(
            tenant_id="tenant-a",
            request_id=f"request-{index:03d}",
            route="small",
            backend_confidence_raw=0.99 if index % 2 == 0 else 0.8,
            correct=index % 2 == 0,
            features={"token_count": float(index + 1)},
            predicted_at_epoch=float(index),
            labeled_at_epoch=float(index + 100),
        )
        for index in range(count)
    ]


def test_online_calibration_uses_chronological_holdout_and_gated_promotion(
    tmp_path: Path,
) -> None:
    config = AdaptationConfig(
        enabled=True,
        registry_path=tmp_path / "registry",
        minimum_labeled_samples=20,
        minimum_brier_improvement=0.001,
    )
    registry = CalibrationRegistry(config)

    first = registry.adapt(_observations(40))
    assert first["promoted"] is True
    assert config.registry_path.joinpath("active-calibrator.json").is_file()
    active = json.loads(
        config.registry_path.joinpath("active-calibrator.json").read_text(encoding="utf-8")
    )
    assert active["chronological_split"] is True
    assert active["holdout_count"] == 10

    second = registry.adapt(_observations(40))
    assert second["promoted"] is False
    assert second["active_version"] == first["version"]
    with pytest.raises(RouterTrainingError, match="no previous"):
        registry.rollback()


def test_page_hinkley_detects_an_upward_error_change() -> None:
    changes = page_hinkley([0.0] * 30 + [1.0] * 30, delta=0.01, threshold=3.0)
    assert changes
    assert changes[0] >= 30
