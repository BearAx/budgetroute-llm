"""Chronological online-calibration candidates, promotion gates, and rollback."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from budgetroute.config import AdaptationConfig
from budgetroute.evaluation.calibration_metrics import (
    brier_score,
    expected_calibration_error,
)
from budgetroute.exceptions import RouterTrainingError
from budgetroute.operations.models import LabeledObservation
from budgetroute.routing.calibration import (
    ProbabilityCalibrator,
    choose_threshold,
    fit_probability_calibrator,
)


def page_hinkley(values: list[float], *, delta: float, threshold: float) -> list[int]:
    """Return indices where an upward mean change is detected."""
    if not values:
        return []
    mean = 0.0
    cumulative = 0.0
    minimum = 0.0
    changes: list[int] = []
    for index, value in enumerate(values, start=1):
        mean += (value - mean) / index
        cumulative += value - mean - delta
        minimum = min(minimum, cumulative)
        if cumulative - minimum > threshold:
            changes.append(index - 1)
            cumulative = 0.0
            minimum = 0.0
            mean = value
    return changes


class CalibrationRegistry:
    """Persist immutable candidates and atomically materialize the active calibrator."""

    def __init__(self, config: AdaptationConfig) -> None:
        self.config = config
        self.root = config.registry_path
        self.manifest_path = self.root / "manifest.json"
        self.active_path = self.root / "active-calibrator.json"

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {"schema_version": 1, "active_version": None, "versions": []}
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RouterTrainingError(f"invalid calibration registry manifest: {exc}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("versions"), list):
            raise RouterTrainingError("calibration registry manifest has an invalid schema")
        return value

    def _load_active_calibrator(self) -> ProbabilityCalibrator:
        if not self.active_path.is_file():
            return ProbabilityCalibrator()
        try:
            payload = json.loads(self.active_path.read_text(encoding="utf-8"))
            return ProbabilityCalibrator(**payload["calibrator"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RouterTrainingError(f"invalid active calibration artifact: {exc}") from exc

    def adapt(self, observations: list[LabeledObservation]) -> dict[str, Any]:
        ordered = sorted(observations, key=lambda item: (item.labeled_at_epoch, item.request_id))
        if len(ordered) < self.config.minimum_labeled_samples:
            raise RouterTrainingError(
                f"online adaptation requires at least {self.config.minimum_labeled_samples} "
                f"labeled observations; received {len(ordered)}"
            )
        holdout_count = max(1, int(len(ordered) * self.config.holdout_fraction))
        split = len(ordered) - holdout_count
        training = ordered[:split]
        holdout = ordered[split:]
        training_raw = [item.backend_confidence_raw for item in training]
        training_labels = [int(item.correct) for item in training]
        holdout_raw = [item.backend_confidence_raw for item in holdout]
        holdout_labels = [int(item.correct) for item in holdout]
        candidate = fit_probability_calibrator(training_raw, training_labels)
        candidate_holdout = candidate.transform(holdout_raw)
        active = self._load_active_calibrator()
        active_holdout = active.transform(holdout_raw)
        candidate_brier = brier_score(candidate_holdout, holdout_labels)
        active_brier = brier_score(active_holdout, holdout_labels)
        candidate_ece = expected_calibration_error(candidate_holdout, holdout_labels)
        active_ece = expected_calibration_error(active_holdout, holdout_labels)
        assert candidate_brier is not None and active_brier is not None
        assert candidate_ece is not None and active_ece is not None
        manifest = self._manifest()
        initial = manifest.get("active_version") is None
        required_improvement = 0.0 if initial else self.config.minimum_brier_improvement
        promoted = (
            candidate_brier <= active_brier - required_improvement
            and candidate_ece <= active_ece + self.config.maximum_ece_regression
            and candidate.warning is None
        )
        calibrated_training = candidate.transform(training_raw)
        threshold = choose_threshold(
            calibrated_training,
            training_labels,
            self.config.target_selective_accuracy,
        )
        serving_threshold = threshold.threshold if threshold is not None else 0.5
        source_payload = [item.model_dump(mode="json") for item in ordered]
        source_hash = hashlib.sha256(
            json.dumps(source_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        version = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid4().hex[:8]
        changes = page_hinkley(
            [float(not item.correct) for item in ordered],
            delta=self.config.change_point_delta,
            threshold=self.config.change_point_threshold,
        )
        artifact = {
            "schema_version": 2,
            "kind": "online_backend_confidence_calibrator",
            "version": version,
            "created_at_epoch": time.time(),
            "source_hash": source_hash,
            "sample_count": len(ordered),
            "training_count": len(training),
            "holdout_count": len(holdout),
            "chronological_split": True,
            "calibrator": asdict(candidate),
            "serving_threshold": serving_threshold,
            "threshold_selection": asdict(threshold) if threshold is not None else None,
            "candidate_metrics": {
                "brier_score": candidate_brier,
                "expected_calibration_error": candidate_ece,
            },
            "active_baseline_metrics": {
                "brier_score": active_brier,
                "expected_calibration_error": active_ece,
            },
            "promotion_gates": {
                "minimum_brier_improvement": required_improvement,
                "maximum_ece_regression": self.config.maximum_ece_regression,
                "passed": promoted,
            },
            "change_points": changes,
            "warning": candidate.warning,
        }
        version_path = self.root / "versions" / f"{version}.json"
        self._atomic_json(version_path, artifact)
        entry = {
            "version": version,
            "path": str(version_path),
            "promoted": promoted,
            "source_hash": source_hash,
            "candidate_brier_score": candidate_brier,
            "created_at_epoch": artifact["created_at_epoch"],
        }
        manifest["versions"].append(entry)
        if promoted:
            manifest["active_version"] = version
            self._atomic_json(self.active_path, artifact)
        self._atomic_json(self.manifest_path, manifest)
        return {
            "version": version,
            "promoted": promoted,
            "active_version": manifest.get("active_version"),
            "artifact": str(version_path),
            "active_artifact": str(self.active_path) if promoted else None,
            "candidate_metrics": artifact["candidate_metrics"],
            "baseline_metrics": artifact["active_baseline_metrics"],
            "change_points": changes,
        }

    def rollback(self, version: str | None = None) -> dict[str, Any]:
        manifest = self._manifest()
        promoted_versions = [
            str(item["version"]) for item in manifest["versions"] if item.get("promoted")
        ]
        if not promoted_versions:
            raise RouterTrainingError("calibration registry has no promoted version to restore")
        active = manifest.get("active_version")
        if version is None:
            candidates = [item for item in promoted_versions if item != active]
            if not candidates:
                raise RouterTrainingError("calibration registry has no previous promoted version")
            version = candidates[-1]
        if version not in promoted_versions:
            raise RouterTrainingError(f"calibration version is not promotable: {version}")
        version_path = self.root / "versions" / f"{version}.json"
        try:
            artifact = json.loads(version_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RouterTrainingError(
                f"cannot restore calibration version {version}: {exc}"
            ) from exc
        self._atomic_json(self.active_path, artifact)
        manifest["active_version"] = version
        manifest.setdefault("rollbacks", []).append(
            {"from": active, "to": version, "occurred_at_epoch": time.time()}
        )
        self._atomic_json(self.manifest_path, manifest)
        return {"rolled_back": True, "from": active, "to": version}
