"""Leakage-safe learned retrieval-benefit artifact construction."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from budgetroute.evaluation.calibration_metrics import (
    brier_score,
    expected_calibration_error,
)
from budgetroute.exceptions import RouterTrainingError
from budgetroute.routing.calibration import choose_threshold, fit_probability_calibrator
from budgetroute.routing.training import (
    _classification_metrics,
    _group_partitions,
    _probabilities,
    _read_jsonl,
    _split_summary,
)


@dataclass(frozen=True)
class RetrievalTrainingRows:
    features: list[dict[str, float]]
    labels: list[int]
    ids: list[str]
    groups: list[str]
    quality_deltas: list[float]


def build_retrieval_training_rows(
    artifact_dir: Path,
    *,
    baseline_policy: str = "always_small",
    retrieval_policy: str = "retrieval_first",
    minimum_quality_improvement: float = 0.01,
) -> RetrievalTrainingRows:
    predictions = _read_jsonl(artifact_dir / "predictions.jsonl")
    routes = _read_jsonl(artifact_dir / "routes.jsonl")
    prediction_lookup = {
        (str(item["example_id"]), str(item.get("policy", ""))): item
        for item in predictions
        if not item.get("error")
    }
    route_lookup = {(str(item["example_id"]), str(item.get("policy", ""))): item for item in routes}
    example_ids = sorted(
        example_id
        for example_id, policy in prediction_lookup
        if policy == baseline_policy
        and (example_id, retrieval_policy) in prediction_lookup
        and (example_id, retrieval_policy) in route_lookup
    )
    features: list[dict[str, float]] = []
    labels: list[int] = []
    groups: list[str] = []
    deltas: list[float] = []
    selected_ids: list[str] = []
    for example_id in example_ids:
        baseline = prediction_lookup[(example_id, baseline_policy)]
        retrieved = prediction_lookup[(example_id, retrieval_policy)]
        route = route_lookup[(example_id, retrieval_policy)]
        raw_features = route.get("features")
        if not isinstance(raw_features, dict):
            continue
        numeric = {
            str(name): float(value)
            for name, value in raw_features.items()
            if isinstance(value, (bool, int, float))
        }
        delta = float(retrieved["quality_score"]) - float(baseline["quality_score"])
        features.append(numeric)
        labels.append(int(delta >= minimum_quality_improvement))
        selected_ids.append(example_id)
        groups.append(str(retrieved.get("group_id") or example_id))
        deltas.append(delta)
    if not features:
        raise RouterTrainingError(
            "no paired always-small/retrieval-first prediction rows were available"
        )
    return RetrievalTrainingRows(features, labels, selected_ids, groups, deltas)


def train_retrieval_benefit_router(
    artifact_dir: Path,
    output_path: Path,
    *,
    minimum_quality_improvement: float = 0.01,
    seed: int = 42,
    target_precision: float = 0.8,
) -> dict[str, Any]:
    try:
        import joblib
        from sklearn.dummy import DummyClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RouterTrainingError(
            "retrieval-benefit training requires scikit-learn and joblib"
        ) from exc
    rows = build_retrieval_training_rows(
        artifact_dir, minimum_quality_improvement=minimum_quality_improvement
    )
    feature_order = sorted({name for row in rows.features for name in row})
    matrix = np.asarray(
        [[row.get(name, 0.0) for name in feature_order] for row in rows.features], dtype=float
    )
    target = np.asarray(rows.labels, dtype=int)
    train_indices, calibration_indices, test_indices, warning = _group_partitions(rows.groups, seed)
    train_labels = target[train_indices]
    if len(set(train_labels.tolist())) == 1:
        model: Any = DummyClassifier(strategy="constant", constant=int(train_labels[0]))
        warning = warning or "training split contained one class; constant classifier used"
    else:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        )
    model.fit(matrix[train_indices], train_labels)
    raw_calibration = _probabilities(model, matrix[calibration_indices])
    calibration_labels = target[calibration_indices].astype(int).tolist()
    calibrator = fit_probability_calibrator(raw_calibration, calibration_labels)
    calibration_probabilities = calibrator.transform(raw_calibration)
    threshold_result = choose_threshold(
        calibration_probabilities, calibration_labels, target_precision
    )
    serving_threshold = threshold_result.threshold if threshold_result is not None else 0.5
    raw_test = _probabilities(model, matrix[test_indices])
    test_probabilities = calibrator.transform(raw_test)
    test_labels = target[test_indices].astype(int).tolist()
    test_predictions = [int(value >= serving_threshold) for value in test_probabilities]
    metrics = _classification_metrics(test_labels, test_predictions, test_probabilities)
    metrics["raw_brier_score"] = brier_score(raw_test, test_labels)
    metrics["raw_expected_calibration_error"] = expected_calibration_error(raw_test, test_labels)
    splits = {
        "train": _split_summary(train_indices, rows.ids, rows.groups, rows.labels),
        "calibration": _split_summary(calibration_indices, rows.ids, rows.groups, rows.labels),
        "test": _split_summary(test_indices, rows.ids, rows.groups, rows.labels),
    }
    source_hash = hashlib.sha256(
        (artifact_dir / "predictions.jsonl").read_bytes()
        + (artifact_dir / "routes.jsonl").read_bytes()
    ).hexdigest()
    metadata = {
        "schema_version": 1,
        "kind": "retrieval_benefit",
        "label_definition": (
            f"retrieval_quality - always_small_quality >= {minimum_quality_improvement}"
        ),
        "minimum_quality_improvement": minimum_quality_improvement,
        "target_precision": target_precision,
        "serving_threshold": serving_threshold,
        "threshold_selection": asdict(threshold_result) if threshold_result else None,
        "calibrator": asdict(calibrator),
        "sample_count": len(rows.labels),
        "positive_count": sum(rows.labels),
        "mean_quality_delta": sum(rows.quality_deltas) / len(rows.quality_deltas),
        "splits": splits,
        "metrics": metrics,
        "source_hash": source_hash,
        "seed": seed,
        "warning": warning or calibrator.warning,
        "package_versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "joblib")
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    joblib.dump(
        {
            "schema_version": 1,
            "kind": "retrieval_benefit",
            "model": model,
            "calibrator": calibrator,
            "serving_threshold": serving_threshold,
            "feature_order": feature_order,
            "training_metadata": metadata,
        },
        temporary,
    )
    temporary.replace(output_path)
    output_path.with_suffix(output_path.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata
