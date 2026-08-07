"""Leakage-safe learned-router training, calibration, persistence, and evaluation."""

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
from budgetroute.routing.calibration import (
    ProbabilityCalibrator,
    choose_threshold,
    fit_probability_calibrator,
)


@dataclass(frozen=True)
class TrainingRows:
    features: list[dict[str, float]]
    labels: list[int]
    ids: list[str]
    groups: list[str]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RouterTrainingError(f"required training artifact does not exist: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_training_rows(artifact_dir: Path, quality_threshold: float) -> TrainingRows:
    predictions = _read_jsonl(artifact_dir / "predictions.jsonl")
    routes = _read_jsonl(artifact_dir / "routes.jsonl")
    route_lookup = {(str(item["example_id"]), str(item.get("policy", ""))): item for item in routes}
    preferred = [item for item in predictions if item.get("policy") == "always_small"]
    selected = preferred or predictions
    features: list[dict[str, float]] = []
    labels: list[int] = []
    ids: list[str] = []
    groups: list[str] = []
    for item in selected:
        if item.get("error"):
            continue
        route = route_lookup.get((str(item["example_id"]), str(item.get("policy", ""))))
        if not route or not isinstance(route.get("features"), dict):
            continue
        numeric = {
            key: float(value)
            for key, value in route["features"].items()
            if isinstance(value, (bool, int, float))
        }
        features.append(numeric)
        labels.append(int(float(item["quality_score"]) >= quality_threshold))
        ids.append(str(item["example_id"]))
        groups.append(str(item.get("group_id") or route.get("group_id") or item["example_id"]))
    if not features:
        raise RouterTrainingError("no joined prediction/feature rows were available for training")
    return TrainingRows(features, labels, ids, groups)


def _classification_metrics(
    labels: list[int], predictions: list[int], probabilities: list[float]
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if not labels:
        raise RouterTrainingError("cannot calculate router metrics for an empty split")
    unique = set(labels)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(unique) == 2 else None,
        "brier_score": brier_score(probabilities, labels),
        "expected_calibration_error": expected_calibration_error(probabilities, labels),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "route_distribution": {
            "small": sum(predictions) / len(predictions),
            "large": 1 - sum(predictions) / len(predictions),
        },
    }


def _probabilities(model: Any, matrix: np.ndarray) -> list[float]:
    raw = model.predict_proba(matrix)
    classes = list(model.classes_)
    return (
        raw[:, classes.index(1)].astype(float).tolist()
        if 1 in classes
        else [float(classes[0] == 1)] * len(matrix)
    )


def _group_partitions(
    groups: list[str], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | None]:
    from sklearn.model_selection import GroupShuffleSplit

    indices = np.arange(len(groups))
    unique_groups = set(groups)
    if len(indices) < 9 or len(unique_groups) < 6:
        return (
            indices,
            indices,
            indices,
            "too few samples or groups for disjoint partitions; all metrics are in-sample",
        )
    group_values = np.asarray(groups)
    first = GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=seed)
    train_indices, held_out_indices = next(first.split(indices, groups=group_values))
    held_out_groups = group_values[held_out_indices]
    if len(set(held_out_groups)) < 2:
        return (
            indices,
            indices,
            indices,
            "group split could not create calibration and test sets; all metrics are in-sample",
        )
    second = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed + 1)
    calibration_relative, test_relative = next(
        second.split(held_out_indices, groups=held_out_groups)
    )
    return (
        train_indices,
        held_out_indices[calibration_relative],
        held_out_indices[test_relative],
        None,
    )


def _split_summary(
    indices: np.ndarray, ids: list[str], groups: list[str], labels: list[int]
) -> dict[str, Any]:
    selected_labels = [labels[int(index)] for index in indices]
    return {
        "sample_count": len(indices),
        "group_count": len({groups[int(index)] for index in indices}),
        "class_counts": {
            "failure": selected_labels.count(0),
            "success": selected_labels.count(1),
        },
        "ids": [ids[int(index)] for index in indices],
        "groups": sorted({groups[int(index)] for index in indices}),
    }


def train_router(
    artifact_dir: Path,
    output_path: Path,
    quality_threshold: float = 0.8,
    seed: int = 42,
    target_selective_accuracy: float = 0.8,
) -> dict[str, Any]:
    try:
        import joblib
        from sklearn.dummy import DummyClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RouterTrainingError(
            "router training requires scikit-learn and joblib; install the learned extra"
        ) from exc

    rows = build_training_rows(artifact_dir, quality_threshold)
    feature_order = sorted({name for item in rows.features for name in item})
    matrix = np.asarray(
        [[item.get(name, 0.0) for name in feature_order] for item in rows.features], dtype=float
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

    raw_calibration_probabilities = _probabilities(model, matrix[calibration_indices])
    calibration_labels = target[calibration_indices].astype(int).tolist()
    calibrator = fit_probability_calibrator(raw_calibration_probabilities, calibration_labels)
    calibrated_probabilities = calibrator.transform(raw_calibration_probabilities)
    threshold_result = choose_threshold(
        calibrated_probabilities, calibration_labels, target_selective_accuracy
    )
    serving_threshold = threshold_result.threshold if threshold_result is not None else 0.5

    raw_test_probabilities = _probabilities(model, matrix[test_indices])
    test_probabilities = calibrator.transform(raw_test_probabilities)
    test_labels = target[test_indices].astype(int).tolist()
    test_predictions = [int(value >= serving_threshold) for value in test_probabilities]
    metrics = _classification_metrics(test_labels, test_predictions, test_probabilities)
    metrics["raw_brier_score"] = brier_score(raw_test_probabilities, test_labels)
    metrics["raw_expected_calibration_error"] = expected_calibration_error(
        raw_test_probabilities, test_labels
    )

    source_hash = hashlib.sha256(
        (artifact_dir / "predictions.jsonl").read_bytes()
        + (artifact_dir / "routes.jsonl").read_bytes()
    ).hexdigest()
    splits = {
        "train": _split_summary(train_indices, rows.ids, rows.groups, rows.labels),
        "calibration": _split_summary(calibration_indices, rows.ids, rows.groups, rows.labels),
        "test": _split_summary(test_indices, rows.ids, rows.groups, rows.labels),
    }
    train_groups = set(splits["train"]["groups"])
    calibration_groups = set(splits["calibration"]["groups"])
    test_groups = set(splits["test"]["groups"])
    disjoint = not (
        train_groups & calibration_groups
        or train_groups & test_groups
        or calibration_groups & test_groups
    )
    if warning is None and not disjoint:
        raise RouterTrainingError("group leakage was detected across router partitions")
    metadata = {
        "schema_version": 2,
        "label_definition": f"small_model_quality >= {quality_threshold}",
        "quality_threshold": quality_threshold,
        "target_selective_accuracy": target_selective_accuracy,
        "serving_threshold": serving_threshold,
        "threshold_selection": asdict(threshold_result) if threshold_result is not None else None,
        "calibrator": asdict(calibrator),
        "seed": seed,
        "dataset_hash": source_hash,
        "sample_count": len(rows.labels),
        "splits": splits,
        "group_disjoint": disjoint,
        "train_ids": splits["train"]["ids"],
        "calibration_ids": splits["calibration"]["ids"],
        "test_ids": splits["test"]["ids"],
        "metrics": metrics,
        "warning": warning or calibrator.warning,
        "package_versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "joblib")
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    joblib.dump(
        {
            "schema_version": 2,
            "model": model,
            "calibrator": calibrator,
            "serving_threshold": serving_threshold,
            "feature_order": feature_order,
            "label_definition": metadata["label_definition"],
            "training_metadata": metadata,
        },
        temporary,
    )
    temporary.replace(output_path)
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def evaluate_router(
    artifact_path: Path,
    artifact_dir: Path,
    threshold: float = 0.8,
    *,
    test_only: bool = True,
) -> dict[str, Any]:
    try:
        import joblib
    except ImportError as exc:
        raise RouterTrainingError("router evaluation requires joblib") from exc
    artifact = joblib.load(artifact_path)
    rows = build_training_rows(artifact_dir, threshold)
    order = [str(item) for item in artifact["feature_order"]]
    matrix = np.asarray([[item.get(name, 0.0) for name in order] for item in rows.features])
    selected = np.arange(len(rows.ids))
    training_metadata = artifact.get("training_metadata", {})
    if test_only and training_metadata.get("test_ids"):
        test_ids = set(str(item) for item in training_metadata["test_ids"])
        selected = np.asarray([index for index, item in enumerate(rows.ids) if item in test_ids])
        if not len(selected):
            raise RouterTrainingError("none of the persisted test IDs exist in these artifacts")
    raw_probabilities = _probabilities(artifact["model"], matrix[selected])
    calibrator = artifact.get("calibrator", ProbabilityCalibrator())
    probabilities = calibrator.transform(raw_probabilities)
    serving_threshold = float(artifact.get("serving_threshold", 0.5))
    labels = [rows.labels[int(index)] for index in selected]
    predictions = [int(value >= serving_threshold) for value in probabilities]
    result = _classification_metrics(labels, predictions, probabilities)
    result.update(
        {
            "evaluation_scope": "test" if test_only else "all",
            "evaluation_ids": [rows.ids[int(index)] for index in selected],
            "serving_threshold": serving_threshold,
            "raw_brier_score": brier_score(raw_probabilities, labels),
            "raw_expected_calibration_error": expected_calibration_error(raw_probabilities, labels),
        }
    )
    return result


def train_backend_confidence_calibrator(
    artifact_dir: Path,
    output_path: Path,
    *,
    policy: str = "always_small",
    quality_threshold: float = 0.8,
    target_selective_accuracy: float = 0.8,
    seed: int = 42,
) -> dict[str, Any]:
    predictions = [
        item
        for item in _read_jsonl(artifact_dir / "predictions.jsonl")
        if item.get("policy") == policy
        and item.get("error") is None
        and item.get("backend_confidence_raw") is not None
    ]
    if not predictions:
        raise RouterTrainingError(
            f"no raw backend-confidence rows were available for policy {policy!r}"
        )
    groups = [str(item.get("group_id") or item["example_id"]) for item in predictions]
    _, calibration_indices, test_indices, warning = _group_partitions(groups, seed)
    raw = [float(item["backend_confidence_raw"]) for item in predictions]
    labels = [int(float(item["quality_score"]) >= quality_threshold) for item in predictions]
    calibration_raw = [raw[int(index)] for index in calibration_indices]
    calibration_labels = [labels[int(index)] for index in calibration_indices]
    calibrator = fit_probability_calibrator(calibration_raw, calibration_labels)
    calibration_probabilities = calibrator.transform(calibration_raw)
    threshold_result = choose_threshold(
        calibration_probabilities, calibration_labels, target_selective_accuracy
    )
    serving_threshold = threshold_result.threshold if threshold_result is not None else 0.5
    test_raw = [raw[int(index)] for index in test_indices]
    test_labels = [labels[int(index)] for index in test_indices]
    test_probabilities = calibrator.transform(test_raw)
    test_predictions = [int(value >= serving_threshold) for value in test_probabilities]
    metadata = {
        "schema_version": 1,
        "kind": "backend_confidence_calibrator",
        "policy": policy,
        "quality_threshold": quality_threshold,
        "target_selective_accuracy": target_selective_accuracy,
        "serving_threshold": serving_threshold,
        "calibrator": asdict(calibrator),
        "calibration_ids": [predictions[int(index)]["example_id"] for index in calibration_indices],
        "test_ids": [predictions[int(index)]["example_id"] for index in test_indices],
        "metrics": _classification_metrics(test_labels, test_predictions, test_probabilities),
        "warning": warning or calibrator.warning,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def load_backend_confidence_calibrator(
    path: Path,
) -> tuple[ProbabilityCalibrator, float, dict[str, Any]]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        calibrator = ProbabilityCalibrator(**metadata["calibrator"])
        threshold = float(metadata["serving_threshold"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RouterTrainingError(f"invalid backend confidence calibrator {path}: {exc}") from exc
    return calibrator, threshold, metadata
