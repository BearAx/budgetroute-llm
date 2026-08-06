"""Learned-router label construction, training, persistence, and evaluation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np

from budgetroute.evaluation.calibration_metrics import (
    brier_score,
    expected_calibration_error,
)
from budgetroute.exceptions import RouterTrainingError


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RouterTrainingError(f"required training artifact does not exist: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_training_rows(
    artifact_dir: Path, quality_threshold: float
) -> tuple[list[dict[str, float]], list[int], list[str]]:
    predictions = _read_jsonl(artifact_dir / "predictions.jsonl")
    routes = _read_jsonl(artifact_dir / "routes.jsonl")
    route_lookup = {(str(item["example_id"]), str(item.get("policy", ""))): item for item in routes}
    preferred = [item for item in predictions if item.get("policy") == "always_small"]
    selected = preferred or predictions
    features: list[dict[str, float]] = []
    labels: list[int] = []
    ids: list[str] = []
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
    if not features:
        raise RouterTrainingError("no joined prediction/feature rows were available for training")
    return features, labels, ids


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


def train_router(
    artifact_dir: Path,
    output_path: Path,
    quality_threshold: float = 0.8,
    seed: int = 42,
) -> dict[str, Any]:
    try:
        import joblib
        from sklearn.dummy import DummyClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RouterTrainingError(
            "router training requires scikit-learn and joblib; install the learned extra"
        ) from exc

    feature_dicts, labels, ids = build_training_rows(artifact_dir, quality_threshold)
    feature_order = sorted({name for item in feature_dicts for name in item})
    matrix = np.asarray(
        [[item.get(name, 0.0) for name in feature_order] for item in feature_dicts], dtype=float
    )
    target = np.asarray(labels, dtype=int)
    warning: str | None = None
    indices = np.arange(len(labels))
    class_counts = {value: labels.count(value) for value in set(labels)}
    if len(labels) >= 8 and len(class_counts) == 2 and min(class_counts.values()) >= 2:
        train_indices, test_indices = train_test_split(
            indices, test_size=0.25, random_state=seed, stratify=target
        )
    else:
        train_indices = test_indices = indices
        warning = (
            "tiny or one-class dataset: evaluation is in-sample and not generalization evidence"
        )
    if len(class_counts) == 1:
        model: Any = DummyClassifier(strategy="constant", constant=labels[0])
    else:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        )
    model.fit(matrix[train_indices], target[train_indices])
    predicted = model.predict(matrix[test_indices]).astype(int).tolist()
    probabilities_raw = model.predict_proba(matrix[test_indices])
    classes = list(model.classes_)
    probabilities = (
        probabilities_raw[:, classes.index(1)].astype(float).tolist()
        if 1 in classes
        else [float(classes[0] == 1)] * len(test_indices)
    )
    metrics = _classification_metrics(target[test_indices].tolist(), predicted, probabilities)
    source_hash = hashlib.sha256(
        (artifact_dir / "predictions.jsonl").read_bytes()
        + (artifact_dir / "routes.jsonl").read_bytes()
    ).hexdigest()
    metadata = {
        "label_definition": f"small_model_quality >= {quality_threshold}",
        "quality_threshold": quality_threshold,
        "seed": seed,
        "dataset_hash": source_hash,
        "sample_count": len(labels),
        "train_ids": [ids[int(index)] for index in train_indices],
        "validation_ids": [ids[int(index)] for index in test_indices],
        "metrics": metrics,
        "warning": warning,
        "package_versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "joblib")
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    joblib.dump(
        {
            "model": model,
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
    artifact_path: Path, artifact_dir: Path, threshold: float = 0.8
) -> dict[str, Any]:
    try:
        import joblib
    except ImportError as exc:
        raise RouterTrainingError("router evaluation requires joblib") from exc
    artifact = joblib.load(artifact_path)
    feature_dicts, labels, _ = build_training_rows(artifact_dir, threshold)
    order = [str(item) for item in artifact["feature_order"]]
    matrix = np.asarray([[item.get(name, 0.0) for name in order] for item in feature_dicts])
    model = artifact["model"]
    predictions = model.predict(matrix).astype(int).tolist()
    raw = model.predict_proba(matrix)
    classes = list(model.classes_)
    probabilities = (
        raw[:, classes.index(1)].astype(float).tolist()
        if 1 in classes
        else [float(classes[0] == 1)] * len(labels)
    )
    return _classification_metrics(labels, predictions, probabilities)
