"""Learned prediction of whether retrieved context improves small-model quality."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from budgetroute.exceptions import RouterTrainingError
from budgetroute.routing.base import feature_snapshot
from budgetroute.routing.calibration import ProbabilityCalibrator
from budgetroute.routing.heuristic import difficulty_score
from budgetroute.schemas import GenerationRequest, RequestFeatures, RouteDecision, RouteName


class LearnedRetrievalPolicy:
    name = "learned_retrieval"

    def __init__(
        self,
        artifact_path: Path,
        benefit_threshold: float | None,
        difficulty_threshold: float,
    ) -> None:
        self.artifact_path = artifact_path
        self.benefit_threshold = benefit_threshold
        self.difficulty_threshold = difficulty_threshold
        self._artifact: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._artifact is not None:
            return self._artifact
        if importlib.util.find_spec("joblib") is None:
            raise RouterTrainingError(
                "learned retrieval routing requires the learned optional dependency"
            )
        if not self.artifact_path.is_file():
            raise RouterTrainingError(
                f"retrieval-benefit artifact does not exist: {self.artifact_path}"
            )
        import joblib

        artifact = joblib.load(self.artifact_path)
        if (
            not isinstance(artifact, dict)
            or artifact.get("kind") != "retrieval_benefit"
            or not {"model", "feature_order"} <= artifact.keys()
        ):
            raise RouterTrainingError("retrieval-benefit artifact has an invalid schema")
        self._artifact = artifact
        return artifact

    def decide(self, request: GenerationRequest, features: RequestFeatures) -> RouteDecision:
        artifact = self._load()
        numeric = features.numeric_snapshot()
        feature_order = [str(item) for item in artifact["feature_order"]]
        vector = np.asarray([[numeric.get(name, 0.0) for name in feature_order]], dtype=float)
        model = artifact["model"]
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(vector)[0]
            classes = list(model.classes_)
            raw_benefit = (
                float(probabilities[classes.index(1)]) if 1 in classes else float(classes[0] == 1)
            )
        else:
            raw_benefit = float(model.predict(vector)[0])
        calibrator = artifact.get("calibrator", ProbabilityCalibrator())
        benefit = calibrator.transform_one(raw_benefit)
        threshold = float(artifact.get("serving_threshold", 0.5))
        if self.benefit_threshold is not None:
            threshold = self.benefit_threshold
        difficulty = difficulty_score(features)
        if features.relevant_context_found and benefit >= threshold:
            route = RouteName.SMALL_WITH_RETRIEVAL
            reason = "Learned paired evidence predicts that retrieved context improves quality"
        elif difficulty >= self.difficulty_threshold:
            route = RouteName.LARGE
            reason = "Retrieval benefit is insufficient and heuristic difficulty is high"
        else:
            route = RouteName.SMALL
            reason = "Retrieval benefit is below threshold and the small backend is sufficient"
        return RouteDecision(
            route=route,
            confidence=max(benefit, 1.0 - benefit),
            difficulty_score=difficulty,
            reason=reason,
            policy=self.name,
            features={
                **feature_snapshot(features),
                "retrieval_benefit_probability": benefit,
                "retrieval_benefit_probability_raw": raw_benefit,
            },
            thresholds={
                "retrieval_benefit": threshold,
                "difficulty": self.difficulty_threshold,
            },
        )
