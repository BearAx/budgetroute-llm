from __future__ import annotations

import json
from pathlib import Path

from budgetroute.routing.retrieval_learned import LearnedRetrievalPolicy
from budgetroute.routing.retrieval_training import (
    build_retrieval_training_rows,
    train_retrieval_benefit_router,
)
from budgetroute.schemas import GenerationRequest, RequestFeatures, RouteName


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_paired_retrieval_benefit_training_and_policy(tmp_path: Path) -> None:
    predictions: list[dict[str, object]] = []
    routes: list[dict[str, object]] = []
    for index in range(12):
        example_id = f"example-{index:02d}"
        helps = index % 2 == 0
        predictions.extend(
            [
                {
                    "example_id": example_id,
                    "group_id": f"group-{index:02d}",
                    "policy": "always_small",
                    "quality_score": 0.0 if helps else 1.0,
                    "error": None,
                },
                {
                    "example_id": example_id,
                    "group_id": f"group-{index:02d}",
                    "policy": "retrieval_first",
                    "quality_score": 1.0,
                    "error": None,
                },
            ]
        )
        routes.append(
            {
                "example_id": example_id,
                "group_id": f"group-{index:02d}",
                "policy": "retrieval_first",
                "features": {
                    "retrieval_similarity": 0.95 if helps else 0.05,
                    "retrieval_margin": 0.5 if helps else 0.01,
                    "relevant_context_found": helps,
                    "token_count": 20 + index,
                },
            }
        )
    _write_jsonl(tmp_path / "predictions.jsonl", predictions)
    _write_jsonl(tmp_path / "routes.jsonl", routes)
    rows = build_retrieval_training_rows(tmp_path)
    assert rows.labels.count(1) == 6
    artifact = tmp_path / "retrieval.joblib"
    metadata = train_retrieval_benefit_router(tmp_path, artifact, seed=7)
    assert metadata["sample_count"] == 12
    assert artifact.is_file()

    policy = LearnedRetrievalPolicy(artifact, None, 0.8)
    features = RequestFeatures(
        char_count=20,
        word_count=4,
        token_count=20,
        line_count=1,
        sentence_count=1,
        digit_ratio=0.0,
        punctuation_count=1,
        question_mark_count=1,
        has_code_block=False,
        has_math_symbols=False,
        has_url=False,
        requests_long_output=False,
        category="factual",
        retrieval_similarity=0.99,
        retrieval_margin=0.8,
        relevant_context_found=True,
    )
    decision = policy.decide(GenerationRequest(prompt="Use the retrieved context"), features)
    assert decision.route == RouteName.SMALL_WITH_RETRIEVAL
    assert "retrieval_benefit_probability" in decision.features
    assert decision.thresholds["retrieval_benefit"] == metadata["serving_threshold"]
