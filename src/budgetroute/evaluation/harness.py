"""Evaluate a configured inference service on validated benchmark records."""

from __future__ import annotations

from budgetroute.evaluation.evaluators import evaluate_answer
from budgetroute.inference.service import InferenceService
from budgetroute.schemas import BenchmarkRecord, GenerationRequest, PredictionRecord, RouteName


def evaluate_records(
    service: InferenceService, records: list[BenchmarkRecord], *, fake: bool
) -> tuple[list[PredictionRecord], list[dict[str, object]], list[dict[str, object]]]:
    predictions: list[PredictionRecord] = []
    routes: list[dict[str, object]] = []
    timings: list[dict[str, object]] = []
    for record in records:
        metadata = dict(record.metadata)
        metadata["must_abstain"] = record.must_abstain
        if fake:
            metadata["fake_reference_answer"] = record.reference_answer
            metadata.setdefault(
                "fake_small_success",
                record.category in {"factual", "classification", "simple_reasoning"}
                and not record.requires_retrieval,
            )
        request = GenerationRequest(
            request_id=record.id,
            prompt=record.prompt,
            requires_retrieval=record.requires_retrieval,
            metadata=metadata,
        )
        try:
            response = service.generate(request)
            evaluation = evaluate_answer(response.answer, record)
            predictions.append(
                PredictionRecord(
                    example_id=record.id,
                    category=record.category,
                    policy=response.router.policy,
                    route=response.route,
                    backend=response.execution.final_backend,
                    raw_answer=response.answer,
                    normalized_answer=evaluation.normalized_answer,
                    reference=record.reference_answer,
                    quality_score=evaluation.score,
                    metric_name=evaluation.metric_name,
                    latency_ms=response.timing.total_ms,
                    generation_ms=response.timing.generation_ms,
                    retrieval_ms=response.timing.retrieval_ms,
                    time_to_first_token_ms=response.timing.time_to_first_token_ms,
                    input_tokens=response.usage.input_tokens
                    + response.usage.escalation_input_tokens,
                    output_tokens=response.usage.output_tokens
                    + response.usage.escalation_output_tokens,
                    process_rss_mb=response.memory.process_rss_mb,
                    peak_cuda_mb=response.memory.peak_cuda_mb,
                    retrieval_used=response.execution.retrieval_used,
                    escalated=response.execution.escalated,
                    abstained=response.execution.abstained,
                    confidence=response.backend_confidence or response.router.confidence,
                )
            )
            routes.append(
                {
                    "example_id": record.id,
                    "policy": response.router.policy,
                    "route": response.route.value,
                    "confidence": response.router.confidence,
                    "features": service.engine.route(request).decision.features,
                    "escalated": response.execution.escalated,
                }
            )
            timings.append({"example_id": record.id, **response.timing.model_dump(mode="json")})
        except Exception as exc:
            predictions.append(
                PredictionRecord(
                    example_id=record.id,
                    category=record.category,
                    policy=service.engine.policy.name,
                    route=RouteName.ABSTAIN,
                    backend=None,
                    raw_answer="",
                    normalized_answer="",
                    reference=record.reference_answer,
                    quality_score=0.0,
                    metric_name=record.evaluation_type.value,
                    latency_ms=0.0,
                    input_tokens=0,
                    output_tokens=0,
                    retrieval_used=False,
                    escalated=False,
                    abstained=True,
                    confidence=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return predictions, routes, timings
