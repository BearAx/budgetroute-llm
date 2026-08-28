"""Health, readiness, configuration, routing, and generation endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import PlainTextResponse

from budgetroute.api.dependencies import (
    MonitorDependency,
    PrincipalDependency,
    ServiceDependency,
    StoreDependency,
)
from budgetroute.inference.batching import AsyncInferenceBatcher
from budgetroute.operations.models import (
    PredictionObservation,
    ReviewCase,
    ReviewResolution,
    ReviewStatus,
)
from budgetroute.schemas import FeedbackRecord, GenerationRequest, GenerationResponse, RouteDecision

router = APIRouter()


@router.get("/healthz", summary="Process liveness")
def healthz() -> dict[str, str]:
    """Return liveness without initializing or probing model hardware."""
    return {"status": "ok"}


@router.get("/readyz", summary="Dependency readiness")
def readyz(
    response: Response, service: ServiceDependency, store: StoreDependency
) -> dict[str, object]:
    inference = service.health()
    operations = store.health()
    ready = bool(inference["ready"]) and bool(operations["ready"])
    if not ready:
        response.status_code = 503
    return {
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "components": {
            "inference": {"ready": bool(inference["ready"])},
            "operations": operations,
        },
    }


@router.get("/v1/config", summary="Sanitized active configuration")
def active_config(request: Request, service: ServiceDependency) -> dict[str, object]:
    summary = service.config.sanitized_summary()
    summary["authentication"] = request.app.state.authenticator.sanitized_summary()
    return summary


@router.post("/v1/route", response_model=RouteDecision, summary="Choose an execution route")
def route_request(payload: GenerationRequest, service: ServiceDependency) -> RouteDecision:
    if len(payload.prompt) > service.config.api.max_prompt_chars:
        raise HTTPException(status_code=413, detail="prompt exceeds the configured character limit")
    return service.route(payload).decision


@router.post(
    "/v1/generate",
    response_model=GenerationResponse,
    response_model_exclude_none=True,
    summary="Route and generate an answer",
)
async def generate(
    payload: GenerationRequest,
    request: Request,
    service: ServiceDependency,
    monitor: MonitorDependency,
    store: StoreDependency,
    principal: PrincipalDependency,
) -> GenerationResponse:
    if len(payload.prompt) > service.config.api.max_prompt_chars:
        raise HTTPException(status_code=413, detail="prompt exceeds the configured character limit")
    scheduler: AsyncInferenceBatcher | None = request.app.state.inference_batcher
    response = (
        await scheduler.submit(payload)
        if scheduler is not None
        else await asyncio.to_thread(service.generate, payload)
    )
    features = service.engine.feature_extractor.extract(payload, response.retrieval)
    monitor.observe(payload, features, response)
    await asyncio.to_thread(
        store.record_prediction,
        PredictionObservation(
            tenant_id=principal.tenant_id,
            request_id=response.request_id,
            route=response.route.value,
            backend_confidence_raw=response.backend_confidence_raw,
            backend_confidence=response.backend_confidence,
            features=features.numeric_snapshot(),
        ),
    )
    await asyncio.to_thread(store.increment_metric, "requests_total")
    await asyncio.to_thread(store.increment_metric, f"route_{response.route.value}_total")
    if response.execution.human_review_required and service.config.api.review_enabled:
        await asyncio.to_thread(
            store.create_review_case,
            principal.tenant_id,
            principal.subject,
            response.request_id,
            response.execution.human_review_reason or response.router.reason,
        )
    public_execution = response.execution.model_copy(update={"initial_answer": None})
    return response.model_copy(update={"execution": public_execution})


@router.get("/metrics", response_class=PlainTextResponse, summary="Prometheus metrics")
def metrics(
    request: Request,
    service: ServiceDependency,
    monitor: MonitorDependency,
    store: StoreDependency,
) -> str:
    if not service.config.api.metrics_enabled:
        raise HTTPException(status_code=404, detail="metrics endpoint is disabled")
    scheduler: AsyncInferenceBatcher | None = request.app.state.inference_batcher
    return monitor.prometheus(
        scheduler.metrics() if scheduler is not None else None,
        store.metric_snapshot(),
    )


@router.get("/v1/monitoring", summary="Sanitized runtime monitoring snapshot")
def monitoring(
    request: Request,
    service: ServiceDependency,
    monitor: MonitorDependency,
    store: StoreDependency,
) -> dict[str, object]:
    scheduler: AsyncInferenceBatcher | None = request.app.state.inference_batcher
    return {
        "monitoring": monitor.snapshot(),
        "scheduler": scheduler.metrics() if scheduler is not None else None,
        "backend_load": {
            name.value: snapshot.model_dump() for name, snapshot in service.load_snapshot().items()
        },
        "shared_metrics": store.metric_snapshot(),
        "shared_feature_summary": store.prediction_feature_summary(),
    }


@router.post("/v1/feedback", status_code=202, summary="Record aggregate correctness feedback")
def feedback(
    payload: FeedbackRecord,
    service: ServiceDependency,
    monitor: MonitorDependency,
    store: StoreDependency,
    principal: PrincipalDependency,
) -> dict[str, object]:
    if not service.config.api.feedback_enabled:
        raise HTTPException(status_code=404, detail="feedback endpoint is disabled")
    inserted = store.record_feedback(principal.tenant_id, principal.subject, payload)
    if inserted:
        monitor.record_feedback(payload)
        store.increment_metric("feedback_total")
        store.increment_metric("feedback_correct_total", float(payload.correct))
    return {
        "accepted": True,
        "idempotent_replay": not inserted,
        "retained_fields": ["request_id", "correct"],
    }


@router.get("/v1/reviews", response_model=list[ReviewCase], summary="List durable review cases")
def list_reviews(
    service: ServiceDependency,
    store: StoreDependency,
    principal: PrincipalDependency,
    status: ReviewStatus | None = None,
    after_sequence: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[ReviewCase]:
    if not service.config.api.review_enabled:
        raise HTTPException(status_code=404, detail="review workflow is disabled")
    tenant_filter = None if principal.permits("admin") else principal.tenant_id
    return store.list_review_cases(
        tenant_filter, status=status, after_sequence=after_sequence, limit=limit
    )


@router.post("/v1/reviews/{case_id}/claim", response_model=ReviewCase)
def claim_review(
    case_id: str,
    service: ServiceDependency,
    store: StoreDependency,
    principal: PrincipalDependency,
) -> ReviewCase:
    if not service.config.api.review_enabled:
        raise HTTPException(status_code=404, detail="review workflow is disabled")
    tenant_filter = None if principal.permits("admin") else principal.tenant_id
    review = store.claim_review_case(case_id, tenant_filter, principal.subject)
    if review is None:
        raise HTTPException(status_code=409, detail="review case is unavailable or already claimed")
    return review


@router.post("/v1/reviews/{case_id}/resolve", response_model=ReviewCase)
def resolve_review(
    case_id: str,
    payload: ReviewResolution,
    service: ServiceDependency,
    store: StoreDependency,
    principal: PrincipalDependency,
) -> ReviewCase:
    if not service.config.api.review_enabled:
        raise HTTPException(status_code=404, detail="review workflow is disabled")
    tenant_filter = None if principal.permits("admin") else principal.tenant_id
    review = store.resolve_review_case(
        case_id,
        tenant_filter,
        principal.subject,
        payload.outcome,
        payload.correct,
        payload.expected_version,
    )
    if review is None:
        raise HTTPException(status_code=409, detail="review transition or version is invalid")
    return review


@router.get("/v1/audit", summary="List tamper-evident operational audit events")
def audit_events(
    store: StoreDependency,
    after_sequence: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, object]:
    events = store.list_audit_events(after_sequence, limit)
    return {
        "events": [event.model_dump(mode="json") for event in events],
        "next_sequence": events[-1].sequence if events else after_sequence,
    }


@router.get("/v1/audit/verify", summary="Verify the retained audit hash chain")
def verify_audit(store: StoreDependency) -> dict[str, object]:
    return store.verify_audit_chain()
