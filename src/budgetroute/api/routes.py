"""Health, readiness, configuration, routing, and generation endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse

from budgetroute.api.dependencies import MonitorDependency, ServiceDependency
from budgetroute.inference.batching import AsyncInferenceBatcher
from budgetroute.schemas import FeedbackRecord, GenerationRequest, GenerationResponse, RouteDecision

router = APIRouter()


@router.get("/healthz", summary="Process liveness")
def healthz() -> dict[str, str]:
    """Return liveness without initializing or probing model hardware."""
    return {"status": "ok"}


@router.get("/readyz", summary="Dependency readiness")
def readyz(service: ServiceDependency) -> dict[str, object]:
    state = service.health()
    ready = bool(state["ready"])
    return {"status": "ready" if ready else "not_ready", "ready": ready}


@router.get("/v1/config", summary="Sanitized active configuration")
def active_config(service: ServiceDependency) -> dict[str, object]:
    return service.config.sanitized_summary()


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
    public_execution = response.execution.model_copy(update={"initial_answer": None})
    return response.model_copy(update={"execution": public_execution})


@router.get("/metrics", response_class=PlainTextResponse, summary="Prometheus metrics")
def metrics(request: Request, service: ServiceDependency, monitor: MonitorDependency) -> str:
    if not service.config.api.metrics_enabled:
        raise HTTPException(status_code=404, detail="metrics endpoint is disabled")
    scheduler: AsyncInferenceBatcher | None = request.app.state.inference_batcher
    return monitor.prometheus(scheduler.metrics() if scheduler is not None else None)


@router.get("/v1/monitoring", summary="Sanitized runtime monitoring snapshot")
def monitoring(
    request: Request, service: ServiceDependency, monitor: MonitorDependency
) -> dict[str, object]:
    scheduler: AsyncInferenceBatcher | None = request.app.state.inference_batcher
    return {
        "monitoring": monitor.snapshot(),
        "scheduler": scheduler.metrics() if scheduler is not None else None,
        "backend_load": {
            name.value: snapshot.model_dump() for name, snapshot in service.load_snapshot().items()
        },
    }


@router.post("/v1/feedback", status_code=202, summary="Record aggregate correctness feedback")
def feedback(
    payload: FeedbackRecord, service: ServiceDependency, monitor: MonitorDependency
) -> dict[str, object]:
    if not service.config.api.feedback_enabled:
        raise HTTPException(status_code=404, detail="feedback endpoint is disabled")
    monitor.record_feedback(payload)
    return {"accepted": True, "retained_fields": ["correct"]}
