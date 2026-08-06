"""Health, readiness, configuration, routing, and generation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from budgetroute.api.dependencies import ServiceDependency
from budgetroute.schemas import GenerationRequest, GenerationResponse, RouteDecision

router = APIRouter()


@router.get("/healthz", summary="Process liveness")
def healthz() -> dict[str, str]:
    """Return liveness without initializing or probing model hardware."""
    return {"status": "ok"}


@router.get("/readyz", summary="Dependency readiness")
def readyz(service: ServiceDependency) -> dict[str, object]:
    state = service.health()
    return {"status": "ready" if state["ready"] else "not_ready", **state}


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
def generate(payload: GenerationRequest, service: ServiceDependency) -> GenerationResponse:
    if len(payload.prompt) > service.config.api.max_prompt_chars:
        raise HTTPException(status_code=413, detail="prompt exceeds the configured character limit")
    response = service.generate(payload)
    public_execution = response.execution.model_copy(update={"initial_answer": None})
    return response.model_copy(update={"execution": public_execution})
