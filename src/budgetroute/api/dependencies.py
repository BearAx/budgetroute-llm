"""FastAPI dependency accessors for application-scoped services."""

from typing import Annotated, cast

from fastapi import Depends, Request

from budgetroute.inference.service import InferenceService
from budgetroute.monitoring.runtime import RuntimeMonitor


def get_service(request: Request) -> InferenceService:
    return cast(InferenceService, request.app.state.inference_service)


ServiceDependency = Annotated[InferenceService, Depends(get_service)]


def get_monitor(request: Request) -> RuntimeMonitor:
    return cast(RuntimeMonitor, request.app.state.runtime_monitor)


MonitorDependency = Annotated[RuntimeMonitor, Depends(get_monitor)]
