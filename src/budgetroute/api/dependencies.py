"""FastAPI dependency accessors for application-scoped services."""

from typing import Annotated, cast

from fastapi import Depends, Request

from budgetroute.inference.service import InferenceService
from budgetroute.monitoring.runtime import RuntimeMonitor
from budgetroute.operations.models import Principal
from budgetroute.operations.store import OperationalStore


def get_service(request: Request) -> InferenceService:
    return cast(InferenceService, request.app.state.inference_service)


ServiceDependency = Annotated[InferenceService, Depends(get_service)]


def get_monitor(request: Request) -> RuntimeMonitor:
    return cast(RuntimeMonitor, request.app.state.runtime_monitor)


MonitorDependency = Annotated[RuntimeMonitor, Depends(get_monitor)]


def get_store(request: Request) -> OperationalStore:
    return cast(OperationalStore, request.app.state.operational_store)


StoreDependency = Annotated[OperationalStore, Depends(get_store)]


def get_principal(request: Request) -> Principal:
    return cast(Principal, request.state.principal)


PrincipalDependency = Annotated[Principal, Depends(get_principal)]
