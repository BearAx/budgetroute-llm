"""FastAPI dependency accessors for application-scoped services."""

from typing import Annotated, cast

from fastapi import Depends, Request

from budgetroute.inference.service import InferenceService


def get_service(request: Request) -> InferenceService:
    return cast(InferenceService, request.app.state.inference_service)


ServiceDependency = Annotated[InferenceService, Depends(get_service)]
