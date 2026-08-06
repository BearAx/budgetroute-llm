"""FastAPI application factory with safe lifecycle and error responses."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from budgetroute import __version__
from budgetroute.api.routes import router
from budgetroute.config import AppConfig, load_config
from budgetroute.exceptions import BudgetRouteError
from budgetroute.inference.service import InferenceService, build_service


def create_app(
    config: AppConfig | None = None,
    *,
    config_path: Path | None = None,
    service: InferenceService | None = None,
) -> FastAPI:
    active_config = config or load_config(config_path or Path("configs/serving/fake.yaml"))
    active_service = service or build_service(active_config)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.inference_service = active_service
        active_service.initialize()
        try:
            yield
        finally:
            active_service.close()

    application = FastAPI(
        title="BudgetRoute-LLM API",
        version=__version__,
        description="Quality-aware language-model routing with structured execution traces.",
        lifespan=lifespan,
    )
    application.state.inference_service = active_service

    @application.exception_handler(BudgetRouteError)
    async def domain_error_handler(request: Request, exc: BudgetRouteError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "path": request.url.path,
                }
            },
        )

    application.include_router(router)
    return application
