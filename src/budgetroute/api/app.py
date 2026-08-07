"""FastAPI application factory with safe lifecycle and error responses."""

from __future__ import annotations

import os
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware

from budgetroute import __version__
from budgetroute.api.routes import router
from budgetroute.api.security import (
    BodyLimitMiddleware,
    SecurityHeadersMiddleware,
    SlidingWindowRateLimiter,
    opaque_client_identity,
)
from budgetroute.config import AppConfig, load_config, validate_runtime_config
from budgetroute.exceptions import (
    BudgetRouteError,
    OverloadError,
    RequestDeadlineError,
)
from budgetroute.inference.batching import AsyncInferenceBatcher
from budgetroute.inference.service import InferenceService, build_service
from budgetroute.monitoring.runtime import RuntimeMonitor


def create_app(
    config: AppConfig | None = None,
    *,
    config_path: Path | None = None,
    service: InferenceService | None = None,
) -> FastAPI:
    active_config = config or load_config(config_path or Path("configs/serving/fake.yaml"))
    validate_runtime_config(active_config)
    active_service = service or build_service(active_config)
    runtime_monitor = RuntimeMonitor(active_config.monitoring)
    inference_batcher = (
        AsyncInferenceBatcher(
            active_service,
            max_batch_size=active_config.batching.max_batch_size,
            max_wait_ms=active_config.batching.max_wait_ms,
            max_queue_size=active_config.batching.max_queue_size,
            submit_timeout_ms=active_config.batching.submit_timeout_ms,
            request_timeout_ms=active_config.batching.request_timeout_ms,
        )
        if active_config.batching.enabled
        else None
    )
    api_secret = (
        os.environ.get(active_config.api.api_key_env) if active_config.api.require_api_key else None
    )
    rate_limiter = SlidingWindowRateLimiter(
        active_config.api.rate_limit_requests,
        active_config.api.rate_limit_window_seconds,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.inference_service = active_service
        application.state.runtime_monitor = runtime_monitor
        application.state.inference_batcher = inference_batcher
        active_service.initialize()
        if inference_batcher is not None:
            await inference_batcher.start()
        try:
            yield
        finally:
            if inference_batcher is not None:
                await inference_batcher.close()
            active_service.close()

    application = FastAPI(
        title="BudgetRoute-LLM API",
        version=__version__,
        description="Quality-aware language-model routing with structured execution traces.",
        lifespan=lifespan,
    )
    application.state.inference_service = active_service
    application.state.runtime_monitor = runtime_monitor
    application.state.inference_batcher = inference_batcher
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=active_config.api.trusted_hosts)
    application.add_middleware(BodyLimitMiddleware, max_bytes=active_config.api.max_request_bytes)
    application.add_middleware(SecurityHeadersMiddleware)

    @application.middleware("http")
    async def admission_control(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        protected = request.url.path.startswith("/v1/") or request.url.path == "/metrics"
        authorization = request.headers.get("authorization", "")
        bearer = authorization[7:] if authorization.lower().startswith("bearer ") else None
        presented = bearer or request.headers.get("x-api-key")
        if (
            protected
            and active_config.api.require_api_key
            and (api_secret is None or not secrets.compare_digest(presented or "", api_secret))
        ):
            return JSONResponse(
                status_code=401,
                content={"error": {"type": "authentication_required"}},
                headers={"WWW-Authenticate": "Bearer"},
            )
        if protected:
            client_host = request.client.host if request.client is not None else None
            allowed, retry_after = rate_limiter.allow(opaque_client_identity(client_host))
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"error": {"type": "rate_limit_exceeded"}},
                    headers={"Retry-After": str(max(1, int(retry_after)))},
                )
        response = await call_next(request)
        response.headers["X-Request-ID"] = str(uuid4())
        return response

    @application.exception_handler(OverloadError)
    async def overload_handler(request: Request, exc: OverloadError) -> JSONResponse:
        runtime_monitor.record_error()
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "path": request.url.path,
                }
            },
            headers={"Retry-After": "1"},
        )

    @application.exception_handler(RequestDeadlineError)
    async def deadline_handler(request: Request, exc: RequestDeadlineError) -> JSONResponse:
        runtime_monitor.record_error()
        return JSONResponse(
            status_code=504,
            content={
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "path": request.url.path,
                }
            },
        )

    @application.exception_handler(BudgetRouteError)
    async def domain_error_handler(request: Request, exc: BudgetRouteError) -> JSONResponse:
        runtime_monitor.record_error()
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
