"""FastAPI application factory with safe lifecycle and error responses."""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
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
    ContentPolicyError,
    OverloadError,
    RequestDeadlineError,
)
from budgetroute.inference.batching import AsyncInferenceBatcher
from budgetroute.inference.service import InferenceService, build_service
from budgetroute.monitoring.runtime import RuntimeMonitor
from budgetroute.operations.auth import TenantAuthenticator, required_scope
from budgetroute.operations.models import AdmissionLease, Principal
from budgetroute.operations.store import build_store


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
    operational_store = build_store(active_config.operations)
    authenticator = TenantAuthenticator(active_config.api)
    replica_id = os.environ.get(active_config.operations.replica_id_env) or (
        f"{socket.gethostname()}:{os.getpid()}"
    )
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
    rate_limiter = SlidingWindowRateLimiter(
        active_config.api.rate_limit_requests,
        active_config.api.rate_limit_window_seconds,
    )

    async def renew_admission_lease(lease_id: str) -> None:
        interval = max(1.0, min(30.0, active_config.operations.lease_ttl_seconds / 3.0))
        while True:
            await asyncio.sleep(interval)
            renewed = await asyncio.to_thread(
                operational_store.renew_lease,
                lease_id,
                active_config.operations.lease_ttl_seconds,
            )
            if not renewed:
                return

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.inference_service = active_service
        application.state.runtime_monitor = runtime_monitor
        application.state.inference_batcher = inference_batcher
        application.state.operational_store = operational_store
        application.state.authenticator = authenticator
        application.state.replica_id = replica_id
        operational_store.initialize()
        active_service.initialize()
        if inference_batcher is not None:
            await inference_batcher.start()
        try:
            yield
        finally:
            if inference_batcher is not None:
                await inference_batcher.close()
            active_service.close()
            operational_store.close()

    application = FastAPI(
        title="BudgetRoute-LLM API",
        version=__version__,
        description="Quality-aware language-model routing with structured execution traces.",
        lifespan=lifespan,
    )
    application.state.inference_service = active_service
    application.state.runtime_monitor = runtime_monitor
    application.state.inference_batcher = inference_batcher
    application.state.operational_store = operational_store
    application.state.authenticator = authenticator
    application.state.replica_id = replica_id
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=active_config.api.trusted_hosts)
    application.add_middleware(BodyLimitMiddleware, max_bytes=active_config.api.max_request_bytes)
    application.add_middleware(SecurityHeadersMiddleware)

    @application.middleware("http")
    async def admission_control(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        scope = required_scope(request.method, request.url.path)
        protected = scope is not None
        authorization = request.headers.get("authorization", "")
        bearer = authorization[7:] if authorization.lower().startswith("bearer ") else None
        presented = bearer or request.headers.get("x-api-key")
        principal = authenticator.authenticate(presented)
        if protected and principal is None:
            return JSONResponse(
                status_code=401,
                content={"error": {"type": "authentication_required"}},
                headers={"WWW-Authenticate": "Bearer"},
            )
        if principal is None:
            principal = Principal(
                tenant_id="public",
                subject="public-health-check",
                scopes=frozenset(),
                authentication_method="local",
            )
        request.state.principal = principal
        if scope is not None and not principal.permits(scope):
            await asyncio.to_thread(
                operational_store.increment_metric, "authorization_denied_total"
            )
            return JSONResponse(
                status_code=403,
                content={"error": {"type": "insufficient_scope", "required_scope": scope}},
            )
        remaining: int | None = None
        if protected:
            if active_config.operations.backend in {"sqlite", "postgres"}:
                quota = await asyncio.to_thread(
                    operational_store.check_quota,
                    principal.tenant_id,
                    active_config.operations.quota_requests,
                    active_config.operations.quota_window_seconds,
                )
                allowed = quota.allowed
                retry_after = quota.retry_after_seconds
                remaining = quota.remaining
            else:
                client_host = request.client.host if request.client is not None else None
                allowed, retry_after = rate_limiter.allow(opaque_client_identity(client_host))
            if not allowed:
                await asyncio.to_thread(
                    operational_store.increment_metric, "quota_rejections_total"
                )
                return JSONResponse(
                    status_code=429,
                    content={"error": {"type": "rate_limit_exceeded"}},
                    headers={"Retry-After": str(max(1, int(retry_after)))},
                )
        lease: AdmissionLease | None = None
        lease_heartbeat: asyncio.Task[None] | None = None
        if request.url.path == "/v1/generate":
            lease = await asyncio.to_thread(
                operational_store.acquire_lease,
                principal.tenant_id,
                replica_id,
                active_config.operations.max_global_inflight,
                active_config.operations.lease_ttl_seconds,
            )
            if lease is None:
                await asyncio.to_thread(
                    operational_store.increment_metric, "global_admission_rejections_total"
                )
                return JSONResponse(
                    status_code=503,
                    content={"error": {"type": "global_capacity_exhausted"}},
                    headers={"Retry-After": "1"},
                )
            lease_heartbeat = asyncio.create_task(renew_admission_lease(lease.lease_id))
        try:
            response = await call_next(request)
        finally:
            if lease_heartbeat is not None:
                lease_heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await lease_heartbeat
            if lease is not None:
                await asyncio.to_thread(operational_store.release_lease, lease.lease_id)
        response.headers["X-Request-ID"] = str(uuid4())
        if remaining is not None:
            response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

    @application.exception_handler(OverloadError)
    async def overload_handler(request: Request, exc: OverloadError) -> JSONResponse:
        runtime_monitor.record_error()
        await asyncio.to_thread(operational_store.increment_metric, "errors_total")
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

    @application.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        del request
        runtime_monitor.record_error()
        await asyncio.to_thread(operational_store.increment_metric, "validation_errors_total")
        issues = [
            {
                "location": [str(part) for part in item.get("loc", ())],
                "type": str(item.get("type", "validation_error")),
            }
            for item in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={"error": {"type": "request_validation_error", "issues": issues}},
        )

    @application.exception_handler(ContentPolicyError)
    async def content_policy_handler(request: Request, exc: ContentPolicyError) -> JSONResponse:
        runtime_monitor.record_error()
        await asyncio.to_thread(
            operational_store.increment_metric, "content_policy_rejections_total"
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "path": request.url.path,
                }
            },
        )

    @application.exception_handler(RequestDeadlineError)
    async def deadline_handler(request: Request, exc: RequestDeadlineError) -> JSONResponse:
        runtime_monitor.record_error()
        await asyncio.to_thread(operational_store.increment_metric, "errors_total")
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
        await asyncio.to_thread(operational_store.increment_metric, "errors_total")
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
