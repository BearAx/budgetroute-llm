"""Multi-endpoint HTTP load generation and honest operational summaries."""

from __future__ import annotations

import asyncio
import ipaddress
import math
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from budgetroute.experiments.artifacts import ArtifactWriter, create_run_directory
from budgetroute.profiling.latency import summarize_latencies


def _safe_target(value: str, *, allow_insecure_http: bool) -> tuple[str, str]:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("load-test targets must be HTTP(S) URLs")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("load-test targets must not contain credentials, queries, or fragments")
    try:
        loopback = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        loopback = parsed.hostname.lower() == "localhost"
    if parsed.scheme != "https" and not loopback and not allow_insecure_http:
        raise ValueError("non-loopback load-test targets require HTTPS")
    port = f":{parsed.port}" if parsed.port is not None else ""
    safe = f"{parsed.scheme}://{parsed.hostname}{port}"
    return value.rstrip("/"), safe


async def run_load_test(
    targets: list[str],
    *,
    request_count: int,
    concurrency: int,
    output_root: Path,
    api_key_env: str | None = None,
    target_latency_ms: float = 1000.0,
    request_timeout_seconds: float = 120.0,
    warmup_requests: int = 0,
    allow_insecure_http: bool = False,
    transport: Any | None = None,
) -> Path:
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - exercised by minimal installations
        raise RuntimeError("load testing requires the 'api' optional dependency") from exc
    if request_count < 1 or concurrency < 1 or concurrency > request_count:
        raise ValueError("request_count and concurrency must be positive; concurrency <= requests")
    validated = [
        _safe_target(target, allow_insecure_http=allow_insecure_http) for target in targets
    ]
    if not validated:
        raise ValueError("at least one load-test target is required")
    token = os.environ.get(api_key_env) if api_key_env else None
    if api_key_env and not token:
        raise ValueError(f"load-test credential environment variable is unset: {api_key_env}")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    semaphore = asyncio.Semaphore(concurrency)
    samples: list[dict[str, Any]] = []

    async with httpx.AsyncClient(
        headers=headers,
        timeout=request_timeout_seconds,
        transport=transport,
    ) as client:

        async def invoke(index: int, *, measured: bool) -> None:
            target_index = index % len(validated)
            target, _ = validated[target_index]
            sample: dict[str, Any] = {"request_index": index, "target_index": target_index}
            async with semaphore:
                started = time.perf_counter_ns()
                try:
                    response = await client.post(
                        f"{target}/v1/generate",
                        json={
                            "request_id": f"load-{index:08d}",
                            "prompt": f"Load validation request {index}: return ready.",
                            "metadata": {
                                "fake_reference_answer": "ready",
                                "fake_small_success": True,
                            },
                        },
                    )
                    sample["status_code"] = response.status_code
                    if response.headers.get("content-type", "").startswith("application/json"):
                        payload = response.json()
                        if isinstance(payload, dict):
                            sample["route"] = payload.get("route")
                            error = payload.get("error")
                            if isinstance(error, dict):
                                sample["error_type"] = error.get("type")
                except Exception as exc:
                    sample.update({"status_code": 0, "error_type": type(exc).__name__})
                sample["latency_ms"] = (time.perf_counter_ns() - started) / 1_000_000
            if measured:
                samples.append(sample)

        if warmup_requests:
            await asyncio.gather(
                *(invoke(-index - 1, measured=False) for index in range(warmup_requests))
            )
        wall_started = time.perf_counter_ns()
        await asyncio.gather(*(invoke(index, measured=True) for index in range(request_count)))
        wall_ms = (time.perf_counter_ns() - wall_started) / 1_000_000

    samples.sort(key=lambda item: int(item["request_index"]))
    latencies = [float(item["latency_ms"]) for item in samples]
    latency = summarize_latencies(latencies)
    success_count = sum(200 <= int(item["status_code"]) < 300 for item in samples)
    overload_count = sum(int(item["status_code"]) in {429, 503} for item in samples)
    error_rate = 1.0 - success_count / len(samples)
    overload_rate = overload_count / len(samples)
    non_overload_error_rate = max(0.0, error_rate - overload_rate)
    p95 = latency.p95_ms or latency.mean_ms or target_latency_ms
    pressure = max(1.0, p95 / target_latency_ms, 1.0 + overload_rate * 4.0)
    multiplier = math.ceil(pressure * 10.0) / 10.0
    capacity_advisory_applicable = success_count > 0 and non_overload_error_rate <= 1e-12
    run_id, run_directory = create_run_directory(output_root, "load-test")
    writer = ArtifactWriter(run_directory)
    writer.write_jsonl("samples.jsonl", samples)
    writer.write_json(
        "summary.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "targets": [safe for _, safe in validated],
            "request_count": request_count,
            "concurrency": concurrency,
            "load_model": "fixed_request_closed_loop",
            "warmup_requests": warmup_requests,
            "measurement_wall_ms": wall_ms,
            "throughput_requests_per_second": request_count / (wall_ms / 1000.0),
            "success_count": success_count,
            "error_rate": error_rate,
            "overload_rate": overload_rate,
            "non_overload_error_rate": non_overload_error_rate,
            "latency": {
                "mean_ms": latency.mean_ms,
                "p50_ms": latency.p50_ms,
                "p95_ms": latency.p95_ms,
                "p99_ms": latency.p99_ms,
                "warning": latency.warning,
            },
            "autoscaling_advisory": {
                "target_latency_ms": target_latency_ms,
                "applicable": capacity_advisory_applicable,
                "replica_multiplier": multiplier if capacity_advisory_applicable else None,
                "method": "max(p95/target, 1 + 4*overload_rate)",
                "warning": (
                    "Heuristic capacity guidance only; non-overload failures or zero successful "
                    "requests suppress the suggestion. Validate any suggested replica count with "
                    "another steady-state load test before deployment."
                ),
            },
            "credential_source": api_key_env,
            "claim_boundary": (
                "Results apply only to the invoked endpoints, fixed-request closed-loop request "
                "mix, concurrency, and measurement window."
            ),
        },
    )
    return run_directory
