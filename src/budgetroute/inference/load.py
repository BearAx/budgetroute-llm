"""Thread-safe backend load telemetry and transparent tracking decorator."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

from budgetroute.backends.base import GenerationBackend
from budgetroute.schemas import BackendGeneration, BackendName, GenerationRequest


@dataclass(frozen=True)
class BackendLoadSnapshot:
    backend: BackendName
    inflight_requests: int
    capacity: int
    utilization: float
    ewma_latency_ms: float
    completed_requests: int
    failed_requests: int
    error_rate: float

    def model_dump(self) -> dict[str, Any]:
        result = asdict(self)
        result["backend"] = self.backend.value
        return result


class BackendLoadTracker:
    """Maintain small, bounded operational counters without retaining prompts."""

    def __init__(self, backend: BackendName, capacity: int, *, alpha: float = 0.2) -> None:
        self.backend = backend
        self.capacity = capacity
        self.alpha = alpha
        self._lock = threading.Lock()
        self._inflight = 0
        self._ewma_latency_ms = 0.0
        self._completed = 0
        self._failed = 0

    def begin(self, request_count: int) -> int:
        with self._lock:
            self._inflight += request_count
        return time.perf_counter_ns()

    def finish(self, started_ns: int, request_count: int, *, failed: bool) -> None:
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        with self._lock:
            self._inflight = max(0, self._inflight - request_count)
            self._completed += request_count
            if failed:
                self._failed += request_count
            if self._ewma_latency_ms == 0.0:
                self._ewma_latency_ms = elapsed_ms
            else:
                self._ewma_latency_ms = (
                    self.alpha * elapsed_ms + (1.0 - self.alpha) * self._ewma_latency_ms
                )

    def snapshot(self) -> BackendLoadSnapshot:
        with self._lock:
            completed = self._completed
            failed = self._failed
            inflight = self._inflight
            latency = self._ewma_latency_ms
        return BackendLoadSnapshot(
            backend=self.backend,
            inflight_requests=inflight,
            capacity=self.capacity,
            utilization=inflight / self.capacity,
            ewma_latency_ms=latency,
            completed_requests=completed,
            failed_requests=failed,
            error_rate=failed / completed if completed else 0.0,
        )


class TrackedBackend:
    """Add load observations while preserving the generation backend protocol."""

    def __init__(self, delegate: GenerationBackend, tracker: BackendLoadTracker) -> None:
        self.delegate = delegate
        self.tracker = tracker
        self._slots = threading.BoundedSemaphore(tracker.capacity)

    def initialize(self) -> None:
        self.delegate.initialize()

    def health(self) -> dict[str, Any]:
        return {**self.delegate.health(), "load": self.tracker.snapshot().model_dump()}

    def token_count(self, text: str) -> int:
        return self.delegate.token_count(text)

    def generate(self, request: GenerationRequest) -> BackendGeneration:
        with self._slots:
            started = self.tracker.begin(1)
            failed = True
            try:
                result = self.delegate.generate(request)
                failed = False
                return result
            finally:
                self.tracker.finish(started, 1, failed=failed)

    def generate_batch(self, requests: list[GenerationRequest]) -> list[BackendGeneration]:
        if not requests:
            return []
        with self._slots:
            started = self.tracker.begin(len(requests))
            failed = True
            try:
                result = self.delegate.generate_batch(requests)
                failed = False
                return result
            finally:
                self.tracker.finish(started, len(requests), failed=failed)

    def metadata(self) -> dict[str, Any]:
        return {**self.delegate.metadata(), "load": self.tracker.snapshot().model_dump()}

    def cleanup(self) -> None:
        self.delegate.cleanup()
