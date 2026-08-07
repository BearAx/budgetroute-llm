"""Reliable asyncio microbatching with bounded waits and orderly shutdown."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from budgetroute.backends.base import GenerationBackend
from budgetroute.exceptions import OverloadError, RequestDeadlineError
from budgetroute.schemas import BackendGeneration, GenerationRequest, GenerationResponse

if TYPE_CHECKING:
    from budgetroute.inference.service import InferenceService


@dataclass
class _BatchItem:
    request: GenerationRequest
    future: asyncio.Future[BackendGeneration]


class AsyncMicrobatcher:
    def __init__(
        self, backend: GenerationBackend, max_batch_size: int = 8, max_wait_ms: float = 10.0
    ) -> None:
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_ms / 1000.0
        self._queue: asyncio.Queue[_BatchItem | None] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._closed = False

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="budgetroute-microbatcher")

    async def submit(self, request: GenerationRequest) -> BackendGeneration:
        if self._closed:
            raise RuntimeError("microbatcher is closed")
        await self.start()
        future: asyncio.Future[BackendGeneration] = asyncio.get_running_loop().create_future()
        await self._queue.put(_BatchItem(request, future))
        return await future

    async def _next_with_timeout(self) -> tuple[bool, _BatchItem | None]:
        try:
            return True, await asyncio.wait_for(self._queue.get(), timeout=self.max_wait_seconds)
        except TimeoutError:
            return False, None

    async def _run(self) -> None:
        stop_after_batch = False
        while True:
            first = await self._queue.get()
            if first is None:
                return
            batch = [first]
            while len(batch) < self.max_batch_size:
                received, item = await self._next_with_timeout()
                if not received:
                    break
                if item is None:
                    stop_after_batch = True
                    break
                batch.append(item)
            try:
                outputs = await asyncio.to_thread(
                    self.backend.generate_batch, [item.request for item in batch]
                )
                if len(outputs) != len(batch):
                    raise RuntimeError("backend returned a different number of batch outputs")
                for item, output in zip(batch, outputs, strict=True):
                    if not item.future.done():
                        item.future.set_result(output)
            except Exception as exc:
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)
            if stop_after_batch:
                return

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._worker is not None:
            await self._queue.put(None)
            await self._worker
            self._worker = None


@dataclass
class _InferenceBatchItem:
    request: GenerationRequest
    future: asyncio.Future[GenerationResponse]
    enqueued_ns: int
    deadline_ns: int


class AsyncInferenceBatcher:
    """Bounded full-pipeline batching with deadlines and observable queue behavior."""

    def __init__(
        self,
        service: InferenceService,
        *,
        max_batch_size: int,
        max_wait_ms: float,
        max_queue_size: int,
        submit_timeout_ms: float,
        request_timeout_ms: float,
    ) -> None:
        self.service = service
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_ms / 1000.0
        self.submit_timeout_seconds = submit_timeout_ms / 1000.0
        self.request_timeout_seconds = request_timeout_ms / 1000.0
        self._queue: asyncio.Queue[_InferenceBatchItem | None] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self._stats: dict[str, float | int] = {
            "submitted": 0,
            "completed": 0,
            "rejected": 0,
            "timed_out": 0,
            "batches": 0,
            "max_observed_queue_depth": 0,
            "total_queue_ms": 0.0,
            "last_batch_size": 0,
        }

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="budgetroute-inference-batcher")

    async def submit(self, request: GenerationRequest) -> GenerationResponse:
        if self._closed:
            raise OverloadError("inference scheduler is closed")
        await self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[GenerationResponse] = loop.create_future()
        now = time.perf_counter_ns()
        item = _InferenceBatchItem(
            request=request,
            future=future,
            enqueued_ns=now,
            deadline_ns=now + int(self.request_timeout_seconds * 1_000_000_000),
        )
        self._stats["submitted"] += 1
        try:
            await asyncio.wait_for(
                self._queue.put(item),
                timeout=min(self.submit_timeout_seconds, self.request_timeout_seconds),
            )
        except TimeoutError as exc:
            self._stats["rejected"] += 1
            future.cancel()
            raise OverloadError("inference queue is full; retry after backoff") from exc
        self._stats["max_observed_queue_depth"] = max(
            int(self._stats["max_observed_queue_depth"]), self._queue.qsize()
        )
        try:
            remaining_seconds = max(0.0, (item.deadline_ns - time.perf_counter_ns()) / 1e9)
            return await asyncio.wait_for(future, timeout=remaining_seconds)
        except TimeoutError as exc:
            self._stats["timed_out"] += 1
            future.cancel()
            raise RequestDeadlineError("inference request exceeded its deadline") from exc

    async def _next_until(self, deadline: float) -> tuple[bool, _InferenceBatchItem | None]:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return False, None
        try:
            return True, await asyncio.wait_for(self._queue.get(), timeout=remaining)
        except TimeoutError:
            return False, None

    async def _run(self) -> None:
        stop_after_batch = False
        while True:
            first = await self._queue.get()
            if first is None:
                return
            batch = [first]
            flush_deadline = asyncio.get_running_loop().time() + self.max_wait_seconds
            while len(batch) < self.max_batch_size:
                received, item = await self._next_until(flush_deadline)
                if not received:
                    break
                if item is None:
                    stop_after_batch = True
                    break
                batch.append(item)
            now_ns = time.perf_counter_ns()
            active: list[_InferenceBatchItem] = []
            for item in batch:
                if now_ns >= item.deadline_ns:
                    self._stats["timed_out"] += 1
                    if not item.future.done():
                        item.future.set_exception(
                            RequestDeadlineError("inference request expired while queued")
                        )
                elif not item.future.cancelled():
                    active.append(item)
            if active:
                self._stats["batches"] += 1
                self._stats["last_batch_size"] = len(active)
                dispatch_ns = time.perf_counter_ns()
                try:
                    outputs = await asyncio.to_thread(
                        self.service.generate_batch, [item.request for item in active]
                    )
                    if len(outputs) != len(active):
                        raise RuntimeError("service returned a different number of batch outputs")
                    for item, output in zip(active, outputs, strict=True):
                        queue_ms = (dispatch_ns - item.enqueued_ns) / 1_000_000
                        self._stats["completed"] += 1
                        self._stats["total_queue_ms"] += queue_ms
                        timing = output.timing.model_copy(
                            update={
                                "queue_ms": queue_ms,
                                "batch_size": len(active),
                                "total_ms": output.timing.total_ms + queue_ms,
                            }
                        )
                        if not item.future.done():
                            item.future.set_result(output.model_copy(update={"timing": timing}))
                except Exception as exc:
                    for item in active:
                        if not item.future.done():
                            item.future.set_exception(exc)
            if stop_after_batch:
                return

    def metrics(self) -> dict[str, Any]:
        completed = int(self._stats["completed"])
        return {
            **self._stats,
            "queue_depth": self._queue.qsize(),
            "average_queue_ms": (
                float(self._stats["total_queue_ms"]) / completed if completed else 0.0
            ),
            "closed": self._closed,
        }

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._worker is not None:
            await self._queue.put(None)
            await self._worker
            self._worker = None
