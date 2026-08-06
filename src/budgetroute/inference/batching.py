"""Reliable asyncio microbatching with bounded waits and orderly shutdown."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from budgetroute.backends.base import GenerationBackend
from budgetroute.schemas import BackendGeneration, GenerationRequest


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
