from __future__ import annotations

import asyncio

from budgetroute.backends.fake import FakeBackend
from budgetroute.config import BackendConfig
from budgetroute.inference.batching import AsyncMicrobatcher
from budgetroute.schemas import BackendName, GenerationRequest


def test_microbatcher_preserves_order_and_flushes() -> None:
    async def scenario() -> None:
        backend = FakeBackend(BackendName.SMALL, BackendConfig(type="fake", quality=1.0))
        backend.initialize()
        batcher = AsyncMicrobatcher(backend, max_batch_size=3, max_wait_ms=2)
        requests = [
            GenerationRequest(request_id=str(index), prompt=f"Prompt {index}") for index in range(5)
        ]
        outputs = await asyncio.gather(*(batcher.submit(request) for request in requests))
        await batcher.close()
        assert len(outputs) == 5
        assert [output.text for output in outputs] == ["A deterministic fake response."] * 5

    asyncio.run(scenario())
