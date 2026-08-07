from __future__ import annotations

import asyncio

import pytest

from budgetroute.backends.fake import FakeBackend
from budgetroute.config import AppConfig, BackendConfig
from budgetroute.exceptions import OverloadError, RequestDeadlineError
from budgetroute.inference.batching import AsyncInferenceBatcher, AsyncMicrobatcher
from budgetroute.inference.service import build_service
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


def test_full_inference_batcher_enforces_deadline_and_closed_state(
    fake_config: AppConfig,
) -> None:
    async def scenario() -> None:
        slow = fake_config.model_copy(
            update={
                "small_backend": fake_config.small_backend.model_copy(
                    update={"artificial_latency_ms": 20}
                )
            }
        )
        service = build_service(slow)
        service.initialize()
        batcher = AsyncInferenceBatcher(
            service,
            max_batch_size=1,
            max_wait_ms=0,
            max_queue_size=1,
            submit_timeout_ms=1,
            request_timeout_ms=1,
        )
        with pytest.raises(RequestDeadlineError, match="deadline"):
            await batcher.submit(GenerationRequest(prompt="What is the capital of France?"))
        await batcher.close()
        with pytest.raises(OverloadError, match="closed"):
            await batcher.submit(GenerationRequest(prompt="another"))
        assert batcher.metrics()["timed_out"] >= 1
        service.close()

    asyncio.run(scenario())


def test_full_inference_batcher_is_bounded_observable_and_ordered(
    fake_config: AppConfig,
) -> None:
    async def scenario() -> None:
        service = build_service(fake_config)
        service.initialize()
        batcher = AsyncInferenceBatcher(
            service,
            max_batch_size=3,
            max_wait_ms=5,
            max_queue_size=8,
            submit_timeout_ms=20,
            request_timeout_ms=1000,
        )
        prompts = [
            "What is the capital of France?",
            "What is 2 + 2?",
            "Which planet is the red planet?",
            "What is the capital of France?",
            "What is 2 + 2?",
        ]
        try:
            outputs = await asyncio.gather(
                *(
                    batcher.submit(GenerationRequest(request_id=str(index), prompt=prompt))
                    for index, prompt in enumerate(prompts)
                )
            )
            assert [output.request_id for output in outputs] == [str(index) for index in range(5)]
            assert all(output.timing.batch_size <= 3 for output in outputs)
            metrics = batcher.metrics()
            assert metrics["completed"] == 5
            assert metrics["batches"] == 2
            assert metrics["max_observed_queue_depth"] <= 8
        finally:
            await batcher.close()
            service.close()

    asyncio.run(scenario())
