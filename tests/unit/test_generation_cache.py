from __future__ import annotations

from pathlib import Path

import pytest

from budgetroute.backends.cached import CachedGenerationBackend
from budgetroute.backends.fake import FakeBackend
from budgetroute.config import BackendConfig
from budgetroute.exceptions import ArtifactError
from budgetroute.experiments.cache import GenerationCache
from budgetroute.schemas import BackendName, GenerationRequest


def test_cache_key_ignores_request_identity_and_replays_without_delegate_init(
    tmp_path: Path,
) -> None:
    config = BackendConfig(type="fake", quality=1.0)
    cache = GenerationCache(tmp_path / "cache")
    live_delegate = FakeBackend(BackendName.SMALL, config)
    live = CachedGenerationBackend(BackendName.SMALL, config, live_delegate, cache, "read_write")
    live.initialize()
    first = live.generate(
        GenerationRequest(request_id="live", prompt="What is the capital of France?")
    )
    assert first.text == "Paris"
    assert first.metadata["cache_hit"] is False

    untouched_delegate = FakeBackend(BackendName.SMALL, config)
    replay = CachedGenerationBackend(
        BackendName.SMALL,
        config,
        untouched_delegate,
        GenerationCache(tmp_path / "cache"),
        "read_only",
    )
    replay.initialize()
    second = replay.generate(
        GenerationRequest(request_id="replay", prompt="What is the capital of France?")
    )
    assert second.text == first.text
    assert second.metadata["cache_hit"] is True
    assert untouched_delegate.health()["initialized"] is False


def test_read_only_cache_miss_is_actionable(tmp_path: Path) -> None:
    config = BackendConfig(type="fake")
    backend = CachedGenerationBackend(
        BackendName.LARGE,
        config,
        FakeBackend(BackendName.LARGE, config),
        GenerationCache(tmp_path / "empty"),
        "read_only",
    )
    backend.initialize()
    with pytest.raises(ArtifactError, match="collect baselines before replay"):
        backend.generate(GenerationRequest(prompt="uncached"))


def test_cache_batches_only_misses_and_preserves_order(tmp_path: Path) -> None:
    config = BackendConfig(type="fake", quality=1.0)
    cache_path = tmp_path / "batch-cache"
    live = CachedGenerationBackend(
        BackendName.SMALL,
        config,
        FakeBackend(BackendName.SMALL, config),
        GenerationCache(cache_path),
        "read_write",
    )
    live.initialize()
    requests = [
        GenerationRequest(prompt="What is the capital of France?"),
        GenerationRequest(prompt="What is 2 + 2?"),
    ]
    generated = live.generate_batch(requests)
    assert [item.text for item in generated] == ["Paris", "4"]
    assert all(item.metadata["cache_hit"] is False for item in generated)

    replay = CachedGenerationBackend(
        BackendName.SMALL,
        config,
        FakeBackend(BackendName.SMALL, config),
        GenerationCache(cache_path),
        "read_only",
    )
    replay.initialize()
    cached = replay.generate_batch(requests)
    assert [item.text for item in cached] == ["Paris", "4"]
    assert all(item.metadata["cache_hit"] is True for item in cached)
