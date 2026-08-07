from __future__ import annotations

import pytest

from budgetroute.backends.fake import FakeBackend
from budgetroute.backends.transformers import TransformersBackend
from budgetroute.config import BackendConfig
from budgetroute.exceptions import BackendError
from budgetroute.schemas import BackendName, GenerationRequest


def test_fake_backend_is_deterministic() -> None:
    backend = FakeBackend(BackendName.SMALL, BackendConfig(type="fake", quality=1.0))
    backend.initialize()
    request = GenerationRequest(request_id="stable", prompt="What is the capital of France?")
    first = backend.generate(request)
    second = backend.generate(request)
    assert first.text == second.text == "Paris"
    assert first.input_tokens == second.input_tokens


def test_fake_backend_simulates_failure() -> None:
    backend = FakeBackend(
        BackendName.SMALL,
        BackendConfig(type="fake", fail_on_substrings=["explode"]),
    )
    backend.initialize()
    with pytest.raises(BackendError, match="simulated"):
        backend.generate(GenerationRequest(prompt="Please explode now"))


def test_transformers_backend_is_lazy() -> None:
    backend = TransformersBackend(
        BackendName.SMALL,
        BackendConfig(type="transformers", model_id="not-downloaded", device="cpu"),
    )
    assert backend.health()["initialized"] is False
    assert backend.metadata()["model_id"] == "not-downloaded"


def test_transformers_cleanup_releases_runtime_references() -> None:
    class Cuda:
        def __init__(self) -> None:
            self.empty_cache_calls = 0

        @staticmethod
        def is_available() -> bool:
            return True

        def empty_cache(self) -> None:
            self.empty_cache_calls += 1

    class Torch:
        def __init__(self) -> None:
            self.cuda = Cuda()

    backend = TransformersBackend(
        BackendName.SMALL,
        BackendConfig(type="transformers", model_id="not-downloaded", device="cpu"),
    )
    torch = Torch()
    backend._model = object()
    backend._tokenizer = object()
    backend._torch = torch

    backend.cleanup()

    assert backend._model is None
    assert backend._tokenizer is None
    assert backend._torch is None
    assert torch.cuda.empty_cache_calls == 1
