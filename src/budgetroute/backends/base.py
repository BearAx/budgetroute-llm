"""Generation backend protocol."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from budgetroute.schemas import BackendGeneration, GenerationRequest


@runtime_checkable
class GenerationBackend(Protocol):
    def initialize(self) -> None: ...

    def health(self) -> dict[str, Any]: ...

    def token_count(self, text: str) -> int: ...

    def generate(self, request: GenerationRequest) -> BackendGeneration: ...

    def generate_batch(self, requests: list[GenerationRequest]) -> list[BackendGeneration]: ...

    def metadata(self) -> dict[str, Any]: ...

    def cleanup(self) -> None: ...
