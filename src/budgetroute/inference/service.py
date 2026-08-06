"""Dependency composition and lifecycle management for inference services."""

from __future__ import annotations

import time
from typing import Any

from budgetroute.backends.base import GenerationBackend
from budgetroute.backends.fake import FakeBackend
from budgetroute.backends.transformers import TransformersBackend
from budgetroute.config import AppConfig, BackendConfig
from budgetroute.inference.engine import InferenceEngine, RouteResult
from budgetroute.retrieval.base import EmbeddingProvider
from budgetroute.retrieval.embeddings import FakeEmbedder, TransformersEmbedder
from budgetroute.retrieval.index import ExactCosineIndex, FaissCosineIndex
from budgetroute.retrieval.service import RetrievalService
from budgetroute.routing.registry import build_policy
from budgetroute.schemas import BackendName, GenerationRequest, GenerationResponse


def _build_backend(name: BackendName, config: BackendConfig) -> GenerationBackend:
    if config.type == "fake":
        return FakeBackend(name, config)
    return TransformersBackend(name, config)


def _build_retriever(config: AppConfig) -> RetrievalService | None:
    if not config.retrieval.enabled:
        return None
    if config.retrieval.embedder == "fake":
        embedder: EmbeddingProvider = FakeEmbedder(config.retrieval.embedding_dimension)
    else:
        assert config.retrieval.embedding_model_id is not None
        device = "cuda" if config.mode == "gpu" else "cpu"
        embedder = TransformersEmbedder(config.retrieval.embedding_model_id, device)
    index_class = FaissCosineIndex if config.retrieval.index_type == "faiss" else ExactCosineIndex
    return RetrievalService(config.retrieval, index_class(embedder))


class InferenceService:
    def __init__(
        self,
        config: AppConfig,
        engine: InferenceEngine,
        retriever: RetrievalService | None,
    ) -> None:
        self.config = config
        self.engine = engine
        self.retriever = retriever
        self._initialized = False
        self.initialization_ms = 0.0

    def initialize(self) -> None:
        if self._initialized:
            return
        start = time.perf_counter_ns()
        self.engine.small_backend.initialize()
        self.engine.large_backend.initialize()
        if self.retriever is not None:
            self.retriever.initialize()
        self.initialization_ms = (time.perf_counter_ns() - start) / 1_000_000
        self._initialized = True

    def route(self, request: GenerationRequest) -> RouteResult:
        if not self._initialized:
            self.initialize()
        return self.engine.route(request)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self._initialized:
            self.initialize()
        return self.engine.generate(request)

    def health(self) -> dict[str, Any]:
        retriever_ready = self.retriever is None or self.retriever.ready
        small = self.engine.small_backend.health()
        large = self.engine.large_backend.health()
        return {
            "initialized": self._initialized,
            "ready": bool(small["ready"] and large["ready"] and retriever_ready),
            "small_backend": small,
            "large_backend": large,
            "retrieval_ready": retriever_ready,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "initialization_ms": self.initialization_ms,
            "small_backend": self.engine.small_backend.metadata(),
            "large_backend": self.engine.large_backend.metadata(),
            "retrieval": self.retriever.metadata() if self.retriever else None,
        }

    def close(self) -> None:
        self.engine.small_backend.cleanup()
        self.engine.large_backend.cleanup()
        self._initialized = False


def build_service(config: AppConfig) -> InferenceService:
    small = _build_backend(BackendName.SMALL, config.small_backend)
    large = _build_backend(BackendName.LARGE, config.large_backend)
    retriever = _build_retriever(config)
    policy = build_policy(config.routing)
    engine = InferenceEngine(small, large, policy, config.routing, retriever)
    return InferenceService(config, engine, retriever)
