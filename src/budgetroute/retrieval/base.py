"""Retrieval domain types and protocols."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, Field

from budgetroute.schemas import RetrievalHit


class Document(BaseModel):
    document_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    document_id: str
    chunk_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed(self, texts: list[str]) -> np.ndarray: ...


class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> list[RetrievalHit]: ...
