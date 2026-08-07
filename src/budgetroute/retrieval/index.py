"""Portable exact cosine-similarity retrieval index."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

from budgetroute.exceptions import RetrievalError
from budgetroute.retrieval.base import Chunk, EmbeddingProvider
from budgetroute.schemas import RetrievalHit


class ExactCosineIndex:
    def __init__(self, embedder: EmbeddingProvider) -> None:
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.embeddings = np.empty((0, embedder.dimension), dtype=np.float32)

    @property
    def ready(self) -> bool:
        return bool(self.chunks) and len(self.chunks) == self.embeddings.shape[0]

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise RetrievalError("cannot build an index from an empty chunk list")
        embeddings = self.embedder.embed([chunk.text for chunk in chunks])
        if embeddings.shape != (len(chunks), self.embedder.dimension):
            raise RetrievalError("embedder returned an unexpected matrix shape")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = embeddings / np.clip(norms, 1e-12, None)
        self.chunks = list(chunks)

    def search(self, query: str, top_k: int = 3) -> list[RetrievalHit]:
        if not self.ready:
            raise RetrievalError("retrieval index is not built")
        query_vector = self.embedder.embed([query])[0]
        norm = float(np.linalg.norm(query_vector))
        if not norm:
            return []
        scores = self.embeddings @ (query_vector / norm)
        count = min(top_k, len(self.chunks))
        order = np.argsort(-scores, kind="stable")[:count]
        return [
            RetrievalHit(
                document_id=self.chunks[int(index)].document_id,
                chunk_id=self.chunks[int(index)].chunk_id,
                text=self.chunks[int(index)].text,
                score=float(scores[int(index)]),
                metadata=self.chunks[int(index)].metadata,
            )
            for index in order
        ]

    def save(self, path: Path) -> None:
        if not self.ready:
            raise RetrievalError("cannot save an index before it is built")
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps([chunk.model_dump(mode="json") for chunk in self.chunks])
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, embeddings=self.embeddings, chunks=np.asarray(serialized))
        temporary.replace(path)

    def load(self, path: Path) -> None:
        if not path.is_file():
            raise RetrievalError(f"retrieval index does not exist: {path}")
        try:
            with np.load(path, allow_pickle=False) as data:
                embeddings = np.asarray(data["embeddings"], dtype=np.float32)
                chunks = [Chunk.model_validate(item) for item in json.loads(str(data["chunks"]))]
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise RetrievalError(f"could not load retrieval index {path}: {exc}") from exc
        if embeddings.shape != (len(chunks), self.embedder.dimension):
            raise RetrievalError("saved index dimension does not match the configured embedder")
        self.embeddings = embeddings
        self.chunks = chunks


class FaissCosineIndex(ExactCosineIndex):
    """Optional exact inner-product FAISS index with portable NumPy persistence."""

    def __init__(self, embedder: EmbeddingProvider) -> None:
        super().__init__(embedder)
        self._faiss_index: Any = None

    def _require_faiss(self) -> Any:
        if importlib.util.find_spec("faiss") is None:
            raise RetrievalError(
                "FAISS index requested but faiss is unavailable; install the retrieval extra"
            )
        import faiss

        return faiss

    def _build_faiss(self) -> None:
        faiss = self._require_faiss()
        self._faiss_index = faiss.IndexFlatIP(self.embedder.dimension)
        self._faiss_index.add(np.ascontiguousarray(self.embeddings, dtype=np.float32))

    def build(self, chunks: list[Chunk]) -> None:
        super().build(chunks)
        self._build_faiss()

    def load(self, path: Path) -> None:
        super().load(path)
        self._build_faiss()

    def search(self, query: str, top_k: int = 3) -> list[RetrievalHit]:
        if not self.ready or self._faiss_index is None:
            raise RetrievalError("FAISS retrieval index is not built")
        query_vector = self.embedder.embed([query]).astype(np.float32)
        norm = float(np.linalg.norm(query_vector[0]))
        if not norm:
            return []
        query_vector /= norm
        count = min(top_k, len(self.chunks))
        scores, indices = self._faiss_index.search(query_vector, count)
        return [
            RetrievalHit(
                document_id=self.chunks[int(index)].document_id,
                chunk_id=self.chunks[int(index)].chunk_id,
                text=self.chunks[int(index)].text,
                score=float(score),
                metadata=self.chunks[int(index)].metadata,
            )
            for score, index in zip(scores[0], indices[0], strict=True)
            if index >= 0
        ]


class FaissHNSWIndex(FaissCosineIndex):
    """Approximate cosine retrieval using FAISS HNSW with explicit search controls."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        *,
        neighbors: int = 32,
        ef_construction: int = 80,
        ef_search: int = 64,
    ) -> None:
        super().__init__(embedder)
        self.neighbors = neighbors
        self.ef_construction = ef_construction
        self.ef_search = ef_search

    def _build_faiss(self) -> None:
        faiss = self._require_faiss()
        self._faiss_index = faiss.IndexHNSWFlat(
            self.embedder.dimension, self.neighbors, faiss.METRIC_INNER_PRODUCT
        )
        self._faiss_index.hnsw.efConstruction = self.ef_construction
        self._faiss_index.hnsw.efSearch = self.ef_search
        self._faiss_index.add(np.ascontiguousarray(self.embeddings, dtype=np.float32))
