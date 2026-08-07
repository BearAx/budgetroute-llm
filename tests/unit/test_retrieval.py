from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from budgetroute.exceptions import RetrievalError
from budgetroute.retrieval.base import Chunk
from budgetroute.retrieval.embeddings import FakeEmbedder
from budgetroute.retrieval.index import ExactCosineIndex, FaissCosineIndex, FaissHNSWIndex


def test_exact_index_ranks_matching_document(tmp_path: Path) -> None:
    index = ExactCosineIndex(FakeEmbedder(128))
    index.build(
        [
            Chunk(document_id="a", chunk_id="a:0", text="orchid release routing"),
            Chunk(document_id="b", chunk_id="b:0", text="bananas and tropical fruit"),
        ]
    )
    assert index.search("Which orchid routing release?", 1)[0].document_id == "a"
    saved = tmp_path / "index.npz"
    index.save(saved)
    restored = ExactCosineIndex(FakeEmbedder(128))
    restored.load(saved)
    assert restored.search("orchid", 1)[0].chunk_id == "a:0"


def test_faiss_index_has_actionable_optional_dependency_error() -> None:
    if importlib.util.find_spec("faiss") is not None:
        pytest.skip("FAISS is installed in this optional test environment")
    index = FaissCosineIndex(FakeEmbedder(64))
    with pytest.raises(RetrievalError, match="FAISS index requested"):
        index.build([Chunk(document_id="a", chunk_id="a:0", text="text")])


def test_hnsw_index_applies_explicit_construction_and_search_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HnswControls:
        efConstruction = 0
        efSearch = 0

    class FakeHnsw:
        def __init__(self, dimension: int, neighbors: int, metric: int) -> None:
            del dimension, neighbors, metric
            self.hnsw = HnswControls()
            self.matrix: np.ndarray | None = None

        def add(self, matrix: np.ndarray) -> None:
            self.matrix = matrix

        def search(self, query: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
            assert self.matrix is not None
            scores = query @ self.matrix.T
            order = np.argsort(-scores, axis=1)[:, :count]
            return np.take_along_axis(scores, order, axis=1), order

    class FakeFaiss:
        METRIC_INNER_PRODUCT = 0
        IndexHNSWFlat = FakeHnsw

    index = FaissHNSWIndex(FakeEmbedder(16), neighbors=8, ef_construction=40, ef_search=20)
    monkeypatch.setattr(index, "_require_faiss", lambda: FakeFaiss())
    index.build(
        [
            Chunk(document_id="a", chunk_id="a-1", text="alpha lantern"),
            Chunk(document_id="b", chunk_id="b-1", text="beta ocean"),
        ]
    )
    assert index._faiss_index.hnsw.efConstruction == 40
    assert index._faiss_index.hnsw.efSearch == 20
    assert index.search("lantern", 1)[0].document_id == "a"
