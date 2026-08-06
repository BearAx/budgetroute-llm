from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from budgetroute.exceptions import RetrievalError
from budgetroute.retrieval.base import Chunk
from budgetroute.retrieval.embeddings import FakeEmbedder
from budgetroute.retrieval.index import ExactCosineIndex, FaissCosineIndex


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
