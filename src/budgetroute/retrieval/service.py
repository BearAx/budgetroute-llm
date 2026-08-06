"""Corpus loading, deterministic chunking, indexing, and timed retrieval."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from budgetroute.config import RetrievalConfig
from budgetroute.exceptions import RetrievalError
from budgetroute.retrieval.base import Chunk, Document
from budgetroute.retrieval.index import ExactCosineIndex
from budgetroute.schemas import RetrievalHit


def load_documents(path: Path) -> list[Document]:
    if not path.exists():
        raise RetrievalError(f"corpus path does not exist: {path}")
    files = sorted(
        item for item in path.rglob("*") if item.suffix.lower() in {".txt", ".md", ".jsonl"}
    )
    documents: list[Document] = []
    for file_path in files:
        if file_path.suffix.lower() == ".jsonl":
            for line_number, line in enumerate(
                file_path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    documents.append(Document.model_validate(raw))
                except (json.JSONDecodeError, ValueError) as exc:
                    raise RetrievalError(
                        f"invalid corpus record {file_path}:{line_number}: {exc}"
                    ) from exc
        else:
            relative = file_path.relative_to(path).as_posix()
            documents.append(
                Document(
                    document_id=relative.replace("/", "__"),
                    text=file_path.read_text(encoding="utf-8"),
                    metadata={"source": relative, "provenance": "original project sample"},
                )
            )
    if not documents:
        raise RetrievalError(f"no .txt, .md, or .jsonl documents found under {path}")
    return documents


def chunk_documents(documents: list[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    step = chunk_size - overlap
    for document in documents:
        text = " ".join(document.text.split())
        if not text:
            continue
        start = 0
        chunk_number = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        document_id=document.document_id,
                        chunk_id=f"{document.document_id}::chunk-{chunk_number:04d}",
                        text=chunk_text,
                        metadata={**document.metadata, "start": start, "end": end},
                    )
                )
                chunk_number += 1
            if end >= len(text):
                break
            start = max(end - overlap, start + step)
    return chunks


class RetrievalService:
    def __init__(self, config: RetrievalConfig, index: ExactCosineIndex) -> None:
        self.config = config
        self.index = index
        self.last_retrieval_ms = 0.0

    @property
    def ready(self) -> bool:
        return self.index.ready

    def initialize(self, *, rebuild: bool = False, persist: bool = False) -> None:
        if self.config.index_path.is_file() and not rebuild:
            self.index.load(self.config.index_path)
            return
        documents = load_documents(self.config.corpus_path)
        chunks = chunk_documents(documents, self.config.chunk_size, self.config.chunk_overlap)
        self.index.build(chunks)
        if persist:
            self.index.save(self.config.index_path)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        start = time.perf_counter_ns()
        hits = self.index.search(query, top_k or self.config.top_k)
        self.last_retrieval_ms = (time.perf_counter_ns() - start) / 1_000_000
        return [hit for hit in hits if hit.score >= self.config.min_similarity]

    def metadata(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "chunk_count": len(self.index.chunks),
            "dimension": self.index.embedder.dimension,
            "index_type": self.config.index_type,
            "index_path": str(self.config.index_path),
        }
