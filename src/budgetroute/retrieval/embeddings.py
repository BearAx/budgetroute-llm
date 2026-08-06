"""Deterministic fake and optional real text embedding providers."""

from __future__ import annotations

import hashlib
import importlib.util
import re
from typing import Any, cast

import numpy as np

from budgetroute.exceptions import RetrievalError


class FakeEmbedder:
    """Signed feature hashing that provides deterministic lexical similarity."""

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row, index] += sign
            norm = float(np.linalg.norm(matrix[row]))
            if norm:
                matrix[row] /= norm
        return matrix


class TransformersEmbedder:
    """Lazy mean-pooling embedder backed by a configurable Transformers encoder."""

    def __init__(self, model_id: str, device: str = "cpu") -> None:
        self.model_id = model_id
        self.device = device
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._dimension: int | None = None

    def _initialize(self) -> None:
        if self._model is not None:
            return
        if any(importlib.util.find_spec(name) is None for name in ("torch", "transformers")):
            raise RetrievalError(
                "real embeddings require torch and transformers; install the transformers extra"
            )
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RetrievalError("CUDA embeddings were requested but CUDA is unavailable")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
        except Exception as exc:
            raise RetrievalError(f"could not load embedding model {self.model_id}: {exc}") from exc
        self._dimension = int(self._model.config.hidden_size)

    @property
    def dimension(self) -> int:
        self._initialize()
        assert self._dimension is not None
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        self._initialize()
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        torch = self._torch
        encoded = self._tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = self._model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
        return cast(np.ndarray, pooled.cpu().numpy().astype(np.float32))
