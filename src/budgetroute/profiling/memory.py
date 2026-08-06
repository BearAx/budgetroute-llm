"""Process and optional CUDA memory measurement."""

from __future__ import annotations

import importlib.util

import psutil

from budgetroute.schemas import MemoryStats


def current_memory() -> MemoryStats:
    rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    peak_cuda_mb: float | None = None
    if importlib.util.find_spec("torch") is not None:
        import torch

        if torch.cuda.is_available():
            peak_cuda_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return MemoryStats(process_rss_mb=rss_mb, peak_cuda_mb=peak_cuda_mb)
