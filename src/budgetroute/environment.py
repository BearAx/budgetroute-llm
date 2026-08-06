"""Environment discovery used by doctor output and experiment metadata."""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from pathlib import Path
from typing import Any

import psutil

from budgetroute import __version__


def optional_dependency_status() -> dict[str, bool]:
    names = ["torch", "transformers", "fastapi", "sklearn", "faiss", "pandas", "matplotlib"]
    return {name: importlib.util.find_spec(name) is not None for name in names}


def torch_environment() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"installed": False, "version": None, "cuda_available": False, "gpu": None}
    import torch

    cuda_available = bool(torch.cuda.is_available())
    gpu = torch.cuda.get_device_name(0) if cuda_available else None
    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu": gpu,
    }


def collect_environment() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    return {
        "package_version": __version__,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "memory_total_mb": round(memory.total / (1024 * 1024), 2),
        "torch": torch_environment(),
        "optional_dependencies": optional_dependency_status(),
    }


def writable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".budgetroute-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError:
        return False
    return True
