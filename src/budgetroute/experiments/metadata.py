"""Experiment identity, environment, and Git provenance."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from budgetroute import __version__
from budgetroute.config import AppConfig
from budgetroute.environment import collect_environment


def configuration_hash(config: AppConfig) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_metadata(path: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=path,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def run_metadata(
    run_id: str,
    config: AppConfig,
    dataset_digest: str,
    repository_root: Path,
    backend_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "package_version": __version__,
        "configuration_hash": configuration_hash(config),
        "dataset_hash": dataset_digest,
        "git": git_metadata(repository_root),
        "random_seed": config.seed,
        "fake": config.benchmark.fake,
        "warmup_runs": config.benchmark.warmup_runs,
        "measured_runs": config.benchmark.measured_runs,
        "concurrency": config.benchmark.concurrency,
        "batch_size": config.benchmark.batch_size,
        "backends": backend_metadata,
    }


def environment_metadata() -> dict[str, Any]:
    return collect_environment()
