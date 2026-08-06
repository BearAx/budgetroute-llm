"""Unique experiment directories and atomic artifact writes."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from budgetroute.exceptions import ArtifactError


def create_run_directory(output_root: Path, slug: str) -> tuple[str, Path]:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base_id = f"{timestamp}_{slug}"
    output_root.mkdir(parents=True, exist_ok=True)
    for suffix in range(1000):
        run_id = base_id if suffix == 0 else f"{base_id}-{suffix:03d}"
        path = output_root / run_id
        try:
            path.mkdir()
            return run_id, path
        except FileExistsError:
            continue
    raise ArtifactError(f"could not allocate a unique run directory under {output_root}")


class ArtifactWriter:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _atomic_text(self, relative_path: str, content: str) -> Path:
        destination = self.directory / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return destination

    def write_json(self, relative_path: str, value: Any) -> Path:
        return self._atomic_text(
            relative_path, json.dumps(value, indent=2, ensure_ascii=False) + "\n"
        )

    def write_yaml(self, relative_path: str, value: Any) -> Path:
        content = yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
        return self._atomic_text(relative_path, content)

    def write_jsonl(self, relative_path: str, rows: list[dict[str, Any]]) -> Path:
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        return self._atomic_text(relative_path, content)
