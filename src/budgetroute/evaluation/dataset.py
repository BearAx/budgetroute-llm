"""Validated JSONL benchmark datasets and future adapter protocol."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from budgetroute.exceptions import BudgetRouteError
from budgetroute.schemas import BenchmarkRecord


class DatasetAdapter(Protocol):
    def load(self) -> list[BenchmarkRecord]: ...


class JsonlDatasetAdapter:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[BenchmarkRecord]:
        return load_dataset(self.path)


def load_dataset(path: str | Path) -> list[BenchmarkRecord]:
    source = Path(path)
    if not source.is_file():
        raise BudgetRouteError(f"benchmark dataset does not exist: {source}")
    records: list[BenchmarkRecord] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = BenchmarkRecord.model_validate_json(line)
        except ValidationError as exc:
            raise BudgetRouteError(
                f"invalid benchmark record {source}:{line_number}: {exc}"
            ) from exc
        if record.id in seen_ids:
            raise BudgetRouteError(f"duplicate benchmark ID {record.id!r} at line {line_number}")
        seen_ids.add(record.id)
        records.append(record)
    if not records:
        raise BudgetRouteError(f"benchmark dataset is empty: {source}")
    return records


def dataset_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_summary(records: list[BenchmarkRecord]) -> dict[str, object]:
    categories: dict[str, int] = {}
    evaluation_types: dict[str, int] = {}
    for record in records:
        categories[record.category] = categories.get(record.category, 0) + 1
        value = record.evaluation_type.value
        evaluation_types[value] = evaluation_types.get(value, 0) + 1
    return {
        "records": len(records),
        "categories": categories,
        "evaluation_types": evaluation_types,
        "requires_retrieval": sum(record.requires_retrieval for record in records),
        "must_abstain": sum(record.must_abstain for record in records),
    }
