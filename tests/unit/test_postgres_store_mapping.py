from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from budgetroute.operations.models import ReviewOutcome, ReviewStatus
from budgetroute.operations.postgres_store import PostgreSQLOperationalStore, _json_object


class _Cursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def fetchall(self) -> list[dict[str, Any]]:
        return self.rows

    def fetchone(self) -> dict[str, Any] | None:
        return self.rows[0] if self.rows else None


class _Connection:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def execute(self, _query: str, _parameters: object = None) -> _Cursor:
        return _Cursor(self.rows)


class _ReadStore(PostgreSQLOperationalStore):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        super().__init__(require_tls=False)
        self.rows = rows

    @contextmanager
    def _connection(self) -> Iterator[_Connection]:
        yield _Connection(self.rows)


def _audit_row(
    *, sequence: int = 1, previous_hash: str = "0" * 64, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sequence": sequence,
        "event_id": f"event-{sequence}",
        "occurred_at": float(sequence),
        "tenant_id": "tenant-a",
        "actor": "reviewer-a",
        "action": "feedback.recorded",
        "target_type": "request",
        "target_id": f"request-{sequence}",
        "details_json": details or {"correct": True},
        "previous_hash": previous_hash,
    }
    canonical = json.dumps(
        {
            "event_id": row["event_id"],
            "occurred_at": row["occurred_at"],
            "tenant_id": row["tenant_id"],
            "actor": row["actor"],
            "action": row["action"],
            "target_type": row["target_type"],
            "target_id": row["target_id"],
            "details": row["details_json"],
            "previous_hash": row["previous_hash"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    row["event_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    return row


def test_postgres_row_mapping_preserves_typed_records() -> None:
    review_row = {
        "sequence": 3,
        "case_id": "case-1",
        "tenant_id": "tenant-a",
        "request_id": "request-1",
        "reason": "low confidence",
        "status": "resolved",
        "assignee": "reviewer-a",
        "outcome": "corrected",
        "correct": False,
        "version": 4,
        "created_at": 10.0,
        "updated_at": 20.0,
    }
    review = _ReadStore([review_row]).list_review_cases("tenant-a")[0]
    assert review.status == ReviewStatus.RESOLVED
    assert review.outcome == ReviewOutcome.CORRECTED
    assert review.correct is False
    assert review.created_at == "1970-01-01T00:00:10Z"

    event = _ReadStore([_audit_row()]).list_audit_events()[0]
    assert event.details == {"correct": True}
    assert event.occurred_at == "1970-01-01T00:00:01Z"

    with pytest.raises(RuntimeError, match="not an object"):
        _json_object("[]")


def test_postgres_read_models_join_labels_metrics_and_feature_summaries() -> None:
    labeled_row = {
        "tenant_id": "tenant-a",
        "request_id": "request-1",
        "route": "small",
        "backend_confidence_raw": 0.7,
        "backend_confidence": 0.8,
        "correct": True,
        "features_json": {"token_count": 12, "entropy": 0.5},
        "created_at": 10.0,
        "labeled_at": 20.0,
    }
    labeled = _ReadStore([labeled_row]).labeled_observations(limit=5)[0]
    assert labeled.correct is True
    assert labeled.features == {"token_count": 12.0, "entropy": 0.5}

    metrics = _ReadStore([{"name": "requests_total", "value": 3.5}]).metric_snapshot()
    assert metrics == {"requests_total": 3.5}

    feature_rows = [
        {"features_json": {"token_count": 10, "route": "small"}},
        {"features_json": '{"token_count":20,"cached":true}'},
    ]
    summary = _ReadStore(feature_rows).prediction_feature_summary(limit=20)
    assert summary == {
        "sample_count": 2,
        "means": {"cached": 1.0, "token_count": 15.0},
    }


def test_postgres_audit_verification_detects_hash_and_continuity_failures() -> None:
    first = _audit_row(sequence=1)
    second = _audit_row(sequence=2, previous_hash=str(first["event_hash"]))
    valid = _ReadStore([first, second]).verify_audit_chain()
    assert valid["valid"] is True
    assert valid["checked_events"] == 2
    assert valid["head_hash"] == second["event_hash"]

    tampered = dict(second)
    tampered["details_json"] = {"correct": False}
    invalid_hash = _ReadStore([first, tampered]).verify_audit_chain()
    assert invalid_hash["reason"] == "event hash mismatch"
    assert invalid_hash["failure_sequence"] == 2

    broken = _audit_row(sequence=2, previous_hash="f" * 64)
    invalid_link = _ReadStore([first, broken]).verify_audit_chain()
    assert invalid_link["reason"] == "previous_hash continuity failure"
    assert invalid_link["checked_events"] == 1
