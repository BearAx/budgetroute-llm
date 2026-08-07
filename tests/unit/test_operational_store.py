from __future__ import annotations

import sqlite3
from pathlib import Path

from budgetroute.operations.models import (
    PredictionObservation,
    ReviewOutcome,
    ReviewStatus,
)
from budgetroute.operations.store import SQLiteOperationalStore
from budgetroute.schemas import FeedbackRecord


def test_sqlite_store_coordinates_quota_and_leases_across_instances(tmp_path: Path) -> None:
    database = tmp_path / "operations.db"
    first = SQLiteOperationalStore(database)
    second = SQLiteOperationalStore(database)
    first.initialize()
    second.initialize()
    try:
        assert first.check_quota("tenant-a", 2, 60.0, now=10.0).remaining == 1
        assert second.check_quota("tenant-a", 2, 60.0, now=11.0).remaining == 0
        rejected = first.check_quota("tenant-a", 2, 60.0, now=12.0)
        assert rejected.allowed is False
        assert rejected.retry_after_seconds == 48.0
        assert second.check_quota("tenant-a", 2, 60.0, now=61.0).allowed is True

        lease = first.acquire_lease("tenant-a", "replica-1", 1, 30.0, now=20.0)
        assert lease is not None
        assert second.acquire_lease("tenant-b", "replica-2", 1, 30.0, now=21.0) is None
        assert second.renew_lease(lease.lease_id, 30.0, now=40.0) is True
        assert second.acquire_lease("tenant-b", "replica-2", 1, 30.0, now=51.0) is None
        first.release_lease(lease.lease_id)
        assert second.acquire_lease("tenant-b", "replica-2", 1, 30.0, now=22.0)
    finally:
        first.close()
        second.close()


def test_store_joins_idempotent_delayed_feedback_and_predictions() -> None:
    store = SQLiteOperationalStore(None)
    store.initialize()
    try:
        observation = PredictionObservation(
            tenant_id="tenant-a",
            request_id="request-1",
            route="small",
            backend_confidence_raw=0.7,
            backend_confidence=0.8,
            features={"token_count": 12.0},
            created_at_epoch=10.0,
        )
        assert store.record_prediction(observation) is True
        assert store.record_prediction(observation) is False
        feedback = FeedbackRecord(request_id="request-1", correct=True, notes="not persisted")
        assert store.record_feedback("tenant-a", "reviewer-a", feedback) is True
        assert store.record_feedback("tenant-a", "reviewer-a", feedback) is False

        labeled = store.labeled_observations()
        assert len(labeled) == 1
        assert labeled[0].correct is True
        assert labeled[0].features == {"token_count": 12.0}
        assert "notes" not in store.list_audit_events()[0].details
    finally:
        store.close()


def test_review_case_lifecycle_is_versioned_and_audited(tmp_path: Path) -> None:
    database = tmp_path / "audit.db"
    store = SQLiteOperationalStore(database)
    store.initialize()
    try:
        review = store.create_review_case("tenant-a", "router", "request-1", "low confidence")
        assert review.status == ReviewStatus.OPEN
        assert (
            store.create_review_case("tenant-a", "router", "request-1", "duplicate").case_id
            == review.case_id
        )

        claimed = store.claim_review_case(review.case_id, "tenant-a", "reviewer-a")
        assert claimed is not None
        assert claimed.status == ReviewStatus.CLAIMED
        assert claimed.version == 2
        assert (
            store.resolve_review_case(
                review.case_id,
                "tenant-a",
                "reviewer-a",
                ReviewOutcome.APPROVED,
                True,
                expected_version=1,
            )
            is None
        )
        resolved = store.resolve_review_case(
            review.case_id,
            "tenant-a",
            "reviewer-a",
            ReviewOutcome.APPROVED,
            True,
            expected_version=2,
        )
        assert resolved is not None
        assert resolved.status == ReviewStatus.RESOLVED
        assert resolved.correct is True
        assert store.list_review_cases("another-tenant") == []

        events = store.list_audit_events()
        assert [event.action for event in events] == [
            "review.created",
            "review.claimed",
            "review.resolved",
        ]
        assert events[1].previous_hash == events[0].event_hash
        assert events[2].previous_hash == events[1].event_hash
        assert store.verify_audit_chain()["valid"] is True
        assert store.prediction_feature_summary()["sample_count"] == 0
        connection = sqlite3.connect(database)
        try:
            connection.execute(
                "UPDATE audit_events SET details_json = ? WHERE sequence = ?",
                ('{"tampered":true}', events[1].sequence),
            )
            connection.commit()
        finally:
            connection.close()
        invalid = store.verify_audit_chain()
        assert invalid["valid"] is False
        assert invalid["checked_events"] == 1
    finally:
        store.close()
