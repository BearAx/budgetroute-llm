from __future__ import annotations

import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from budgetroute.operations.models import PredictionObservation, ReviewOutcome, ReviewStatus
from budgetroute.operations.postgres_store import PostgreSQLOperationalStore
from budgetroute.schemas import FeedbackRecord

pytestmark = pytest.mark.postgres

_TEST_DSN = os.environ.get("BUDGETROUTE_TEST_POSTGRES_DSN")
if not _TEST_DSN:
    pytest.skip(
        "BUDGETROUTE_TEST_POSTGRES_DSN is required for PostgreSQL integration tests",
        allow_module_level=True,
    )

psycopg = pytest.importorskip("psycopg")
from psycopg import sql  # noqa: E402
from psycopg.conninfo import make_conninfo  # noqa: E402


@pytest.fixture
def postgres_stores(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[list[PostgreSQLOperationalStore], str]]:
    assert _TEST_DSN is not None
    schema = f"budgetroute_test_{uuid4().hex}"
    with psycopg.connect(_TEST_DSN, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    isolated_dsn = make_conninfo(_TEST_DSN, options=f"-c search_path={schema}")
    monkeypatch.setenv("BUDGETROUTE_TEST_ISOLATED_DSN", isolated_dsn)
    stores = [
        PostgreSQLOperationalStore(
            dsn_env="BUDGETROUTE_TEST_ISOLATED_DSN",
            pool_min_size=1,
            pool_max_size=8,
            connect_timeout_seconds=10,
            require_tls=False,
        )
        for _ in range(2)
    ]
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lambda store: store.initialize(), stores))
        yield stores, isolated_dsn
    finally:
        for store in stores:
            store.close()
        with psycopg.connect(_TEST_DSN, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def test_postgres_store_matches_durable_workflow_contract(
    postgres_stores: tuple[list[PostgreSQLOperationalStore], str],
) -> None:
    stores, isolated_dsn = postgres_stores
    first, second = stores
    assert first.health() == {"backend": "postgres", "ready": True}

    assert first.check_quota("tenant-a", 2, 60.0, now=10.0).remaining == 1
    assert second.check_quota("tenant-a", 2, 60.0, now=11.0).remaining == 0
    rejected = first.check_quota("tenant-a", 2, 60.0, now=12.0)
    assert rejected.allowed is False
    assert rejected.retry_after_seconds == 48.0

    lease = first.acquire_lease("tenant-a", "replica-1", 1, 30.0, now=20.0)
    assert lease is not None
    assert second.acquire_lease("tenant-b", "replica-2", 1, 30.0, now=21.0) is None
    assert second.renew_lease(lease.lease_id, 30.0, now=40.0) is True
    first.release_lease(lease.lease_id)

    observation = PredictionObservation(
        tenant_id="tenant-a",
        request_id="request-1",
        route="small",
        backend_confidence_raw=0.7,
        backend_confidence=0.8,
        features={"token_count": 12.0},
        created_at_epoch=10.0,
    )
    assert first.record_prediction(observation) is True
    assert second.record_prediction(observation) is False
    feedback = FeedbackRecord(request_id="request-1", correct=True, notes="not persisted")
    assert second.record_feedback("tenant-a", "reviewer-a", feedback) is True
    assert first.record_feedback("tenant-a", "reviewer-a", feedback) is False
    assert first.labeled_observations()[0].features == {"token_count": 12.0}

    review = first.create_review_case("tenant-a", "router", "request-2", "low confidence")
    duplicate = second.create_review_case("tenant-a", "router", "request-2", "duplicate")
    assert duplicate.case_id == review.case_id
    claimed = second.claim_review_case(review.case_id, "tenant-a", "reviewer-a")
    assert claimed is not None and claimed.status == ReviewStatus.CLAIMED
    assert (
        first.resolve_review_case(
            review.case_id,
            "tenant-a",
            "reviewer-a",
            ReviewOutcome.APPROVED,
            True,
            expected_version=1,
        )
        is None
    )
    resolved = first.resolve_review_case(
        review.case_id,
        "tenant-a",
        "reviewer-a",
        ReviewOutcome.APPROVED,
        True,
        expected_version=2,
    )
    assert resolved is not None and resolved.status == ReviewStatus.RESOLVED
    assert second.list_review_cases("another-tenant") == []

    first.increment_metric("requests_total", 2)
    second.increment_metric("requests_total", 3)
    assert first.metric_snapshot()["requests_total"] == 5
    assert first.prediction_feature_summary()["means"]["token_count"] == 12
    assert first.verify_audit_chain()["valid"] is True

    events = first.list_audit_events()
    with psycopg.connect(isolated_dsn, autocommit=True) as connection:
        connection.execute(
            "UPDATE audit_events SET details_json = '{\"tampered\": true}'::jsonb "
            "WHERE sequence = %s",
            (events[1].sequence,),
        )
    assert first.verify_audit_chain()["valid"] is False


def test_postgres_global_operations_are_atomic_under_concurrency(
    postgres_stores: tuple[list[PostgreSQLOperationalStore], str],
) -> None:
    stores, _ = postgres_stores

    def quota_call(index: int) -> bool:
        return stores[index % 2].check_quota("tenant-concurrent", 17, 60.0, now=10.0).allowed

    with ThreadPoolExecutor(max_workers=16) as executor:
        quota_results = list(executor.map(quota_call, range(80)))
    assert sum(quota_results) == 17

    def lease_call(index: int):
        return stores[index % 2].acquire_lease(
            f"tenant-{index}", f"replica-{index % 2}", 5, 30.0, now=20.0
        )

    with ThreadPoolExecutor(max_workers=16) as executor:
        leases = [lease for lease in executor.map(lease_call, range(40)) if lease is not None]
    assert len(leases) == 5

    def feedback_call(index: int) -> bool:
        return stores[index % 2].record_feedback(
            "tenant-concurrent",
            f"reviewer-{index}",
            FeedbackRecord(request_id=f"request-{index}", correct=index % 2 == 0),
        )

    with ThreadPoolExecutor(max_workers=16) as executor:
        assert all(executor.map(feedback_call, range(40)))
    audit = stores[0].verify_audit_chain()
    assert audit["valid"] is True
    assert audit["checked_events"] == 40
