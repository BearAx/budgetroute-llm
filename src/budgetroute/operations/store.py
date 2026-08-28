"""Transactional SQLite operational store usable across local service replicas."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

from budgetroute.config import OperationsConfig
from budgetroute.operations.models import (
    AdmissionLease,
    AuditEvent,
    LabeledObservation,
    PredictionObservation,
    QuotaDecision,
    ReviewCase,
    ReviewOutcome,
    ReviewStatus,
)
from budgetroute.schemas import FeedbackRecord


class OperationalStore(Protocol):
    def initialize(self) -> None: ...

    def close(self) -> None: ...

    def health(self) -> dict[str, object]: ...

    def check_quota(
        self, tenant_id: str, limit: int, window_seconds: float, *, now: float | None = None
    ) -> QuotaDecision: ...

    def acquire_lease(
        self,
        tenant_id: str,
        replica_id: str,
        max_inflight: int,
        ttl_seconds: float,
        *,
        now: float | None = None,
    ) -> AdmissionLease | None: ...

    def release_lease(self, lease_id: str) -> None: ...

    def renew_lease(
        self, lease_id: str, ttl_seconds: float, *, now: float | None = None
    ) -> bool: ...

    def record_prediction(self, observation: PredictionObservation) -> bool: ...

    def record_feedback(self, tenant_id: str, actor: str, feedback: FeedbackRecord) -> bool: ...

    def create_review_case(
        self, tenant_id: str, actor: str, request_id: str, reason: str
    ) -> ReviewCase: ...

    def list_review_cases(
        self,
        tenant_id: str | None,
        *,
        status: ReviewStatus | None = None,
        after_sequence: int = 0,
        limit: int = 50,
    ) -> list[ReviewCase]: ...

    def claim_review_case(
        self, case_id: str, tenant_id: str | None, actor: str
    ) -> ReviewCase | None: ...

    def resolve_review_case(
        self,
        case_id: str,
        tenant_id: str | None,
        actor: str,
        outcome: ReviewOutcome,
        correct: bool | None,
        expected_version: int | None,
    ) -> ReviewCase | None: ...

    def increment_metric(self, name: str, value: float = 1.0) -> None: ...

    def metric_snapshot(self) -> dict[str, float]: ...

    def list_audit_events(self, after_sequence: int = 0, limit: int = 100) -> list[AuditEvent]: ...

    def labeled_observations(self, limit: int | None = None) -> list[LabeledObservation]: ...

    def verify_audit_chain(self) -> dict[str, Any]: ...

    def prediction_feature_summary(self, limit: int = 10_000) -> dict[str, Any]: ...


def _iso_time(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat().replace("+00:00", "Z")


class SQLiteOperationalStore:
    """One schema for in-memory tests and cross-process SQLite coordination."""

    def __init__(self, path: Path | None, *, audit_retention_events: int = 100_000) -> None:
        self.path = path
        self.audit_retention_events = audit_retention_events
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None

    def initialize(self) -> None:
        with self._lock:
            if self._connection is not None:
                return
            database = ":memory:" if self.path is None else str(self.path)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                database, timeout=10.0, isolation_level=None, check_same_thread=False
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 10000")
            if self.path is not None:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute("PRAGMA synchronous = NORMAL")
            self._connection = connection
            self._migrate()

    def _migrate(self) -> None:
        connection = self._require_connection()
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS quota_windows (
                tenant_id TEXT NOT NULL,
                window_start REAL NOT NULL,
                request_count INTEGER NOT NULL,
                PRIMARY KEY (tenant_id, window_start)
            );
            CREATE TABLE IF NOT EXISTS admission_leases (
                lease_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                replica_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_admission_leases_expires
                ON admission_leases(expires_at);
            CREATE TABLE IF NOT EXISTS predictions (
                tenant_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                route TEXT NOT NULL,
                backend_confidence_raw REAL,
                backend_confidence REAL,
                features_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (tenant_id, request_id)
            );
            CREATE TABLE IF NOT EXISTS feedback (
                tenant_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                correct INTEGER NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (tenant_id, request_id)
            );
            CREATE TABLE IF NOT EXISTS review_cases (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL UNIQUE,
                tenant_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL,
                assignee TEXT,
                outcome TEXT,
                correct INTEGER,
                version INTEGER NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE (tenant_id, request_id)
            );
            CREATE INDEX IF NOT EXISTS idx_review_cases_tenant_status
                ON review_cases(tenant_id, status, sequence);
            CREATE TABLE IF NOT EXISTS metrics (
                name TEXT PRIMARY KEY,
                value REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                occurred_at REAL NOT NULL,
                tenant_id TEXT NOT NULL,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                details_json TEXT NOT NULL,
                previous_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL
            );
            PRAGMA user_version = 1;
            """
        )

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("operational store is not initialized")
        return self._connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._require_connection()
        with self._lock:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def health(self) -> dict[str, object]:
        try:
            with self._lock:
                row = self._require_connection().execute("SELECT 1 AS healthy").fetchone()
            ready = row is not None and int(row["healthy"]) == 1
        except (RuntimeError, sqlite3.Error):
            ready = False
        return {
            "backend": "memory" if self.path is None else "sqlite",
            "ready": ready,
        }

    def check_quota(
        self, tenant_id: str, limit: int, window_seconds: float, *, now: float | None = None
    ) -> QuotaDecision:
        current = time.time() if now is None else now
        window_start = math.floor(current / window_seconds) * window_seconds
        with self._transaction() as connection:
            connection.execute(
                "DELETE FROM quota_windows WHERE window_start < ?",
                (window_start - window_seconds,),
            )
            row = connection.execute(
                "SELECT request_count FROM quota_windows WHERE tenant_id = ? AND window_start = ?",
                (tenant_id, window_start),
            ).fetchone()
            count = int(row["request_count"]) if row is not None else 0
            if count >= limit:
                return QuotaDecision(
                    allowed=False,
                    remaining=0,
                    retry_after_seconds=max(0.0, window_start + window_seconds - current),
                )
            if row is None:
                connection.execute(
                    "INSERT INTO quota_windows(tenant_id, window_start, request_count) VALUES (?, ?, 1)",
                    (tenant_id, window_start),
                )
            else:
                connection.execute(
                    "UPDATE quota_windows SET request_count = request_count + 1 "
                    "WHERE tenant_id = ? AND window_start = ?",
                    (tenant_id, window_start),
                )
            return QuotaDecision(
                allowed=True,
                remaining=max(0, limit - count - 1),
                retry_after_seconds=0.0,
            )

    def acquire_lease(
        self,
        tenant_id: str,
        replica_id: str,
        max_inflight: int,
        ttl_seconds: float,
        *,
        now: float | None = None,
    ) -> AdmissionLease | None:
        current = time.time() if now is None else now
        with self._transaction() as connection:
            connection.execute("DELETE FROM admission_leases WHERE expires_at <= ?", (current,))
            active = int(
                cast(
                    sqlite3.Row,
                    connection.execute("SELECT COUNT(*) AS count FROM admission_leases").fetchone(),
                )["count"]
            )
            if active >= max_inflight:
                return None
            lease = AdmissionLease(
                lease_id=str(uuid4()),
                tenant_id=tenant_id,
                replica_id=replica_id,
                expires_at_epoch=current + ttl_seconds,
            )
            connection.execute(
                "INSERT INTO admission_leases VALUES (?, ?, ?, ?, ?)",
                (lease.lease_id, tenant_id, replica_id, current, lease.expires_at_epoch),
            )
            return lease

    def release_lease(self, lease_id: str) -> None:
        with self._transaction() as connection:
            connection.execute("DELETE FROM admission_leases WHERE lease_id = ?", (lease_id,))

    def renew_lease(self, lease_id: str, ttl_seconds: float, *, now: float | None = None) -> bool:
        current = time.time() if now is None else now
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE admission_leases SET expires_at = ? WHERE lease_id = ? AND expires_at > ?",
                (current + ttl_seconds, lease_id, current),
            )
            return cursor.rowcount == 1

    def _append_audit_locked(
        self,
        connection: sqlite3.Connection,
        *,
        tenant_id: str,
        actor: str,
        action: str,
        target_type: str,
        target_id: str,
        details: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> None:
        occurred_at = time.time() if now is None else now
        previous_row = connection.execute(
            "SELECT event_hash FROM audit_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        previous_hash = str(previous_row["event_hash"]) if previous_row is not None else "0" * 64
        event_id = str(uuid4())
        details_json = json.dumps(details or {}, sort_keys=True, separators=(",", ":"))
        canonical = json.dumps(
            {
                "event_id": event_id,
                "occurred_at": occurred_at,
                "tenant_id": tenant_id,
                "actor": actor,
                "action": action,
                "target_type": target_type,
                "target_id": target_id,
                "details": json.loads(details_json),
                "previous_hash": previous_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        event_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        connection.execute(
            "INSERT INTO audit_events(event_id, occurred_at, tenant_id, actor, action, "
            "target_type, target_id, details_json, previous_hash, event_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                occurred_at,
                tenant_id,
                actor,
                action,
                target_type,
                target_id,
                details_json,
                previous_hash,
                event_hash,
            ),
        )
        count = int(
            cast(
                sqlite3.Row,
                connection.execute("SELECT COUNT(*) AS count FROM audit_events").fetchone(),
            )["count"]
        )
        excess = count - self.audit_retention_events
        if excess > 0:
            connection.execute(
                "DELETE FROM audit_events WHERE sequence IN "
                "(SELECT sequence FROM audit_events ORDER BY sequence LIMIT ?)",
                (excess,),
            )

    def record_prediction(self, observation: PredictionObservation) -> bool:
        created = observation.created_at_epoch or time.time()
        with self._transaction() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    observation.tenant_id,
                    observation.request_id,
                    observation.route,
                    observation.backend_confidence_raw,
                    observation.backend_confidence,
                    json.dumps(observation.features, sort_keys=True, separators=(",", ":")),
                    created,
                ),
            )
            return cursor.rowcount == 1

    def record_feedback(self, tenant_id: str, actor: str, feedback: FeedbackRecord) -> bool:
        with self._transaction() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO feedback VALUES (?, ?, ?, ?)",
                (tenant_id, feedback.request_id, int(feedback.correct), time.time()),
            )
            inserted = cursor.rowcount == 1
            if inserted:
                self._append_audit_locked(
                    connection,
                    tenant_id=tenant_id,
                    actor=actor,
                    action="feedback.recorded",
                    target_type="request",
                    target_id=feedback.request_id,
                    details={"correct": feedback.correct},
                )
            return inserted

    @staticmethod
    def _review_from_row(row: sqlite3.Row) -> ReviewCase:
        return ReviewCase(
            sequence=int(row["sequence"]),
            case_id=str(row["case_id"]),
            tenant_id=str(row["tenant_id"]),
            request_id=str(row["request_id"]),
            reason=str(row["reason"]),
            status=ReviewStatus(str(row["status"])),
            assignee=str(row["assignee"]) if row["assignee"] is not None else None,
            outcome=ReviewOutcome(str(row["outcome"])) if row["outcome"] is not None else None,
            correct=bool(row["correct"]) if row["correct"] is not None else None,
            version=int(row["version"]),
            created_at=_iso_time(float(row["created_at"])),
            updated_at=_iso_time(float(row["updated_at"])),
        )

    def create_review_case(
        self, tenant_id: str, actor: str, request_id: str, reason: str
    ) -> ReviewCase:
        current = time.time()
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM review_cases WHERE tenant_id = ? AND request_id = ?",
                (tenant_id, request_id),
            ).fetchone()
            if existing is not None:
                return self._review_from_row(existing)
            case_id = str(uuid4())
            connection.execute(
                "INSERT INTO review_cases(case_id, tenant_id, request_id, reason, status, "
                "version, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
                (
                    case_id,
                    tenant_id,
                    request_id,
                    reason[:1000],
                    ReviewStatus.OPEN.value,
                    current,
                    current,
                ),
            )
            self._append_audit_locked(
                connection,
                tenant_id=tenant_id,
                actor=actor,
                action="review.created",
                target_type="review_case",
                target_id=case_id,
                details={"request_id": request_id},
                now=current,
            )
            row = connection.execute(
                "SELECT * FROM review_cases WHERE case_id = ?", (case_id,)
            ).fetchone()
            assert row is not None
            return self._review_from_row(row)

    def list_review_cases(
        self,
        tenant_id: str | None,
        *,
        status: ReviewStatus | None = None,
        after_sequence: int = 0,
        limit: int = 50,
    ) -> list[ReviewCase]:
        clauses = ["sequence > ?"]
        values: list[Any] = [after_sequence]
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            values.append(tenant_id)
        if status is not None:
            clauses.append("status = ?")
            values.append(status.value)
        values.append(min(200, max(1, limit)))
        with self._lock:
            rows = (
                self._require_connection()
                .execute(
                    f"SELECT * FROM review_cases WHERE {' AND '.join(clauses)} "
                    "ORDER BY sequence LIMIT ?",
                    values,
                )
                .fetchall()
            )
        return [self._review_from_row(row) for row in rows]

    def claim_review_case(
        self, case_id: str, tenant_id: str | None, actor: str
    ) -> ReviewCase | None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM review_cases WHERE case_id = ?", (case_id,)
            ).fetchone()
            if row is None or (tenant_id is not None and row["tenant_id"] != tenant_id):
                return None
            if row["status"] == ReviewStatus.CLAIMED.value and row["assignee"] == actor:
                return self._review_from_row(row)
            if row["status"] != ReviewStatus.OPEN.value:
                return None
            current = time.time()
            connection.execute(
                "UPDATE review_cases SET status = ?, assignee = ?, version = version + 1, "
                "updated_at = ? WHERE case_id = ?",
                (ReviewStatus.CLAIMED.value, actor, current, case_id),
            )
            self._append_audit_locked(
                connection,
                tenant_id=str(row["tenant_id"]),
                actor=actor,
                action="review.claimed",
                target_type="review_case",
                target_id=case_id,
                now=current,
            )
            updated = connection.execute(
                "SELECT * FROM review_cases WHERE case_id = ?", (case_id,)
            ).fetchone()
            assert updated is not None
            return self._review_from_row(updated)

    def resolve_review_case(
        self,
        case_id: str,
        tenant_id: str | None,
        actor: str,
        outcome: ReviewOutcome,
        correct: bool | None,
        expected_version: int | None,
    ) -> ReviewCase | None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM review_cases WHERE case_id = ?", (case_id,)
            ).fetchone()
            if row is None or (tenant_id is not None and row["tenant_id"] != tenant_id):
                return None
            if row["status"] not in {ReviewStatus.OPEN.value, ReviewStatus.CLAIMED.value}:
                return None
            if expected_version is not None and int(row["version"]) != expected_version:
                return None
            current = time.time()
            status = (
                ReviewStatus.DISMISSED
                if outcome == ReviewOutcome.NOT_ACTIONABLE
                else ReviewStatus.RESOLVED
            )
            connection.execute(
                "UPDATE review_cases SET status = ?, assignee = COALESCE(assignee, ?), "
                "outcome = ?, correct = ?, version = version + 1, updated_at = ? "
                "WHERE case_id = ?",
                (status.value, actor, outcome.value, correct, current, case_id),
            )
            self._append_audit_locked(
                connection,
                tenant_id=str(row["tenant_id"]),
                actor=actor,
                action="review.resolved",
                target_type="review_case",
                target_id=case_id,
                details={"outcome": outcome.value, "correct": correct},
                now=current,
            )
            if correct is not None:
                connection.execute(
                    "INSERT OR IGNORE INTO feedback VALUES (?, ?, ?, ?)",
                    (str(row["tenant_id"]), str(row["request_id"]), int(correct), current),
                )
            updated = connection.execute(
                "SELECT * FROM review_cases WHERE case_id = ?", (case_id,)
            ).fetchone()
            assert updated is not None
            return self._review_from_row(updated)

    def increment_metric(self, name: str, value: float = 1.0) -> None:
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO metrics(name, value) VALUES (?, ?) "
                "ON CONFLICT(name) DO UPDATE SET value = value + excluded.value",
                (name, value),
            )

    def metric_snapshot(self) -> dict[str, float]:
        with self._lock:
            rows = (
                self._require_connection()
                .execute("SELECT name, value FROM metrics ORDER BY name")
                .fetchall()
            )
        return {str(row["name"]): float(row["value"]) for row in rows}

    def list_audit_events(self, after_sequence: int = 0, limit: int = 100) -> list[AuditEvent]:
        with self._lock:
            rows = (
                self._require_connection()
                .execute(
                    "SELECT * FROM audit_events WHERE sequence > ? ORDER BY sequence LIMIT ?",
                    (after_sequence, min(500, max(1, limit))),
                )
                .fetchall()
            )
        return [
            AuditEvent(
                sequence=int(row["sequence"]),
                event_id=str(row["event_id"]),
                occurred_at=_iso_time(float(row["occurred_at"])),
                tenant_id=str(row["tenant_id"]),
                actor=str(row["actor"]),
                action=str(row["action"]),
                target_type=str(row["target_type"]),
                target_id=str(row["target_id"]),
                details=json.loads(str(row["details_json"])),
                previous_hash=str(row["previous_hash"]),
                event_hash=str(row["event_hash"]),
            )
            for row in rows
        ]

    def labeled_observations(self, limit: int | None = None) -> list[LabeledObservation]:
        query = (
            "SELECT p.*, f.correct, f.created_at AS labeled_at FROM predictions p "
            "JOIN feedback f ON f.tenant_id = p.tenant_id AND f.request_id = p.request_id "
            "WHERE p.backend_confidence_raw IS NOT NULL ORDER BY f.created_at, p.request_id"
        )
        parameters: tuple[int, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            parameters = (max(1, limit),)
        with self._lock:
            rows = self._require_connection().execute(query, parameters).fetchall()
        return [
            LabeledObservation(
                tenant_id=str(row["tenant_id"]),
                request_id=str(row["request_id"]),
                route=str(row["route"]),
                backend_confidence_raw=float(row["backend_confidence_raw"]),
                backend_confidence=(
                    float(row["backend_confidence"])
                    if row["backend_confidence"] is not None
                    else None
                ),
                correct=bool(row["correct"]),
                features=json.loads(str(row["features_json"])),
                predicted_at_epoch=float(row["created_at"]),
                labeled_at_epoch=float(row["labeled_at"]),
            )
            for row in rows
        ]

    def verify_audit_chain(self) -> dict[str, Any]:
        with self._lock:
            rows = (
                self._require_connection()
                .execute("SELECT * FROM audit_events ORDER BY sequence")
                .fetchall()
            )
        previous: str | None = None
        for checked, row in enumerate(rows):
            previous_hash = str(row["previous_hash"])
            if previous is not None and previous_hash != previous:
                return {
                    "valid": False,
                    "checked_events": checked,
                    "failure_sequence": int(row["sequence"]),
                    "reason": "previous_hash continuity failure",
                }
            details = json.loads(str(row["details_json"]))
            canonical = json.dumps(
                {
                    "event_id": str(row["event_id"]),
                    "occurred_at": float(row["occurred_at"]),
                    "tenant_id": str(row["tenant_id"]),
                    "actor": str(row["actor"]),
                    "action": str(row["action"]),
                    "target_type": str(row["target_type"]),
                    "target_id": str(row["target_id"]),
                    "details": details,
                    "previous_hash": previous_hash,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if expected != row["event_hash"]:
                return {
                    "valid": False,
                    "checked_events": checked,
                    "failure_sequence": int(row["sequence"]),
                    "reason": "event hash mismatch",
                }
            previous = expected
        return {
            "valid": True,
            "checked_events": len(rows),
            "first_previous_hash": str(rows[0]["previous_hash"]) if rows else None,
            "head_hash": previous,
        }

    def prediction_feature_summary(self, limit: int = 10_000) -> dict[str, Any]:
        with self._lock:
            rows = (
                self._require_connection()
                .execute(
                    "SELECT features_json FROM predictions ORDER BY created_at DESC LIMIT ?",
                    (max(1, limit),),
                )
                .fetchall()
            )
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for row in rows:
            features = json.loads(str(row["features_json"]))
            for name, value in features.items():
                if not isinstance(value, (bool, int, float)):
                    continue
                totals[name] = totals.get(name, 0.0) + float(value)
                counts[name] = counts.get(name, 0) + 1
        return {
            "sample_count": len(rows),
            "means": {name: totals[name] / counts[name] for name in sorted(totals) if counts[name]},
        }


def build_store(config: OperationsConfig) -> OperationalStore:
    if config.backend == "postgres":
        from budgetroute.operations.postgres_store import PostgreSQLOperationalStore

        return PostgreSQLOperationalStore(
            dsn_env=config.postgres_dsn_env,
            pool_min_size=config.postgres_pool_min_size,
            pool_max_size=config.postgres_pool_max_size,
            connect_timeout_seconds=config.postgres_connect_timeout_seconds,
            require_tls=config.postgres_require_tls,
            auto_migrate=config.postgres_auto_migrate,
            audit_retention_events=config.audit_retention_events,
        )
    path = config.database_path if config.backend == "sqlite" else None
    return SQLiteOperationalStore(path, audit_retention_events=config.audit_retention_events)
