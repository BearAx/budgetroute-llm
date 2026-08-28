"""Pooled PostgreSQL operational store for coordination across service hosts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.resources import files
from typing import Any
from uuid import uuid4

from budgetroute.exceptions import ConfigurationError
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

_MIGRATION_LOCK = 1_883_441_001
_ADMISSION_LOCK = 1_883_441_002
_AUDIT_LOCK = 1_883_441_003
_TLS_MODES = {"require", "verify-ca", "verify-full"}
_MIGRATION_PACKAGE = "budgetroute.operations.migrations.postgres"


def _iso_time(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat().replace("+00:00", "Z")


def _json_object(value: object) -> dict[str, Any]:
    decoded = json.loads(value) if isinstance(value, str) else value
    if not isinstance(decoded, dict):
        raise RuntimeError("operational JSON record is not an object")
    return {str(key): item for key, item in decoded.items()}


class PostgreSQLOperationalStore:
    """Transactional control-plane state shared through a PostgreSQL cluster."""

    def __init__(
        self,
        *,
        dsn_env: str = "BUDGETROUTE_POSTGRES_DSN",
        pool_min_size: int = 1,
        pool_max_size: int = 16,
        connect_timeout_seconds: float = 10.0,
        require_tls: bool = True,
        auto_migrate: bool = True,
        audit_retention_events: int = 100_000,
    ) -> None:
        self.dsn_env = dsn_env
        self.pool_min_size = pool_min_size
        self.pool_max_size = pool_max_size
        self.connect_timeout_seconds = connect_timeout_seconds
        self.require_tls = require_tls
        self.auto_migrate = auto_migrate
        self.audit_retention_events = audit_retention_events
        self._pool: Any | None = None

    def initialize(self) -> None:
        if self._pool is not None:
            return
        dsn = os.environ.get(self.dsn_env)
        if not dsn:
            raise ConfigurationError(
                f"PostgreSQL operational store requires environment variable {self.dsn_env}"
            )
        try:
            from psycopg.conninfo import conninfo_to_dict
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise ConfigurationError(
                "PostgreSQL operational store requires the 'postgres' optional dependency"
            ) from exc

        try:
            parameters = conninfo_to_dict(dsn)
        except Exception as exc:
            raise ConfigurationError(
                "PostgreSQL operational-store DSN is invalid "
                f"({type(exc).__name__}); secret-bearing parser details were suppressed"
            ) from None
        sslmode = str(parameters.get("sslmode", "prefer")).lower()
        if self.require_tls and sslmode not in _TLS_MODES:
            raise ConfigurationError(
                f"{self.dsn_env} must set sslmode=require, verify-ca, or verify-full"
            )

        try:
            pool = ConnectionPool(
                conninfo=dsn,
                min_size=self.pool_min_size,
                max_size=self.pool_max_size,
                timeout=self.connect_timeout_seconds,
                kwargs={
                    "application_name": "budgetroute-operational-store",
                    "connect_timeout": max(1, math.ceil(self.connect_timeout_seconds)),
                    "row_factory": dict_row,
                },
                open=False,
            )
            self._pool = pool
            pool.open(wait=True, timeout=self.connect_timeout_seconds)
            if self.auto_migrate:
                self._migrate()
            else:
                self._verify_migrations()
        except ConfigurationError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise ConfigurationError(
                "could not initialize PostgreSQL operational store "
                f"({type(exc).__name__}); connection details were suppressed"
            ) from None

    def _require_pool(self) -> Any:
        if self._pool is None:
            raise RuntimeError("operational store is not initialized")
        return self._pool

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        with self._require_pool().connection() as connection:
            yield connection

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def health(self) -> dict[str, object]:
        try:
            pool = self._require_pool()
            with pool.connection(timeout=min(2.0, self.connect_timeout_seconds)) as connection:
                row = connection.execute("SELECT 1 AS healthy").fetchone()
            ready = row is not None and int(row["healthy"]) == 1
        except Exception:
            ready = False
        return {"backend": "postgres", "ready": ready}

    def _migrate(self) -> None:
        migration_files = sorted(
            (entry for entry in files(_MIGRATION_PACKAGE).iterdir() if entry.name.endswith(".sql")),
            key=lambda entry: entry.name,
        )
        available = {entry.name.removesuffix(".sql") for entry in migration_files}
        with self._connection() as connection:
            connection.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK,))
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS budgetroute_schema_migrations (
                    version TEXT PRIMARY KEY,
                    checksum_sha256 CHAR(64) NOT NULL,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            applied_rows = connection.execute(
                "SELECT version, checksum_sha256 FROM budgetroute_schema_migrations"
            ).fetchall()
            applied = {str(row["version"]): str(row["checksum_sha256"]) for row in applied_rows}
            unknown = sorted(set(applied) - available)
            if unknown:
                raise ConfigurationError(
                    "database contains migrations newer than this application: "
                    + ", ".join(unknown)
                )
            for resource in migration_files:
                version = resource.name.removesuffix(".sql")
                sql = resource.read_text(encoding="utf-8")
                checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
                if version in applied:
                    if applied[version] != checksum:
                        raise ConfigurationError(
                            f"PostgreSQL migration checksum mismatch for {version}"
                        )
                    continue
                connection.execute(sql)
                connection.execute(
                    "INSERT INTO budgetroute_schema_migrations(version, checksum_sha256) "
                    "VALUES (%s, %s)",
                    (version, checksum),
                )

    def _verify_migrations(self) -> None:
        migration_files = sorted(
            (entry for entry in files(_MIGRATION_PACKAGE).iterdir() if entry.name.endswith(".sql")),
            key=lambda entry: entry.name,
        )
        expected = {
            entry.name.removesuffix(".sql"): hashlib.sha256(
                entry.read_text(encoding="utf-8").encode("utf-8")
            ).hexdigest()
            for entry in migration_files
        }
        with self._connection() as connection:
            connection.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK,))
            table = connection.execute(
                "SELECT to_regclass('budgetroute_schema_migrations') AS name"
            ).fetchone()
            if table is None or table["name"] is None:
                raise ConfigurationError(
                    "PostgreSQL schema is not initialized; run budgetroute migrate-store"
                )
            rows = connection.execute(
                "SELECT version, checksum_sha256 FROM budgetroute_schema_migrations"
            ).fetchall()
        applied = {str(row["version"]): str(row["checksum_sha256"]) for row in rows}
        unknown = sorted(set(applied) - set(expected))
        missing = sorted(set(expected) - set(applied))
        if unknown:
            raise ConfigurationError(
                "database contains migrations newer than this application: " + ", ".join(unknown)
            )
        if missing:
            raise ConfigurationError(
                "database is missing required migrations: " + ", ".join(missing)
            )
        mismatched = sorted(
            version for version, checksum in expected.items() if applied[version] != checksum
        )
        if mismatched:
            raise ConfigurationError(
                "PostgreSQL migration checksum mismatch for " + ", ".join(mismatched)
            )

    @staticmethod
    def _current_epoch(connection: Any, now: float | None) -> float:
        if now is not None:
            return now
        row = connection.execute(
            "SELECT EXTRACT(EPOCH FROM clock_timestamp())::double precision AS epoch"
        ).fetchone()
        if row is None:
            raise RuntimeError("PostgreSQL did not return database time")
        return float(row["epoch"])

    def check_quota(
        self, tenant_id: str, limit: int, window_seconds: float, *, now: float | None = None
    ) -> QuotaDecision:
        if limit < 1 or window_seconds <= 0:
            raise ValueError("quota limit and window must be positive")
        with self._connection() as connection:
            current = self._current_epoch(connection, now)
            window_start = math.floor(current / window_seconds) * window_seconds
            connection.execute(
                "DELETE FROM quota_windows WHERE tenant_id = %s AND window_start < %s",
                (tenant_id, window_start),
            )
            row = connection.execute(
                """
                INSERT INTO quota_windows(tenant_id, window_start, request_count)
                VALUES (%s, %s, 1)
                ON CONFLICT (tenant_id, window_start) DO UPDATE
                SET request_count = quota_windows.request_count + 1
                WHERE quota_windows.request_count < %s
                RETURNING request_count
                """,
                (tenant_id, window_start, limit),
            ).fetchone()
            if row is None:
                return QuotaDecision(
                    allowed=False,
                    remaining=0,
                    retry_after_seconds=max(0.0, window_start + window_seconds - current),
                )
            count = int(row["request_count"])
            return QuotaDecision(
                allowed=True,
                remaining=max(0, limit - count),
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
        if max_inflight < 1 or ttl_seconds <= 0:
            raise ValueError("lease capacity and TTL must be positive")
        with self._connection() as connection:
            connection.execute("SELECT pg_advisory_xact_lock(%s)", (_ADMISSION_LOCK,))
            current = self._current_epoch(connection, now)
            connection.execute("DELETE FROM admission_leases WHERE expires_at <= %s", (current,))
            row = connection.execute("SELECT COUNT(*) AS count FROM admission_leases").fetchone()
            if row is None:
                raise RuntimeError("PostgreSQL did not return the active lease count")
            if int(row["count"]) >= max_inflight:
                return None
            lease = AdmissionLease(
                lease_id=str(uuid4()),
                tenant_id=tenant_id,
                replica_id=replica_id,
                expires_at_epoch=current + ttl_seconds,
            )
            connection.execute(
                "INSERT INTO admission_leases"
                "(lease_id, tenant_id, replica_id, created_at, expires_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    lease.lease_id,
                    lease.tenant_id,
                    lease.replica_id,
                    current,
                    lease.expires_at_epoch,
                ),
            )
            return lease

    def release_lease(self, lease_id: str) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM admission_leases WHERE lease_id = %s", (lease_id,))

    def renew_lease(self, lease_id: str, ttl_seconds: float, *, now: float | None = None) -> bool:
        if ttl_seconds <= 0:
            raise ValueError("lease TTL must be positive")
        with self._connection() as connection:
            current = self._current_epoch(connection, now)
            cursor = connection.execute(
                "UPDATE admission_leases SET expires_at = %s "
                "WHERE lease_id = %s AND expires_at > %s",
                (current + ttl_seconds, lease_id, current),
            )
            return bool(cursor.rowcount == 1)

    def _append_audit_locked(
        self,
        connection: Any,
        *,
        tenant_id: str,
        actor: str,
        action: str,
        target_type: str,
        target_id: str,
        details: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> None:
        connection.execute("SELECT pg_advisory_xact_lock(%s)", (_AUDIT_LOCK,))
        occurred_at = self._current_epoch(connection, now)
        previous_row = connection.execute(
            "SELECT event_hash FROM audit_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        previous_hash = str(previous_row["event_hash"]) if previous_row is not None else "0" * 64
        event_id = str(uuid4())
        details_value = details or {}
        details_json = json.dumps(details_value, sort_keys=True, separators=(",", ":"))
        canonical = json.dumps(
            {
                "event_id": event_id,
                "occurred_at": occurred_at,
                "tenant_id": tenant_id,
                "actor": actor,
                "action": action,
                "target_type": target_type,
                "target_id": target_id,
                "details": details_value,
                "previous_hash": previous_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        event_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        connection.execute(
            """
            INSERT INTO audit_events(
                event_id, occurred_at, tenant_id, actor, action, target_type,
                target_id, details_json, previous_hash, event_hash
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
            """,
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
        connection.execute(
            "DELETE FROM audit_events WHERE sequence IN "
            "(SELECT sequence FROM audit_events ORDER BY sequence DESC OFFSET %s)",
            (self.audit_retention_events,),
        )

    def record_prediction(self, observation: PredictionObservation) -> bool:
        with self._connection() as connection:
            created = self._current_epoch(connection, observation.created_at_epoch)
            cursor = connection.execute(
                """
                INSERT INTO predictions(
                    tenant_id, request_id, route, backend_confidence_raw,
                    backend_confidence, features_json, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (tenant_id, request_id) DO NOTHING
                """,
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
            return bool(cursor.rowcount == 1)

    def record_feedback(self, tenant_id: str, actor: str, feedback: FeedbackRecord) -> bool:
        with self._connection() as connection:
            current = self._current_epoch(connection, None)
            cursor = connection.execute(
                "INSERT INTO feedback(tenant_id, request_id, correct, created_at) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (tenant_id, request_id) DO NOTHING",
                (tenant_id, feedback.request_id, feedback.correct, current),
            )
            inserted = bool(cursor.rowcount == 1)
            if inserted:
                self._append_audit_locked(
                    connection,
                    tenant_id=tenant_id,
                    actor=actor,
                    action="feedback.recorded",
                    target_type="request",
                    target_id=feedback.request_id,
                    details={"correct": feedback.correct},
                    now=current,
                )
            return inserted

    @staticmethod
    def _review_from_row(row: Mapping[str, Any]) -> ReviewCase:
        return ReviewCase(
            sequence=int(row["sequence"]),
            case_id=str(row["case_id"]),
            tenant_id=str(row["tenant_id"]),
            request_id=str(row["request_id"]),
            reason=str(row["reason"]),
            status=ReviewStatus(str(row["status"])),
            assignee=str(row["assignee"]) if row["assignee"] is not None else None,
            outcome=(ReviewOutcome(str(row["outcome"])) if row["outcome"] is not None else None),
            correct=bool(row["correct"]) if row["correct"] is not None else None,
            version=int(row["version"]),
            created_at=_iso_time(float(row["created_at"])),
            updated_at=_iso_time(float(row["updated_at"])),
        )

    def create_review_case(
        self, tenant_id: str, actor: str, request_id: str, reason: str
    ) -> ReviewCase:
        with self._connection() as connection:
            current = self._current_epoch(connection, None)
            case_id = str(uuid4())
            row = connection.execute(
                """
                INSERT INTO review_cases(
                    case_id, tenant_id, request_id, reason, status,
                    version, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, 1, %s, %s)
                ON CONFLICT (tenant_id, request_id) DO NOTHING
                RETURNING *
                """,
                (
                    case_id,
                    tenant_id,
                    request_id,
                    reason[:1000],
                    ReviewStatus.OPEN.value,
                    current,
                    current,
                ),
            ).fetchone()
            if row is None:
                row = connection.execute(
                    "SELECT * FROM review_cases WHERE tenant_id = %s AND request_id = %s",
                    (tenant_id, request_id),
                ).fetchone()
                if row is None:
                    raise RuntimeError("concurrent review creation did not produce a row")
                return self._review_from_row(row)
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
            return self._review_from_row(row)

    def list_review_cases(
        self,
        tenant_id: str | None,
        *,
        status: ReviewStatus | None = None,
        after_sequence: int = 0,
        limit: int = 50,
    ) -> list[ReviewCase]:
        clauses = ["sequence > %s"]
        values: list[Any] = [after_sequence]
        if tenant_id is not None:
            clauses.append("tenant_id = %s")
            values.append(tenant_id)
        if status is not None:
            clauses.append("status = %s")
            values.append(status.value)
        values.append(min(200, max(1, limit)))
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM review_cases WHERE {' AND '.join(clauses)} "
                "ORDER BY sequence LIMIT %s",
                values,
            ).fetchall()
        return [self._review_from_row(row) for row in rows]

    def claim_review_case(
        self, case_id: str, tenant_id: str | None, actor: str
    ) -> ReviewCase | None:
        clauses = ["case_id = %s"]
        values: list[Any] = [case_id]
        if tenant_id is not None:
            clauses.append("tenant_id = %s")
            values.append(tenant_id)
        with self._connection() as connection:
            row = connection.execute(
                f"SELECT * FROM review_cases WHERE {' AND '.join(clauses)} FOR UPDATE",
                values,
            ).fetchone()
            if row is None:
                return None
            if row["status"] == ReviewStatus.CLAIMED.value and row["assignee"] == actor:
                return self._review_from_row(row)
            if row["status"] != ReviewStatus.OPEN.value:
                return None
            current = self._current_epoch(connection, None)
            updated = connection.execute(
                "UPDATE review_cases SET status = %s, assignee = %s, "
                "version = version + 1, updated_at = %s WHERE case_id = %s RETURNING *",
                (ReviewStatus.CLAIMED.value, actor, current, case_id),
            ).fetchone()
            if updated is None:
                raise RuntimeError("claimed review row disappeared")
            self._append_audit_locked(
                connection,
                tenant_id=str(row["tenant_id"]),
                actor=actor,
                action="review.claimed",
                target_type="review_case",
                target_id=case_id,
                now=current,
            )
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
        clauses = ["case_id = %s"]
        values: list[Any] = [case_id]
        if tenant_id is not None:
            clauses.append("tenant_id = %s")
            values.append(tenant_id)
        with self._connection() as connection:
            row = connection.execute(
                f"SELECT * FROM review_cases WHERE {' AND '.join(clauses)} FOR UPDATE",
                values,
            ).fetchone()
            if row is None or row["status"] not in {
                ReviewStatus.OPEN.value,
                ReviewStatus.CLAIMED.value,
            }:
                return None
            if expected_version is not None and int(row["version"]) != expected_version:
                return None
            current = self._current_epoch(connection, None)
            status = (
                ReviewStatus.DISMISSED
                if outcome == ReviewOutcome.NOT_ACTIONABLE
                else ReviewStatus.RESOLVED
            )
            updated = connection.execute(
                "UPDATE review_cases SET status = %s, assignee = COALESCE(assignee, %s), "
                "outcome = %s, correct = %s, version = version + 1, updated_at = %s "
                "WHERE case_id = %s RETURNING *",
                (status.value, actor, outcome.value, correct, current, case_id),
            ).fetchone()
            if updated is None:
                raise RuntimeError("resolved review row disappeared")
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
                    "INSERT INTO feedback(tenant_id, request_id, correct, created_at) "
                    "VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (tenant_id, request_id) DO NOTHING",
                    (str(row["tenant_id"]), str(row["request_id"]), correct, current),
                )
            return self._review_from_row(updated)

    def increment_metric(self, name: str, value: float = 1.0) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO metrics(name, value) VALUES (%s, %s) "
                "ON CONFLICT (name) DO UPDATE SET value = metrics.value + EXCLUDED.value",
                (name, value),
            )

    def metric_snapshot(self) -> dict[str, float]:
        with self._connection() as connection:
            rows = connection.execute("SELECT name, value FROM metrics ORDER BY name").fetchall()
        return {str(row["name"]): float(row["value"]) for row in rows}

    def list_audit_events(self, after_sequence: int = 0, limit: int = 100) -> list[AuditEvent]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events WHERE sequence > %s ORDER BY sequence LIMIT %s",
                (after_sequence, min(500, max(1, limit))),
            ).fetchall()
        return [self._audit_from_row(row) for row in rows]

    @staticmethod
    def _audit_from_row(row: Mapping[str, Any]) -> AuditEvent:
        return AuditEvent(
            sequence=int(row["sequence"]),
            event_id=str(row["event_id"]),
            occurred_at=_iso_time(float(row["occurred_at"])),
            tenant_id=str(row["tenant_id"]),
            actor=str(row["actor"]),
            action=str(row["action"]),
            target_type=str(row["target_type"]),
            target_id=str(row["target_id"]),
            details=_json_object(row["details_json"]),
            previous_hash=str(row["previous_hash"]),
            event_hash=str(row["event_hash"]),
        )

    def labeled_observations(self, limit: int | None = None) -> list[LabeledObservation]:
        query = (
            "SELECT p.*, f.correct, f.created_at AS labeled_at FROM predictions p "
            "JOIN feedback f ON f.tenant_id = p.tenant_id AND f.request_id = p.request_id "
            "WHERE p.backend_confidence_raw IS NOT NULL ORDER BY f.created_at, p.request_id"
        )
        parameters: tuple[int, ...] = ()
        if limit is not None:
            query += " LIMIT %s"
            parameters = (max(1, limit),)
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
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
                features={
                    name: float(value) for name, value in _json_object(row["features_json"]).items()
                },
                predicted_at_epoch=float(row["created_at"]),
                labeled_at_epoch=float(row["labeled_at"]),
            )
            for row in rows
        ]

    def verify_audit_chain(self) -> dict[str, Any]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM audit_events ORDER BY sequence").fetchall()
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
            details = _json_object(row["details_json"])
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
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT features_json FROM predictions ORDER BY created_at DESC LIMIT %s",
                (max(1, limit),),
            ).fetchall()
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for row in rows:
            for name, value in _json_object(row["features_json"]).items():
                if not isinstance(value, (bool, int, float)):
                    continue
                totals[name] = totals.get(name, 0.0) + float(value)
                counts[name] = counts.get(name, 0) + 1
        return {
            "sample_count": len(rows),
            "means": {name: totals[name] / counts[name] for name in sorted(totals) if counts[name]},
        }
