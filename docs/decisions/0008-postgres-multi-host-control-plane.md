# ADR 0008: PostgreSQL multi-host operational control plane

- Status: accepted
- Date: 2026-08-28

## Context

ADR 0006 deliberately chose SQLite for an offline-capable, same-host control plane. That backend proves transactional quotas, leases, delayed labels, reviews, metrics, and audit behavior without infrastructure, but it cannot safely coordinate replicas on independent filesystems or nodes. A resume-scale system needs a real multi-host path without coupling inference code to database calls or pretending that Kubernetes alone supplies consistency.

## Decision

Implement the complete `OperationalStore` protocol with PostgreSQL and Psycopg 3 pooling. Read the DSN only from a named environment variable and require an explicit encrypted `sslmode` in production. Use database time for quota windows and lease expiry. Use atomic upserts and uniqueness constraints for quotas/idempotency, row locks plus versions for review transitions, and transaction-scoped advisory locks for the global migration, admission, and audit-head critical sections. Never hold a database transaction while model inference runs.

Ship ordered SQL migrations as package resources. Serialize their application, record a SHA-256 checksum, and fail startup for missing, modified, or unknown migrations. Allow automatic migration for local development, while the production profile runs an explicit DDL-capable migration job and gives API replicas a narrower runtime credential. Readiness must query both inference and operational storage and return 503 when either is unavailable.

## Consequences

Independent API hosts can share one transactional operational state without silently reverting quotas to process-local memory. Real PostgreSQL CI verifies parity and high-contention quota, lease, migration, feedback, and audit behavior. Bounded pools and short transactions provide an inspectable capacity model.

PostgreSQL remains infrastructure with failure modes: operators must provision HA, replication/failover, backups/PITR, TLS trust, credential rotation, monitoring, network policy, and recovery drills. Advisory locks serialize a few global decisions and can become contention points that require target-load evidence. The adapter targets one PostgreSQL primary/consistency domain, not multi-region active/active consensus. Forward-only migrations require expand/contract compatibility; older binaries reject unknown schema versions by design.
