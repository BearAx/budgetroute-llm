# ADR 0006: SQLite operational control plane

- Status: accepted
- Date: 2026-08-07

## Context

The original API queue, rate limiter, metrics, feedback, and review outcome existed only in one process. Phase 6 needs meaningful coordination and durable workflow behavior without requiring a hosted database for the offline portfolio project. Prompt persistence would also create unnecessary privacy and replay semantics.

## Decision

Define an `OperationalStore` boundary and ship a transactional SQLite implementation using WAL on a reliable single host. The store owns fixed-window tenant quotas, expiring global admission leases, aggregate counters, privacy-minimal predictions, idempotent delayed labels, versioned review cases, and hash-chained audit events. Credentials remain environment-only and prompts/answers are not stored. Local batching stays per process; global leases bound total admitted generation but do not create a durable request queue.

## Consequences

Multiple workers sharing one database coordinate real state and recover abandoned capacity through lease expiry. Tests can exercise the same SQL semantics in memory or temporary files. SQLite is not presented as multi-host consensus or a network-filesystem solution; deployments needing that topology must implement the protocol with PostgreSQL/Redis while preserving transactional and idempotency guarantees. Audit hashes detect retained-row edits but do not defeat full database replacement by an administrator.
