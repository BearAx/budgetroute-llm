# ADR 0004: Use a bounded full-pipeline batch scheduler

## Decision

Batch complete inference requests at the API boundary with a bounded queue, first-item flush deadline, admission timeout, request deadline, ordered result mapping, and draining shutdown. Execute each batch in small and large backend waves so cascade decisions remain semantically correct. Backends retain their direct batch protocol and concurrency slots.

## Consequences

Batching reaches model calls and is observable through queue/batch metrics. Overload is rejected instead of consuming unbounded memory. Per-process state remains simple and testable, but replicas need shared admission control and telemetry. Load-aware routing is schedule-dependent, so artifacts must retain observed snapshots.
