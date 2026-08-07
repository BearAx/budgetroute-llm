# Roadmap

Completed milestones are marked explicitly. Completion means the repository capability and its offline validation exist; it does not imply universal model quality, hardware performance, or production certification.

## v0.1 — functional benchmark and API (complete)

- [x] Fake/Transformers benchmark path, typed artifacts, API, packaging, and a non-comparative CPU acceptance run.

## v0.2 — datasets, replay, and calibration (complete)

- [x] Licensed revision-pinned datasets, integrity manifests, content-addressed generation replay, group-safe router training, and confidence calibration.

## v0.3 — scheduling and backends (complete)

- [x] Bounded deadline-aware batching, padded Transformers batches, load-aware routing, and local OpenAI-compatible runtimes.

## v0.4 — operational and research guardrails (complete)

- [x] Multi-objective routing, overload handling, API security controls, Prometheus metrics, feature drift, and GitHub security automation.

## v0.5 — distributed operational control plane (Phase 6, complete)

- [x] Transactional SQLite quotas and expiring global admission leases shared by same-host replicas.
- [x] Environment-backed tenant credentials, scoped authorization, and tenant-isolated review/feedback queries.
- [x] Durable delayed labels, human-review cases, shared counters, and hash-chained audit events with verification.
- [x] Explicit built-in TLS or trusted external TLS termination for non-loopback serving.
- [x] Configurable prompt/output substring rules and bounded metadata validation without logging rejected content.

## v0.6 — adaptation and deployment validation (Phase 7, complete)

- [x] Chronological delayed-label recalibration, Page-Hinkley change points, gated promotion, immutable versions, and rollback.
- [x] Learned retrieval-benefit routing from paired outcomes with group-safe partitions and safe fallbacks.
- [x] Revision-pinned semantic embeddings and optional FAISS HNSW approximate search.
- [x] Reproducible runtime compatibility probes and multi-endpoint HTTP load artifacts with measured wall throughput.
- [x] Bootstrap uncertainty, claim-size warnings, stronger deterministic answer extraction, and richer feature/category drift.
- [x] A documented mitigation matrix for repository, deployment, evidence, and irreducible model-safety limitations.

## Next evidence milestones

- [x] Publish an initial license-compliant comparison on named models, data, hardware, runtime revisions, and a clean source commit: Qwen2.5-0.5B/1.5B on stratified MMLU-100 using an RTX 3060 Laptop GPU.
- Expand the initial study to larger held-out samples, repeated timing trials, trained/calibrated routing, and batch/concurrency matrices before making quality-retention or capacity claims.
- Implement a PostgreSQL or Redis `OperationalStore` for multi-host coordination; SQLite intentionally targets one shared host/filesystem.
- Integrate an external identity provider, secret manager, WAF/service mesh, centralized telemetry, and ticketing system for sensitive production use.
- Add domain-specific semantic/human evaluation and red-team evidence; no generic filter can prove factuality, usefulness, or prompt-injection safety.
- Validate HNSW recall/latency and autoscaling behavior on the target corpus and fleet before setting production thresholds.
