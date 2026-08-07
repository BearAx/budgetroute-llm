# Changelog

All notable changes follow Keep a Changelog conventions. The project uses semantic versioning.

## [Unreleased]

No unreleased changes yet.

## [0.6.0] - 2026-08-07

### Added

- Transactional SQLite coordination for replica-shared quotas, global admission leases, aggregate metrics, delayed labels, durable human-review cases, and hash-chained audit events.
- Environment-backed multi-tenant credentials with scoped inference, feedback, review, and administrative access; tenant data remains isolated at API queries.
- Chronological delayed-label confidence adaptation with Page-Hinkley change points, immutable candidates, explicit promotion gates, atomic activation, and rollback.
- Learned retrieval-benefit routing trained from paired baseline/retrieval evidence with group-safe partitions and calibrated thresholds.
- Revision-pinned Transformers embeddings, optional FAISS HNSW search, runtime compatibility probes, and multi-endpoint HTTP load artifacts.
- Bootstrap uncertainty intervals, minimum-sample claim warnings, improved numeric/fraction/percentage extraction, phrase-safe keyword evaluation, and measured wall-clock throughput.
- Configurable prompt/output substring policy and bounded metadata validation, built-in TLS options, explicit external TLS termination, audit verification, and a distributed serving profile.

### Changed

- Runtime drift combines numeric mean/quantile shifts with category-distribution total variation.
- Security validation now fails non-loopback configurations without scoped authentication and an explicit TLS boundary.
- Documentation distinguishes mitigated repository limitations from deployment-specific evidence and infrastructure requirements.

## [0.4.0] - 2026-08-07

### Added

- Bounded deadline-aware inference scheduler with queue telemetry, overload rejection, and orderly shutdown.
- Ordered small/large backend-wave batch execution and padded Transformers generation.
- OpenAI-compatible local-runtime backend with safe endpoint and credential configuration.
- Thread-safe backend load telemetry plus load-aware and quality/cost/latency-aware policies.
- Explicit human-review outcomes, configured cost estimates, Prometheus metrics, aggregate feedback, and rolling feature-shift detection.
- API-key authentication, trusted Host validation, body/prompt/rate limits, secure response headers, and a deployment security audit command.
- Dependabot configuration, CodeQL extended scanning, dependency review, `pip-audit`, and private vulnerability reporting guidance.

### Changed

- Fake smoke benchmarking now exercises ordered backend batching and the adaptive policies without presenting fake measurements as performance evidence.
- Reports include human-review counts and configured cost units.
- GitHub Actions use current Node 24-compatible checkout/setup actions.

## [0.2.0] - 2026-08-06

### Added

- Revision-pinned GSM8K, MMLU, and HotpotQA adapters with deterministic JSONL/corpus materialization and integrity manifests.
- Revision/license-pinned SmolLM2 CPU and Qwen2.5 GPU model profiles.
- Content-addressed generation collection, model-free replay, cache integrity checks, and live/replay agreement verification.
- Model-derived token likelihood/log-probability/entropy signals with separate raw and calibrated confidence.
- Group-aware train/calibration/test router partitions, scalar temperature calibration, calibration-only threshold selection, and confidence-calibrated cascade support.
- CLI workflows for dataset materialization, baseline collection, replay, replay verification, and backend-confidence calibration.

### Changed

- Learned-router evaluation now defaults to the persisted untouched test partition.
- Real reports explicitly distinguish live execution from cache replay.

## [0.1.0] - 2026-08-06

### Added

- Initial quality-aware routing, retrieval, inference, evaluation, reporting, CLI, API, and offline fake-mode implementation.
