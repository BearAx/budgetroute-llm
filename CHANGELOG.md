# Changelog

All notable changes follow Keep a Changelog conventions. The project uses semantic versioning.

## [Unreleased]

No unreleased changes yet.

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
