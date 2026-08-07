# Roadmap

Completed milestones are marked explicitly; unmarked items remain proposals.

## v0.1 — functional benchmark and API (complete)

- [x] Stabilize the fake/Transformers benchmark path and artifact schema.
- [x] Validate the API and packaging across supported platforms.
- [x] Validate a small, explicitly non-comparative real-model CPU acceptance run separately from fake outputs.

## v0.2 — datasets, replay, and calibration (complete)

- [x] Add licensed, revision-pinned public benchmark adapters and integrity manifests.
- [x] Add content-addressed baseline collection and model-free replay with agreement verification.
- [x] Add group-aware train/calibration/test controls, calibrated classifiers, and calibration-only threshold selection.
- [x] Preserve raw and calibrated confidence plus selective-routing metrics.

## v0.3 — scheduling and backends (complete)

- [x] Add bounded deadline-aware API batching and ordered batch benchmark execution.
- [x] Add true padded Transformers generation while preserving heterogeneous token limits.
- [x] Add thread-safe load telemetry and load-aware routing.
- [x] Add an OpenAI-compatible local-runtime backend for vLLM, llama.cpp server, Ollama, and compatible servers.
- [x] Record requested/enabled compilation and quantization state plus first compiled execution time; keep performance claims gated on real artifacts.

## v0.4 — operational and research guardrails (complete)

- [x] Add configurable quality/cost/latency multi-objective routing and explicit human-review outcomes.
- [x] Add queue deadlines, overload rejection, API authentication, trusted hosts, request/rate limits, and defensive headers.
- [x] Add Prometheus metrics, privacy-preserving aggregate feedback, and rolling feature-shift detection.
- [x] Add Dependabot, CodeQL, dependency review/audit, and private vulnerability reporting guidance.

## Future research

- Distributed scheduling and shared quotas/telemetry across replicas.
- Durable human-review and outcome-feedback integrations with audited access controls.
- Online recalibration with delayed labels, rollback criteria, and change-point detection.
- Learned retrieval-benefit prediction and semantic retrieval backends.
- Large-scale live runtime/quantization/compilation compatibility matrices with published immutable artifacts.
- Multi-host tail-latency load tests, admission-control tuning, and autoscaling policies.
