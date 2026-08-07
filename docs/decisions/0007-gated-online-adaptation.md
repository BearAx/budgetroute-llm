# ADR 0007: Gated, operator-invoked adaptation

- Status: accepted
- Date: 2026-08-07

## Context

Model confidence is not a stable correctness probability. Automatically fitting and activating calibration on recent labels risks feedback loops, leakage, regressions, and irreproducible serving behavior.

## Decision

Adaptation runs outside the request path. It sorts delayed labels chronologically, trains on the earlier portion, evaluates a candidate on a held-out tail against the active calibrator, and applies explicit Brier-improvement and ECE-regression gates. Every candidate is an immutable, source-hashed version. Promotion atomically updates the active materialization and manifest; rollback is explicit. Page-Hinkley reports upward error change points but does not trigger promotion or rollback.

Retrieval-benefit learning remains a separate supervised artifact trained from paired baseline/retrieval outcomes because it estimates a different target from small-model success.

## Consequences

Changes are traceable, testable, and recoverable, while request latency and policy state remain deterministic within a deployed version. Recent labeled evidence, holdout size, and thresholds remain deployment-specific. A passed gate justifies a controlled rollout, not future correctness; operators still need monitoring, canarying, and rollback practice.
