# Evaluation definitions

Metrics operate on actual prediction artifacts. A benchmark record explicitly chooses its evaluator.

## Answer quality

- **Normalized exact match:** lowercase, trim, replace punctuation with spaces, collapse whitespace, then compare equality.
- **Token F1:** multiset token overlap; harmonic mean of token precision and recall. Two empty answers score 1; one empty side scores 0.
- **Numeric correctness:** prefer boxed/final-answer/“answer is”/“therefore” spans, then the last signed decimal/scientific number; fractions and percentages are normalized before absolute/relative tolerance comparison.
- **Classification accuracy:** normalized answer or first normalized token equals the normalized label.
- **Keyword coverage:** fraction of configured normalized keyword phrases found on normalized token boundaries.
- **Abstention correctness:** whether an answer beginning with `ABSTAIN` (or an explicit cannot-answer phrase) matches `must_abstain`.
- **Task accuracy:** fraction of examples whose deterministic quality score is exactly 1.

Open-ended semantic similarity is not used as the only correctness measure. A future judge integration must remain optional and preserve these deterministic metrics.

## Routing and escalation

- **Route distribution:** request share assigned to each route.
- **Routing accuracy / small-success prediction accuracy:** fraction whose predicted small-model-success label equals the benchmark-derived label.
- **Escalation precision:** necessary escalations divided by all escalations. Undefined when there are none.
- **Escalation recall:** necessary escalations caught divided by all necessary escalations. Undefined when none are necessary.
- **False escalation rate:** unnecessary escalations divided by true small successes.
- **False non-escalation rate:** missed necessary escalations divided by true small failures.
- **Coverage:** non-abstained predictions divided by all predictions.
- **Selective accuracy:** mean quality among covered predictions; undefined at zero coverage.
- **Human-review count:** requests explicitly withheld for an external review workflow. It is reported separately from ordinary abstention even though both reduce automated coverage.
- **Estimated cost units:** sum of policy-selected configured token-cost estimates. This is not an observed bill or energy measurement.

The learned label is `small_model_quality >= configured_quality_threshold`. Training prefers `always_small` artifact rows so route outcomes do not contaminate the label. `group_id` keeps repeated or related examples together. With sufficient groups, training uses disjoint train, calibration, and test partitions: the model fits on train, temperature and serving threshold fit on calibration, and metrics are reported on untouched test. Fake artifacts may train only a clearly identified fake demonstration router.

## Calibration

- **Brier score:** mean squared difference between confidence and binary correctness; lower is better.
- **Expected calibration error:** weighted absolute gap between mean confidence and empirical accuracy across equal-width bins.
- **Reliability data:** bin boundaries, count, mean confidence, and empirical accuracy, including empty bins.

Transformers backends emit length-normalized token likelihood, sequence log-probability, minimum token log-probability, mean entropy, and vocabulary-normalized entropy. These are model uncertainty signals, not correctness probabilities. Raw confidence remains in artifacts. Brier/ECE correctness labels use the configured benchmark quality threshold. A scalar temperature calibrator may transform small-model confidence for cascade decisions; the calibrated value is stored separately.

Constant predictions, one-class targets, empty inputs, and zero denominators return documented `null`/empty values rather than crashing. On tiny datasets with fewer than six groups or nine samples, training labels its unavoidable in-sample fallback explicitly.

## Sampling uncertainty

Configured deterministic percentile bootstrap resampling operates over independent `group_id` units when available and records confidence intervals for supported aggregate metrics. The benchmark also emits a warning when its independent sample count is below `minimum_samples_for_claims`. Intervals describe the observed sampling process; they do not repair biased or unrepresentative data.

## Systems metrics

- p50 and p95 total request latency; p99 only with at least 100 samples.
- scheduler queue, routing, retrieval, generation, escalation, total, and TTFT where supported.
- actual batch size, configured concurrent batch workers, queue depth/rejections/deadlines for service runs, and backend inflight utilization.
- throughput from actual measured wall time rather than summed per-request latency; batching/concurrency configuration must match before comparison.
- input/output tokens and generated tokens per second.
- process RSS and optional peak CUDA allocated memory.
- failures, escalation count, abstentions, backend and route shares.

Cache replay cannot reproduce queueing or contention and is labeled accordingly. Fake timing validates instrumentation only. Percentile warnings appear below 20 observations.

## Distribution shift, feedback, and adaptation

The in-process monitor combines mean and q10/q50/q90 numeric shifts in baseline-standard-deviation units with category-distribution total variation. It freezes an automatic baseline at `minimum_samples`, or loads a supplied baseline. `drift_detected` compares the combined score with a configured threshold. This is an operational warning, not a statistical guarantee or automatic retraining trigger.

The operational store retains a prediction's tenant/request ID, route, confidence, numeric features, and timestamp, then joins idempotent correctness feedback. Notes are not persisted. Operator-invoked adaptation orders labels by time, trains on the earlier segment, tests the candidate on a held-out tail against the active calibrator, applies Brier/ECE gates, records Page-Hinkley error change points, and supports explicit rollback. Promotion is not proof against future shift.
