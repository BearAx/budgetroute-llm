# Evaluation definitions

Metrics operate on actual prediction artifacts. A benchmark record explicitly chooses its evaluator.

## Answer quality

- **Normalized exact match:** lowercase, trim, replace punctuation with spaces, collapse whitespace, then compare equality.
- **Token F1:** multiset token overlap; harmonic mean of token precision and recall. Two empty answers score 1; one empty side scores 0.
- **Numeric correctness:** extract the first signed decimal/scientific number and compare with configurable absolute and relative tolerance.
- **Classification accuracy:** normalized answer or first normalized token equals the normalized label.
- **Keyword coverage:** fraction of configured normalized keywords found in the answer.
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

The initial learned label is `small_model_quality >= configured_quality_threshold`. Training prefers `always_small` artifact rows so route outcomes do not contaminate the label. Fake artifacts may train only a clearly identified fake demonstration router.

## Calibration

- **Brier score:** mean squared difference between confidence and binary correctness; lower is better.
- **Expected calibration error:** weighted absolute gap between mean confidence and empirical accuracy across equal-width bins.
- **Reliability data:** bin boundaries, count, mean confidence, and empirical accuracy, including empty bins.

Constant predictions, one-class targets, empty inputs, and zero denominators return documented `null`/empty values rather than crashing. On tiny router datasets, training can use a constant classifier and labels the evaluation as in-sample.

## Systems metrics

- p50 and p95 total request latency; p99 only with at least 100 samples.
- routing, retrieval, generation, escalation, total, and TTFT where supported.
- throughput as successful sequential requests divided by summed request time in the initial harness.
- input/output tokens and generated tokens per second.
- process RSS and optional peak CUDA allocated memory.
- failures, escalation count, abstentions, backend and route shares.

Sequential-throughput values are not concurrency capacity measurements. Percentile warnings appear below 20 observations.

