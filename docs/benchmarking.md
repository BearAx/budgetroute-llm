# Benchmarking protocol

## Lifecycle and timing

Model download and loading occur before measured requests. Service initialization time is recorded per policy. Configured warm-up requests run after initialization, their aggregate duration is recorded separately, and they are excluded from predictions and steady-state metrics. Each timed region uses `time.perf_counter_ns`, a monotonic high-resolution clock.

The trace separates routing, retrieval, generation, escalation, and total request time. Transformers generation synchronizes CUDA immediately before and after the measured region. CUDA peak allocation counters are reset before generation and read afterward when available. `psutil` records process RSS.

Compilation warm-up belongs outside measured steady-state requests; metadata records whether compile or quantization was requested and actually enabled. Never claim an optimization speed-up without directly comparable artifacts.

## Sample size and percentiles

p50 and p95 are reported for non-empty samples, accompanied by a warning below 20 values. p99 is omitted below 100 values. Tail values from tiny datasets are illustrative and should not be compared as stable estimates.

Failed requests are recorded in `errors.jsonl` and excluded from successful latency summaries, while failure counts remain visible. Cold-start and warm-start values must be labeled separately; v0.1 reports warm steady-state request values.

## Comparison discipline

Compare policies only when model identifiers, revisions, prompt templates, generation parameters, data, warm-up, measured count, batch size, concurrency, precision, quantization, compiler state, hardware, drivers, and software environment are compatible. Inspect resolved configuration and environment metadata before interpreting differences.

Fake artificial delay verifies instrumentation only. It is not a proxy for real hardware or model behavior.
