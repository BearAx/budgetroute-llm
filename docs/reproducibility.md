# Reproducibility

Each run directory is named with a UTC timestamp and workload slug, with a numeric suffix on collision. It contains:

```text
config.resolved.yaml  exact effective configuration
environment.json     Python, OS, CPU/memory, optional dependencies, PyTorch/CUDA/GPU
run.json             identity, hashes, Git, seed, workload, backend metadata, status
metrics.json          aggregate and per-policy metrics
predictions.jsonl     answer, reference, evaluator, quality, route, usage, error
routes.jsonl          policy, route, confidence, feature/load snapshot, estimates, escalation/review
timings.jsonl         queue/batch/routing/retrieval/generation/escalation/total timing
errors.jsonl          failed request records
report/               generated Markdown, CSV, and figures
```

Configuration and dataset SHA-256 hashes identify inputs. Git commit and dirty state are recorded when a commit exists. Backend metadata records fake/Transformers/OpenAI-compatible type, local endpoint (never its credential), device request/resolution, precision, model/tokenizer IDs and exact requested/resolved revisions, model license/card, generation settings, concurrency slots, configured cost rates, load telemetry, artificial delay, initialization/warm-up time, and optimization request/success/first compiled execution as applicable. When configured, `run.json` embeds the validated dataset manifest and cache/replay mode.

Generation cache entries contain their complete backend and request fingerprints plus a typed generation envelope. Entry paths derive from SHA-256 keys, and reads revalidate both key and backend fingerprint. Replay artifacts keep the original generation measurements and separately record replay overhead.

Atomic temporary-file replacement reduces partial final JSON/YAML/JSONL files. A run status reports completion with or without request errors. Reports read the artifact directory and never fill missing metrics with invented values.

Load-aware choices depend on the observed concurrent schedule, so exact route reproduction additionally requires the same batch/concurrency/load pattern; every observed snapshot is stored in `routes.jsonl`. Cost units are configuration inputs. Drift/feedback service state is intentionally not part of benchmark artifacts unless a deployment exports it separately.

For a reproducible real comparison, use committed revision-pinned profiles, retain environment lock information, use consistent hardware and scheduler state, run `verify-replay`, and publish immutable experiment artifacts separately from the source repository.
