# Reproducibility

Each run directory is named with a UTC timestamp and workload slug, with a numeric suffix on collision. It contains:

```text
config.resolved.yaml  exact effective configuration
environment.json     Python, OS, CPU/memory, optional dependencies, PyTorch/CUDA/GPU
run.json             identity, hashes, Git, seed, workload, backend metadata, status
metrics.json          aggregate and per-policy metrics
predictions.jsonl     answer, reference, evaluator, quality, route, usage, error
routes.jsonl          policy, route, confidence, feature snapshot, escalation
timings.jsonl         separated request timing fields
errors.jsonl          failed request records
report/               generated Markdown, CSV, and figures
```

Configuration and dataset SHA-256 hashes identify inputs. Git commit and dirty state are recorded when a commit exists; an uncommitted new repository legitimately records `null`. The backend metadata records fake/real type, device request/resolution, precision, model/tokenizer IDs, configured artificial delay, initialization time, warm-up time, and optimization request/success as applicable.

Atomic temporary-file replacement reduces partial final JSON/YAML/JSONL files. A run status reports completion with or without request errors. Reports read the artifact directory and never fill missing metrics with invented values.

For a reproducible real comparison, save immutable model revisions and dataset versions in configuration/metadata, retain environment lock information, use consistent hardware state, and publish artifacts separately from the source repository.
