# Configuration

`pyproject.toml` is authoritative for packaging and tool settings. Runtime behavior uses typed Pydantic models loaded from YAML.

## Composition

A file may contain `extends` as one path or an ordered path list. Paths resolve relative to the including file first, then the current working directory. Included mappings are deep-merged left to right; the current file wins. Cycles, missing files, non-mapping roots, and invalid types fail with actionable errors.

Model identifiers live only in `configs/models/`. Routing fragments live in `configs/routing/`. Benchmark and serving files compose base/model values and add workload-specific settings.

Public dataset specifications live in `configs/datasets/` and pin source, configuration, immutable revision, split, license, homepage, adapter, and prompt version. Model profiles pin model/tokenizer revisions, license identifiers, model-card URLs, precision, and deterministic generation seeds.

## Validation

Validation rejects missing Transformers/OpenAI-compatible model IDs, malformed or credential-bearing runtime URLs, fake quantization, CPU FP16/quantization, invalid retrieval chunk overlap, missing real embedding IDs, unusable routing weights/probabilities, learned routing without an artifact path, retrieval-first without retrieval, fake mode with real backends, and non-loopback API binding without authentication.

Runtime initialization separately rejects explicit CUDA or BF16 requests unsupported by the installed PyTorch/hardware and missing quantization dependencies. Explicit GPU selection never silently falls back. `device: auto` may select CPU or CUDA and records the result.

OpenAI-compatible backends require `base_url` ending at the API root (normally `/v1`). Loopback HTTP is permitted for local runtimes. Non-loopback endpoints require both `allow_remote_endpoint: true` and HTTPS. `api_key_env` names the environment variable; secret values never enter resolved configuration or metadata. `max_concurrency` bounds concurrent backend calls. `request_logprobs` is opt-in because some compatible servers reject that extension. Optional input/output cost-unit rates are estimates used by `budget_aware`, not billing observations.

Benchmark cache mode is one of `off`, `read_write`, `read_only`, or `refresh`. `read_only` requires an existing cache and never initializes generation models. `refresh` always regenerates and replaces the keyed entry. Real benchmark profiles set `require_pinned_revisions: true`, which rejects unpinned Transformers backends. `routing.learned_success_threshold` can override an artifact threshold; leaving it unset uses the calibration-selected threshold. `routing.cascade_calibrator_path` attaches a portable small-backend confidence calibrator and its selected cascade threshold.

`batching` controls API queue enablement, maximum batch/queue sizes, first-item wait, admission timeout, and end-to-end request deadline. Benchmark `batch_size` controls ordered backend grouping and `concurrency` controls concurrent batch workers. `routing.load_aware` uses trusted backend load snapshots. `routing.budget_aware` uses target latency, optional maximum estimated cost units, and quality/latency/cost weights. Human review is opt-in and thresholded.

`monitoring` controls the numeric rolling window, minimum baseline sample count, drift threshold, and optional JSON baseline path. The automatic baseline freezes when the minimum sample count is first reached. For controlled experiments, provide a representative versioned baseline instead.

## Environment overrides

Operational values can be overridden with:

```text
BUDGETROUTE_OUTPUT_DIR
BUDGETROUTE_HOST
BUDGETROUTE_PORT
BUDGETROUTE_LOG_LEVEL
BUDGETROUTE_API_KEY
```

The first four values are direct configuration overrides. `BUDGETROUTE_API_KEY` is read only when the selected API configuration names it as the credential source. The API configuration response is sanitized and never includes secret values, backend failure markers, or arbitrary environment variables. The project does not load `.env` implicitly; a shell, container runtime, or orchestrator must provide secrets.
