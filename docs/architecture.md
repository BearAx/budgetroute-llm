# Architecture

## Boundaries and dependency direction

`schemas.py` and `config.py` define stable typed boundaries. Domain packages depend inward on these types; composition in `inference/service.py` selects concrete implementations. Backends do not import routing, routing does not call generation, evaluators do not own services, and report code reads artifacts rather than rerunning experiments.

The main protocols are `GenerationBackend`, `EmbeddingProvider`, `Retriever`, and `RoutingPolicy`. This makes fake and optional real implementations interchangeable without global registries or import-time model loading.

## Request lifecycle

1. Pydantic validates prompt length, fields, and enums.
2. Optional pre-routing retrieval runs for retrieval-required or retrieval-first requests.
3. `RequestFeatureExtractor` produces deterministic, interpretable values.
4. The selected policy returns a `RouteDecision` with confidence, reason, feature snapshot, and thresholds.
5. `InferenceEngine` invokes small, retrieval-assisted small, large, cascade, abstention, or explicit human-review behavior.
6. Batch execution preserves request order while grouping small work, evaluating cascade confidence, then grouping direct/escalated large work.
7. Cascade retains the small answer only in internal traces when configured and records escalation tokens/latency separately.
8. The engine returns a `GenerationResponse` with usage, queue/batch timing, memory, retrieval, cost estimate, and execution traces.
9. The evaluation harness applies the record's deterministic evaluator and emits prediction/route/timing rows.

## Backend lifecycle

Backends implement initialize, health, token count, single/batch generate, metadata, and cleanup. Fake backends initialize immediately. Transformers imports, tokenizer/model loading, device checks, quantization, and compilation happen only in `initialize`; importing `budgetroute` never downloads a model. Transformers batch generation uses left padding, one framework call per output-token-limit group, and restores original request order. The OpenAI-compatible adapter targets the portable chat-completions subset and supports loopback vLLM, llama.cpp server, Ollama, and compatible runtimes.

`TrackedBackend` is a transparent decorator with bounded concurrent invocation slots and thread-safe inflight, latency, completion, and error telemetry. Adaptive policies receive snapshots through a trusted callback; clients cannot inject load state through request metadata.

The FastAPI lifespan initializes configured services and cleans them up. `/healthz` itself performs no initialization or model probe. `/readyz` reports already-held service state.

## Retrieval

Document loading and deterministic chunking produce typed chunks. An embedder maps text to normalized vectors. `ExactCosineIndex` owns portable NumPy persistence/search, and `RetrievalService` owns timing and score filtering. See `retrieval.md`.

## Artifacts

The experiment runner is the only component that creates a run directory. It writes final files atomically where practical. Reports consume `run.json`, `metrics.json`, and JSONL rows inside that same directory; missing values produce an honest empty state. Ordinary outputs are ignored by Git.

## Generation cache and replay

`GenerationCache` stores typed backend generations under SHA-256 content keys. `CachedGenerationBackend` wraps any backend in live collection modes and becomes a model-free backend in read-only mode. Because it implements the same protocol, routing, inference, evaluation, and reporting do not contain cache-specific branches beyond replay timing labels. Live/replay verification compares stable semantic response signatures.

## Router training

Training joins always-small outcomes to prompt/retrieval feature snapshots. Group-aware train, calibration, and test partitions prevent related records from crossing boundaries. The base classifier fits only on train; scalar temperature and serving threshold fit only on calibration; final metrics use test. Persisted artifacts carry the feature order, classifier, calibrator, selected threshold, partitions, source hash, and package versions.

## Scheduling and concurrency

`AsyncInferenceBatcher` owns a capacity-bounded asyncio queue, one deadline measured from the first queued item, size/deadline flush, admission timeout, request deadline, batch result-order mapping, exception propagation, queue metrics, and sentinel-based draining shutdown. It submits the entire pipeline to `InferenceService.generate_batch`, so grouping reaches backend calls. Benchmark `batch_size` and `concurrency` are executed rather than metadata-only.

The scheduler is per process. Backend concurrency slots protect non-thread-safe local models, while OpenAI-compatible backends may use configured parallel HTTP calls. Distributed queues, shared quotas, and autoscaling remain external deployment concerns.

## Operations and security

FastAPI middleware enforces byte limits (including chunked bodies), trusted Host values, optional constant-time API-key checks, per-identity sliding-window limits, and defensive response headers. Non-loopback binding is invalid without authentication. The scheduler maps overload and deadline failures to retryable HTTP responses.

`RuntimeMonitor` retains numeric feature windows and aggregate counters only—not prompts, answers, feedback notes, or identifiers. It exports Prometheus text and a sanitized drift snapshot. Human review is a typed response state; an external durable workflow must consume it.
