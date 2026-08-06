# Architecture

## Boundaries and dependency direction

`schemas.py` and `config.py` define stable typed boundaries. Domain packages depend inward on these types; composition in `inference/service.py` selects concrete implementations. Backends do not import routing, routing does not call generation, evaluators do not own services, and report code reads artifacts rather than rerunning experiments.

The main protocols are `GenerationBackend`, `EmbeddingProvider`, `Retriever`, and `RoutingPolicy`. This makes fake and optional real implementations interchangeable without global registries or import-time model loading.

## Request lifecycle

1. Pydantic validates prompt length, fields, and enums.
2. Optional pre-routing retrieval runs for retrieval-required or retrieval-first requests.
3. `RequestFeatureExtractor` produces deterministic, interpretable values.
4. The selected policy returns a `RouteDecision` with confidence, reason, feature snapshot, and thresholds.
5. `InferenceEngine` invokes small, retrieval-assisted small, large, cascade, or abstention behavior.
6. Cascade retains the small answer only in internal traces when configured and records escalation tokens/latency separately.
7. The engine returns a `GenerationResponse` with usage, timing, memory, retrieval, and execution traces.
8. The evaluation harness applies the record's deterministic evaluator and emits prediction/route/timing rows.

## Backend lifecycle

Backends implement initialize, health, token count, single/batch generate, metadata, and cleanup. Fake backends initialize immediately. Transformers imports, tokenizer/model loading, device checks, quantization, and compilation happen only in `initialize`; importing `budgetroute` never downloads a model.

The FastAPI lifespan initializes configured services and cleans them up. `/healthz` itself performs no initialization or model probe. `/readyz` reports already-held service state.

## Retrieval

Document loading and deterministic chunking produce typed chunks. An embedder maps text to normalized vectors. `ExactCosineIndex` owns portable NumPy persistence/search, and `RetrievalService` owns timing and score filtering. See `retrieval.md`.

## Artifacts

The experiment runner is the only component that creates a run directory. It writes final files atomically where practical. Reports consume `run.json`, `metrics.json`, and JSONL rows inside that same directory; missing values produce an honest empty state. Ordinary outputs are ignored by Git.

## Concurrency

The synchronous engine is safe when dependencies are safe for concurrent calls. `AsyncMicrobatcher` owns an asyncio queue, bounded wait, size-based flush, exception propagation, result-order mapping, and sentinel-based shutdown. Distributed scheduling is intentionally outside v0.1.

