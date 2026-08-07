# Limitations

## Data and scientific scope

The original 12-record sample exists to cover code paths. It is too small, curated, and unrepresentative for scientific conclusions, model rankings, or production thresholds. The fake backend can consume reference answers to simulate controlled success/failure and therefore has no external validity. The pinned public adapters improve provenance but do not make a tiny `--limit` smoke selection representative.

## Models and hardware

Real quality depends on model weights, licenses, revisions, tokenizer/chat templates, generation settings, and prompt construction. Real systems results depend on CPU/GPU, VRAM, drivers, PyTorch/CUDA versions, thermal state, compilation, quantization, batching, and cache state. A two-record CPU integration run validates the pinned SmolLM2 path but is not a comparative benchmark. The configured Qwen CUDA path remains unmeasured in this repository.

## Confidence and routing

Fake confidence is configured behavior. Transformers confidence is a genuine length-normalized token-likelihood signal with entropy diagnostics, but it is not correctness probability. Scalar calibration and learned thresholds remain specific to the chosen model, prompt template, dataset, and split. Distribution shift can invalidate them. The rolling mean-shift monitor detects only coarse numeric feature changes; it does not prove semantic drift, label drift, or safety. Heuristic/adaptive quality estimates and categories are intentionally simple and English-centric.

## Evaluation

Deterministic metrics are reproducible but incomplete for open-ended correctness, factuality, style, safety, and usefulness. Numeric evaluation reads the first number. Keyword coverage can reward shallow mention. No default LLM judge is used, avoiding cost/nondeterminism but limiting semantic assessment.

## Systems

The harness executes configured batch size and concurrent batch workers, and Transformers uses padded batch generation. These features do not imply a speedup: padding waste, output-length skew, GPU saturation, compilation, quantization, thermal behavior, local-server scheduling, and memory pressure can reverse results. Cache replay preserves recorded generation cost but cannot reproduce contention or queueing. p95/p99 and overload thresholds require much larger live runs.

## Retrieval and service

Fake lexical hashing is not a semantic embedder. Exact search scales linearly. Similarity and margin are retrieval signals, not calibrated evidence that context will improve an answer.

API-key authentication, trusted hosts, limits, queueing, metrics, drift state, and feedback counts are per process. They do not provide user authorization, shared multi-replica quotas, durable audit records, tenant isolation, TLS, prompt-injection defense, content policy enforcement, persistent scheduling, or distributed coordination. Human review is only an explicit response state and has no built-in case queue. Use a gateway, secret manager, durable telemetry/review systems, and a deployment-specific threat model.

The OpenAI-compatible adapter targets the common chat-completions subset. Servers may ignore seeds/log probabilities, tokenize differently, reject optional fields, or expose vendor-specific batching semantics. Remote endpoints are disabled by default; enabling one changes the privacy and trust boundary.
