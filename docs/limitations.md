# Limitations

## Data and scientific scope

The original 12-record sample exists to cover code paths. It is too small, curated, and unrepresentative for scientific conclusions, model rankings, or production thresholds. The fake backend can consume reference answers to simulate controlled success/failure and therefore has no external validity.

## Models and hardware

Real quality depends on model weights, licenses, revisions, tokenizer/chat templates, generation settings, and prompt construction. Real systems results depend on CPU/GPU, VRAM, drivers, PyTorch/CUDA versions, thermal state, compilation, quantization, batching, and cache state. Example model IDs were not benchmarked during repository creation.

## Confidence and routing

Fake confidence is configured behavior. The Transformers backend currently exposes a neutral placeholder confidence rather than claiming calibrated token likelihood. Cascade and learned routing therefore require task-specific real calibration before meaningful use. Heuristic categories and thresholds are intentionally simple and English-centric.

## Evaluation

Deterministic metrics are reproducible but incomplete for open-ended correctness, factuality, style, safety, and usefulness. Numeric evaluation reads the first number. Keyword coverage can reward shallow mention. No default LLM judge is used, avoiding cost/nondeterminism but limiting semantic assessment.

## Systems

The initial harness is sequential despite recording configured concurrency; throughput is derived from summed per-request time. The microbatcher is tested locally, but the Transformers batch method uses stable per-request generation rather than optimized padded tensor batching. p95/p99 estimates require much larger runs.

## Retrieval and service

Fake lexical hashing is not a semantic embedder. Exact search scales linearly. The API lacks production authentication, rate limits, persistent scheduling, and distributed coordination. These are explicit roadmap items rather than hidden claims.

