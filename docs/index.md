# Documentation

- [Architecture](architecture.md): data/control planes, batch/global admission, durable operations, adaptation, artifacts.
- [Evaluation](evaluation.md): deterministic quality, routing, selective, and calibration metrics.
- [Benchmarking](benchmarking.md): timing, warm-up, CUDA, percentiles, memory, and interpretation.
- [Real evaluation workflow](real-evaluation-workflow.md): pinned datasets/models, cache collection, replay, calibration, and training.
- [Routing](routing.md): every policy, features, thresholds, labels, and failure modes.
- [Retrieval](retrieval.md): pinned embeddings, exact/FAISS/HNSW search, persistence, benefit learning.
- [Configuration](configuration.md): YAML composition, validation, environment overrides.
- [Development](development.md): setup, tests, style, packaging, contribution workflow.
- [Deployment](deployment.md): tenant/TLS setup, same-host replicas, review/audit/adaptation, load, Docker, GitHub controls.
- [Security policy](../SECURITY.md): private disclosure, supported versions, automated controls, limitations.
- [Reproducibility](reproducibility.md): run identity, hashes, metadata, artifact contract.
- [Published held-out learned-router benchmark](../reports/benchmarks/qwen25-mmlu-500-learned-rtx3060.md): paired Qwen2.5 outcomes, disjoint training/calibration/test partitions, RTX 3060 Laptop GPU, negative result and limits.
- [Initial real benchmark](../reports/benchmarks/qwen25-mmlu-100-rtx3060.md): Qwen2.5, stratified MMLU-100, RTX 3060 Laptop GPU, results and limits.
- [Limitations](limitations.md): implemented mitigations versus infrastructure, evidence, and model-safety boundaries.
- [Portfolio notes](portfolio-notes.md): descriptions, resume placeholders, demo, interviews.
- [Decisions](decisions/index.md): concise architecture decision records.
