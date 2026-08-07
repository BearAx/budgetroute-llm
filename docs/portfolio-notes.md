# Portfolio notes

## Short description

BudgetRoute-LLM is a typed Python research system that dynamically batches and routes requests across small/large local language models, retrieval, cascade escalation, abstention, and human review while measuring deterministic quality, cost, and systems trade-offs.

## Extended description

The project separates model backends, scheduling/load telemetry, routing policies, retrieval, evaluation, profiling, monitoring, artifacts, and reporting behind testable interfaces. An offline deterministic mode exercises the same CLI, FastAPI, bounded batching, adaptive routing, API security, benchmark, learned-router, and report paths as optional Hugging Face CPU/CUDA or local OpenAI-compatible execution. Reproducibility metadata and explicit fake labeling prevent unmeasured portfolio claims.

## Resume bullets

- Built a quality-aware LLM routing system across small/large models, retrieval, cascade, and abstention; on **[real dataset/model/hardware]**, achieved **[measured quality]** at **[measured p50/p95 latency]** versus **[baseline]**.
- Engineered reproducible experiment artifacts, deterministic evaluation, calibration metrics, FastAPI serving, and offline CI with **[real test count/coverage at time of application]**.
- Implemented optional CPU/CUDA Transformers execution and portable cosine retrieval, reducing **[measured compute or latency]** by **[real measured percentage]** at a **[real configured quality target]**.
- Built a bounded deadline-aware scheduler with ordered cascade batching, load/cost-aware policies, Prometheus metrics, drift alerts, and secure overload handling; sustained **[measured throughput]** at **[measured p95]** on **[hardware/runtime]**.
- Added a safe OpenAI-compatible local-runtime boundary and GitHub security automation (CodeQL, dependency review, Dependabot, `pip-audit`), with **[measured security/operations evidence]** from the published workflow artifacts.

Replace every bracket only with traceable real artifacts. Do not use fake benchmark values.

## Two-minute demo

1. State the research question and show the architecture diagram (20 seconds).
2. Run `budgetroute demo` and point out easy, difficult, retrieval, cascade, and abstention traces (35 seconds).
3. Send concurrent API requests and show bounded batches, queue metrics, load-aware traces, and the prominent fake label (30 seconds).
4. Open the policy CSV/figures and explain that they validate reporting, not performance (20 seconds).
5. Show CPU/GPU YAML and explain how a real artifact would replace placeholders (15 seconds).

## Interview discussion points

- Why routing quality labels must come from actual small-model evaluation.
- Preventing feature leakage and training-serving skew.
- Tail latency, cold/warm separation, CUDA synchronization, and batching trade-offs.
- Calibration versus raw generation confidence.
- Retrieval benefit prediction and irrelevant-context risk.
- Artifact immutability, configuration hashes, Git dirty state, and honest reporting.
- Why deterministic metrics precede an optional LLM judge.
- Why queue bounds/deadlines matter more than average throughput under overload.
- Why load-aware routing is intentionally schedule-dependent and must record trusted load snapshots.
- Why API keys and in-process rate limits are useful but not tenant authorization or distributed security.
