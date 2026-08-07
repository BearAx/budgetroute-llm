# Portfolio notes

## Short description

BudgetRoute-LLM is a typed Python LLM routing and evaluation system that combines dynamic batching, retrieval, cascade escalation, tenant-aware serving, durable review/feedback, and gated confidence adaptation while measuring quality, cost, and systems trade-offs.

## Extended description

The project separates model backends, scheduling/load telemetry, routing, retrieval, evaluation, operations, adaptation, artifacts, and reporting behind testable interfaces. An offline deterministic mode exercises the same CLI, FastAPI, bounded batching, tenant scopes, SQLite coordination, review/audit, learned policies, and reports as optional Hugging Face CPU/CUDA or local OpenAI-compatible execution. Reproducibility metadata and explicit fake labeling prevent unmeasured portfolio claims.

## Resume bullets

- Built a quality-aware LLM routing system across small/large models, retrieval, cascade, and abstention; on **[real dataset/model/hardware]**, achieved **[measured quality]** at **[measured p50/p95 latency]** versus **[baseline]**.
- Engineered reproducible experiment artifacts, deterministic evaluation, calibration metrics, FastAPI serving, and offline CI with **[real test count/coverage at time of application]**.
- Implemented optional CPU/CUDA Transformers execution and portable cosine retrieval, reducing **[measured compute or latency]** by **[real measured percentage]** at a **[real configured quality target]**.
- Built a bounded deadline-aware scheduler with ordered cascade batching, load/cost-aware policies, Prometheus metrics, drift alerts, and secure overload handling; sustained **[measured throughput]** at **[measured p95]** on **[hardware/runtime]**.
- Added a safe OpenAI-compatible local-runtime boundary and GitHub security automation (CodeQL, dependency review, Dependabot, `pip-audit`), with **[measured security/operations evidence]** from the published workflow artifacts.
- Implemented transactional same-host replica coordination, scoped tenant access, durable review/feedback, tamper-evident audit verification, and gated delayed-label recalibration; validated **[real deployment topology and evidence]**.
- Trained a separate retrieval-benefit policy and added revision-pinned semantic HNSW retrieval; measured **[recall/quality delta/latency]** on **[named corpus and embedding revision]**.

Replace every bracket only with traceable real artifacts. Do not use fake benchmark values.

## Two-minute demo

1. State the research question and show the architecture diagram (20 seconds).
2. Run `budgetroute demo` and point out easy, difficult, retrieval, cascade, and abstention traces (35 seconds).
3. Show tenant scopes, a durable review transition, audit verification, and an adaptation candidate (35 seconds).
4. Open the policy/uncertainty or load artifact and explain its claim boundary (20 seconds).
5. Show the pinned real-model/semantic profiles and name the evidence still required (10 seconds).

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
- Why local batching and global admission leases are distinct, and where SQLite stops scaling.
- Why hash chaining is tamper-evident but not an externally notarized audit log.
- Why calibration promotion uses chronological evidence and never mutates the live model in the request path.
- Why literal content rules do not solve prompt injection or semantic model safety.
