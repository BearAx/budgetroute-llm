# Portfolio notes

## Short description

BudgetRoute-LLM is a typed Python research system that routes requests across small/large local language models, retrieval, cascade escalation, and abstention while measuring deterministic quality and systems trade-offs.

## Extended description

The project separates model backends, routing policies, retrieval, evaluation, profiling, artifacts, and reporting behind testable interfaces. An offline deterministic mode exercises the same CLI, FastAPI, batching, benchmark, learned-router, and report paths as optional Hugging Face CPU/CUDA execution. Reproducibility metadata and explicit fake labeling prevent unmeasured portfolio claims.

## Resume bullets

- Built a quality-aware LLM routing system across small/large models, retrieval, cascade, and abstention; on **[real dataset/model/hardware]**, achieved **[measured quality]** at **[measured p50/p95 latency]** versus **[baseline]**.
- Engineered reproducible experiment artifacts, deterministic evaluation, calibration metrics, FastAPI serving, and offline CI with **[real test count/coverage at time of application]**.
- Implemented optional CPU/CUDA Transformers execution and portable cosine retrieval, reducing **[measured compute or latency]** by **[real measured percentage]** at a **[real configured quality target]**.

Replace every bracket only with traceable real artifacts. Do not use fake benchmark values.

## Two-minute demo

1. State the research question and show the architecture diagram (20 seconds).
2. Run `budgetroute demo` and point out easy, difficult, retrieval, cascade, and abstention traces (35 seconds).
3. Run the fake benchmark and show the unique artifact directory and prominent fake label (30 seconds).
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

