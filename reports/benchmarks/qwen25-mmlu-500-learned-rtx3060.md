# Paired-quality routing on held-out MMLU

## Technical summary

This completed live CUDA study does **not** support a quality-preserving or latency-reducing routing claim. Qwen2.5-0.5B and Qwen2.5-1.5B were first run on the same 500 deterministic subject-stratified MMLU test questions. A paired-quality router was fit on 300 examples, calibrated on 100, and evaluated once on 100 persisted, untouched examples.

On that held-out set, always-large achieved **53% accuracy** (95% grouped bootstrap CI **43%-63%**) at **423.3 ms p50 / 528.6 ms p95**. The learned router sent 8 requests to the small model and 92 to the large model, achieving **51%** (CI **41%-61%**) at **445.9 ms p50 / 503.0 ms p95**. It made no quality wins over always-large and lost two questions. The observed p50 was 5.3% higher, so this run provides no measured latency win.

The calibrated cascade also reached **51%** and escalated 83/100 requests, producing **689.7 ms p50 / 895.9 ms p95**. The honest conclusion is that the current prompt-only features and token-confidence signal are insufficient for this model pair and sample. The experiment is valuable because it closes train/test leakage, names the failure, and identifies the next model improvements rather than presenting a weak router as a success.

## The learned router did not beat the fixed-large baseline

All four policies evaluated the same 100 held-out records once after a cache-isolated warm-up. Latency is live end-to-end latency per successful request. Accuracy is exact multiple-choice classification correctness.

| Policy | Route / escalation behavior | Accuracy (95% CI) | p50 ms | p95 ms | p99 ms | req/s |
|---|---|---:|---:|---:|---:|---:|
| Always small | 100 small | 41% (31%-51%) | 353.9 | 429.3 | 451.9 | 3.19 |
| Always large | 100 large | 53% (43%-63%) | 423.3 | 528.6 | 543.7 | 2.45 |
| Learned paired quality | 8 small / 92 large | 51% (41%-61%) | 445.9 | 503.0 | 530.7 | 2.40 |
| Calibrated cascade | 100 small first / 83 escalations | 51% (41%-61%) | 689.7 | 895.9 | 928.0 | 1.52 |

The learned and cascade outputs each matched 51 always-large successes, added no unique success, and replaced two correct large-model answers with incorrect answers. Their 51% point estimate is therefore exactly two percentage points below always-large on paired records. The bootstrap intervals overlap; no hypothesis test or repeated-trial timing design was used.

The live matrix completed **400/400 requests with zero failures**. Its measured policy intervals totaled 179.76 seconds, with 3,979.6 MiB maximum allocated CUDA memory. Aggregate throughput combines sequential policy runs and is not a serving-capacity result.

## The 500-pair development set contains routing opportunity

The baseline collection completed 1,000/1,000 live requests with zero failures. Across 500 paired questions, small was correct on 40.2% (CI 36.0%-44.4%) and large on 55.2% (CI 50.8%-59.4%). The outcome matrix was:

| Paired outcome | Questions | Share |
|---|---:|---:|
| Both correct | 152 | 30.4% |
| Small only correct | 49 | 9.8% |
| Large only correct | 124 | 24.8% |
| Neither correct | 175 | 35.0% |

An outcome oracle choosing the correct backend whenever either succeeded would score 65.0% on all 500 questions, versus 55.2% for always-large. On the held-out 100, the oracle was 63% versus 53% for always-large. Routing opportunity therefore exists in the observed outcomes, but this classifier did not capture it.

Always-small was faster than always-large in the 500-question collection: **337.8 versus 426.9 ms p50** and **413.9 versus 484.7 ms p95**. That establishes a model-level latency/quality trade-off on this laptop; it does not establish that the learned policy realizes the trade-off.

## Split, target, and validation design

The paired label is `small_model_quality >= large_model_quality`. It treats ties—including both-correct and both-wrong cases—as safe small routes because selecting small does not reduce observed quality for that example. The deterministic group split contained 300 training, 100 calibration, and 100 test examples with no group overlap.

| Partition | Records | Paired-safe | Paired-unsafe | Role |
|---|---:|---:|---:|---|
| Train | 300 | 226 | 74 | Fit scaling and logistic classifier |
| Calibration | 100 | 72 | 28 | Fit temperature and choose serving threshold |
| Test | 100 | 78 | 22 | Final classification and routing outcomes only |

The calibration-selected learned threshold was 0.8055, targeting 80% selective accuracy. It achieved 83.3% on 12% calibration coverage, then 75.0% precision on 8% test coverage. Test ROC AUC was 0.600, F1 was 0.140, and thresholded accuracy was 0.260 because the conservative policy rejected most of the 78 safe-small examples. End-to-end quality is the decision metric; classifier accuracy alone would obscure the two observed quality losses.

The cascade confidence calibrator used small-model token likelihood, fit a temperature of 20.0 on 100 calibration examples, and selected threshold 0.5324. Its test ROC AUC was 0.673, but 83 escalations still produced the same two quality losses as the learned router and materially worse latency.

## Exact models, data, and runtime

| Component | Exact setup |
|---|---|
| Small model | `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775` |
| Large model | `Qwen/Qwen2.5-1.5B-Instruct` at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Model execution | Hugging Face Transformers 5.14.1, PyTorch 2.12.1+cu130, CUDA runtime 13.0, FP16, no quantization or compilation |
| Generation | Chat template, greedy decoding, 8 maximum new tokens, temperature 0, seed 42 |
| Dataset | `cais/mmlu`, `test`, config `all`, revision `c30699e8356da336a370243923dbaf21066bb9fe`, MIT |
| Parent selection | 500 records, deterministic subject-stratified round robin, seed 42, prompt version v1 |
| Held-out selection | Persisted 100 router-test IDs; derived manifest links to the parent SHA-256 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB, current 95 W ceiling, driver 610.62, WDDM |
| Host | Lenovo 82JQ; AMD Ryzen 7 5800H (8C/16T); 15.86 GiB RAM; Windows 11 Pro build 26200 |
| Python | 3.13.13 |

The parent dataset digest is `5b25d20f38e91120bd3303ad32cc35d54ae9eb0326468532f41dcfaa2b31ede7`. The derived held-out digest is `c578c3024ce93db35be6c659570f8b3b5cb4e9393e337b742362aacee1d2bfce` and its manifest records the parent digest.

## Reproducibility and robustness checks

- Baseline command: `python -m budgetroute benchmark --config configs/benchmarks/real-gpu-mmlu-500-live.yaml`
- Training command: `python -m budgetroute train-router --artifacts outputs/20260807T121512Z_benchmark --output outputs/router/mmlu-500-paired.joblib --quality-threshold 1.0 --target-selective-accuracy 0.8 --label-strategy paired_quality --seed 42`
- Held-out command: `python -m budgetroute benchmark --config configs/benchmarks/real-gpu-mmlu-500-learned-live.yaml`
- Baseline run ID / timestamp: `20260807T121512Z_benchmark` / 2026-08-07 12:21:21 UTC
- Held-out run ID / timestamp: `20260807T122221Z_benchmark` / 2026-08-07 12:25:53 UTC
- Benchmark source commit: `80b85333b8cbf3484717a8c1f04a3a25a7f6efc0`
- Git state during both publication runs: clean
- Baseline configuration SHA-256: `8be6a368df379e380a19df3c05505fc62f3b3c3641fe6bc4d50402aadd20ae1f`
- Held-out configuration SHA-256: `51f8b39146839961ce225783c6f3ac04b96ea837ab3af1df5846de96b406ac03`
- Clean rerun versus development baseline: 1,000/1,000 answers, quality scores, errors, and routes matched
- Replay audit: 12/12 semantic comparisons matched across three records and four policies
- Bootstrap: 2,000 deterministic grouped resamples over 500 baseline groups or 100 held-out groups at 95% confidence
- Load shape: batch size 1, concurrency 1, one cache-isolated warm-up and one measured pass per policy

The sibling [JSON evidence](qwen25-mmlu-500-learned-rtx3060.json) preserves exact numeric values. Raw predictions, routes, timings, model generations, caches, datasets, and executable `joblib` artifacts remain ignored.

## Limitations and uncertainty

- The held-out set was sampled from the same MMLU-500 development universe before fitting. It is untouched by model fitting and threshold selection, but it is not an external dataset or a full-suite estimate.
- Prompt features are intentionally cheap and generic. They omit subject identity, model embeddings, per-backend answer agreement, and other signals that could improve routing but add cost or leakage risk.
- The paired-safe class includes both-wrong ties. That target optimizes non-degradation relative to large, not absolute correctness, and its 75.2% prevalence makes thresholded accuracy easy to misread.
- Each live policy has one sequential measured pass on one laptop with desktop processes active. Request percentiles are not repeated machine-level trials; thermal and background variation can change them.
- The observed learned p95 was lower than always-large while p50 and mean were higher. Without repeated randomized policy-order trials, this mixed timing result is not evidence of a latency improvement.
- Local monetary cost rates are zero. Route share and token counts are compute proxies, not observed savings, energy, or cloud billing.
- Exact-match multiple-choice accuracy does not establish factual reliability, safety, usefulness, or performance on representative deployment traffic.

## Recommended next steps

1. Keep always-large as the quality baseline for this model pair; do not deploy the current learned or cascade artifacts as a claimed optimization.
2. Add model-aware but pre-generation features, such as prompt embeddings and subject/category encoding, then select them using train/calibration only.
3. Evaluate a two-stage agreement or draft-verification policy whose added compute is measured explicitly; do not assume confidence is correctness.
4. Reserve a new, larger external or later-seeded test set before tuning again, and add paired uncertainty for quality deltas.
5. Run randomized repeated timing trials plus batch/concurrency matrices before making latency, throughput, or capacity claims.

## Further questions

- Can prompt embeddings identify the 49/500 cases where small succeeded and large failed without leaking answer labels?
- Does a larger model gap create more predictable routing opportunity, or merely a stronger always-large baseline?
- Would optimizing expected quality minus measured compute cost produce a more useful operating point than the 80% selective-accuracy constraint?
- How stable are route decisions across prompt templates, MMLU seeds, runtime versions, and representative domain data?
