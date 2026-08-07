# Qwen2.5 routing on stratified MMLU-100

## Technical summary

This is a completed live CUDA benchmark, not fake or replay timing. On 100 deterministic subject-stratified MMLU test questions, Qwen2.5-1.5B achieved the highest observed accuracy at **57%** (95% grouped bootstrap CI **47%-66%**) with **461.4 ms p50** and **536.9 ms p95** request latency. The heuristic routed 78 requests to Qwen2.5-0.5B and 22 to Qwen2.5-1.5B, reaching **46%** accuracy (CI **36%-56%**) at **390.3 ms p50** and **490.1 ms p95**.

Relative to always-large, the heuristic's observed p50 was 15.4% lower and p95 was 8.7% lower, but observed accuracy was 11 percentage points lower. The accuracy intervals overlap and this single 100-question sample is not sufficient to claim quality retention, a causal speedup, or general MMLU performance. The honest outcome is a measured latency/quality trade-off and a concrete target for a learned router.

## Policy results

All policies evaluated the same 100 records once after one cache-isolated warmup. Latency is end-to-end per successful request; throughput is each policy's sequential measurement-wall request rate. Accuracy is classification correctness.

| Policy | Routes / escalation | Accuracy (95% CI) | p50 ms | p95 ms | p99 ms | req/s |
|---|---|---:|---:|---:|---:|---:|
| Always small | 100 small | 43% (33%-53%) | 372.0 | 452.5 | 509.5 | 3.07 |
| Always large | 100 large | 57% (47%-66%) | 461.4 | 536.9 | 566.4 | 2.27 |
| Heuristic | 78 small / 22 large | 46% (36%-56%) | 390.3 | 490.1 | 517.5 | 2.82 |
| Cascade | 100 small first / 12 escalations | 43% (33%-52%) | 385.8 | 586.4 | 667.8 | 2.61 |

The uncalibrated cascade did not improve accuracy and produced the worst tail latency. Its 12 escalations and ECE of 0.447 are evidence that raw token likelihood is not a correctness probability and that the cascade threshold needs task-specific calibration.

The complete matrix contained 400/400 successful requests, zero errors, a 150.43 s measured wall interval, 2.66 aggregate requests/s, and 3,948.3 MiB maximum allocated CUDA memory. Aggregate throughput combines four sequential policy runs and is not a serving-capacity result.

## Models, data, and hardware

| Component | Exact setup |
|---|---|
| Small model | `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775` |
| Large model | `Qwen/Qwen2.5-1.5B-Instruct` at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Model execution | Hugging Face Transformers 5.14.1, PyTorch 2.12.1+cu130, CUDA runtime 13.0, FP16, no quantization or compilation |
| Generation | chat template, greedy decoding, 8 maximum new tokens, temperature 0, seed 42 |
| Dataset | `cais/mmlu`, `test`, config `all`, revision `c30699e8356da336a370243923dbaf21066bb9fe`, MIT |
| Selection | 100 records, deterministic subject-stratified round robin, seed 42, prompt version v1 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB, 95 W power cap, driver 610.62, WDDM |
| Host | Lenovo 82JQ; AMD Ryzen 7 5800H (8C/16T); 15.86 GiB RAM; Windows 11 Pro build 26200 |
| Python | 3.13.13 |

The MMLU JSONL digest is `9d1136e067a45d37b0f7704ba0bad422fe0c6dd833c8566d3f4ab07c612cd0ab`. The sample spans all 57 subjects in round-robin order, but one or two questions per subject cannot support subject-level conclusions.

## Protocol and provenance

- Live command: `python -m budgetroute benchmark --config configs/benchmarks/real-gpu-mmlu-100-live.yaml`
- Replay audit: `python -m budgetroute verify-replay --config configs/benchmarks/real-gpu-mmlu-100.yaml --sample-size 3`
- Run ID: `20260807T111806Z_benchmark`
- Run timestamp: 2026-08-07 11:21:10 UTC
- Benchmark source commit: `320ef0c85421f43f8449b499724e19f0acf24113`
- Git state during the benchmark: clean
- Configuration SHA-256: `7bd3395c92d2aa554fba65fc0e14e943782b1ce99738be81a713f388718de57f`
- Dataset SHA-256: `9d1136e067a45d37b0f7704ba0bad422fe0c6dd833c8566d3f4ab07c612cd0ab`
- Replay verification after lifecycle hardening: 12/12 semantic comparisons matched across three records and four policies at commit `9f4099d07e302726c3f545ecf862e5eb442a5735`
- Bootstrap: 1,000 deterministic grouped resamples over 100 record groups at 95% confidence
- Load shape: batch size 1, concurrency 1, one warmup and one measured pass per policy

The sibling [JSON summary](qwen25-mmlu-100-rtx3060.json) preserves exact numeric values. Raw local artifacts include resolved configuration, environment, predictions, routes, timings, errors, metrics, and figures; they remain ignored because the repository does not commit ordinary outputs or generated text.

## Limitations and next experiment

- This is a 100-question sample, not the full MMLU suite. Bootstrap intervals describe sampling variation in these records and do not remove selection or prompt bias.
- Each policy has one measured pass on one laptop under WDDM with desktop processes active. p95/p99 are across requests, not independent machine runs; thermals and background load may affect timing.
- The benchmark is sequential at concurrency 1 with batching disabled. It is not a saturation, multi-user, or deployment capacity test.
- Local cost units are configured as zero, so the report makes no monetary-savings claim. Route share and token counts are the available compute proxies.
- The heuristic and raw-confidence cascade were not trained/calibrated on a separate MMLU split. Their negative result must not be reframed as a production routing win.
- The next defensible comparison is a larger held-out sample with a trained/calibrated router, repeated timing trials, and batch/concurrency matrices. Production claims additionally require target-runtime load tests and representative domain/human evaluation.
