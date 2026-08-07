# Routing policies

Every decision contains route, policy, confidence, optional difficulty, reason, feature snapshot, and thresholds.

## Baselines

`always_small` estimates the inexpensive quality floor. `always_large` estimates the expensive quality ceiling. `random` samples from normalized configured probabilities using a hash of seed, request ID, and prompt, making repeated ordered inputs reproducible.

## Heuristic

The heuristic difficulty score combines bounded contributions from request length, sentences, code fences, mathematical symbols, long-output indicators, code/numeric category, URLs, and digit ratio. Rules are readable and configurable. Retrieval-required requests use the small model with context when a hit survives the score threshold. Ambiguous or explicitly unanswerable requests can abstain.

Heuristics are portable but dataset-sensitive. Feature thresholds must be validated on representative data, not tuned on test labels.

## Learned

The learned policy predicts whether the small backend will meet a quality threshold using the same numeric feature snapshot at training and serving. The artifact stores the preprocessing/model pipeline, ordered features, label definition, source hash, three group-aware partitions, scalar calibrator, calibration-selected serving threshold, seed, raw/calibrated test metrics, warnings, and package versions.

Logistic regression is used when both training classes exist; a dummy classifier provides explicit, non-crashing behavior for one-class demonstrations. Related records share `group_id` and cannot cross train/calibration/test boundaries when enough groups exist. Tiny in-sample fallbacks are labeled and are not generalization evidence.

## Retrieval first

Retrieval runs before final routing. A hit above the configured similarity threshold selects small-with-retrieval; otherwise backend choice falls back to difficulty. This can improve context-dependent quality but adds retrieval latency to all requests and can inject irrelevant text.

## Cascade

The small backend runs first. Transformers confidence is the geometric mean selected-token probability, accompanied by log-probability and entropy signals. If a configured calibration artifact exists, its scalar temperature transforms this raw value before thresholding. Below the selected threshold, the large backend runs and the trace records the initial answer (internally when enabled), escalation reason, extra input/output tokens, and extra latency. High false escalation wastes compute; high false non-escalation harms quality.

## Load aware

`load_aware` combines deterministic difficulty with trusted per-backend inflight utilization and exponentially weighted observed latency. Easy requests prefer the lower-pressure/latency backend. Hard requests prefer the large backend, but may use a guarded cascade when the large backend crosses the overload threshold. When both backends are overloaded and a sufficiently hard request crosses the review threshold, the policy emits explicit human review.

Load values come from backend decorators, not client metadata. Decisions include both snapshots and active thresholds. Because live load is time-dependent, load-aware routing is not expected to choose identical routes under different concurrency schedules; benchmark artifacts preserve every decision.

## Budget aware

`budget_aware` estimates each backend's input/output cost units from configured per-1,000-token rates, combines estimated quality, load-adjusted latency, and cost with explicit weights, then selects the highest-utility eligible backend. `max_estimated_cost_units` can make a route ineligible. If no backend fits, the policy abstains or requests human review according to configuration.

Cost units are deployment supplied. They can represent money, energy, reserved GPU time, or normalized compute, but must use one consistent unit within a run. They are estimates, not observed invoices. The transparent quality estimator must be validated or replaced with deployment evidence before operational use.

## Abstention

Abstention is a first-class route, not an exception. It trades coverage for selective accuracy when required context is absent or the request is unanswerable. Production policies need domain-specific calibration and user experience design.

## Human review

Human review is distinct from abstention and records a reason in the execution trace. The response states that an external integration is required. BudgetRoute-LLM does not create a ticket, persist a case, notify a person, or claim that review occurred; deployments must connect this outcome to a durable authenticated workflow and feed reviewed outcomes back through an approved channel.
