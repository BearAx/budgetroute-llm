# Routing policies

Every decision contains route, policy, confidence, optional difficulty, reason, feature snapshot, and thresholds.

## Baselines

`always_small` estimates the inexpensive quality floor. `always_large` estimates the expensive quality ceiling. `random` samples from normalized configured probabilities using a hash of seed, request ID, and prompt, making repeated ordered inputs reproducible.

## Heuristic

The heuristic difficulty score combines bounded contributions from request length, sentences, code fences, mathematical symbols, long-output indicators, code/numeric category, URLs, and digit ratio. Rules are readable and configurable. Retrieval-required requests use the small model with context when a hit survives the score threshold. Ambiguous or explicitly unanswerable requests can abstain.

Heuristics are portable but dataset-sensitive. Feature thresholds must be validated on representative data, not tuned on test labels.

## Learned

The learned policy predicts whether the small backend will meet a quality threshold using the same numeric feature snapshot at training and serving. The artifact stores the preprocessing/model pipeline, ordered features, label definition, source hash, split IDs, seed, metrics, warning, and package versions.

Logistic regression is used when both classes exist; a dummy classifier provides explicit, non-crashing behavior for one-class demonstrations. Tiny in-sample results are not generalization evidence. Near-duplicate prompt grouping should be added for larger adapters.

## Retrieval first

Retrieval runs before final routing. A hit above the configured similarity threshold selects small-with-retrieval; otherwise backend choice falls back to difficulty. This can improve context-dependent quality but adds retrieval latency to all requests and can inject irrelevant text.

## Cascade

The small backend runs first. If backend confidence is below `cascade_confidence_threshold`, the large backend runs and the trace records the initial answer (internally when enabled), escalation reason, extra input/output tokens, and extra latency. High false escalation wastes compute; high false non-escalation harms quality.

## Abstention

Abstention is a first-class route, not an exception. It trades coverage for selective accuracy when required context is absent or the request is unanswerable. Production policies need domain-specific calibration and user experience design.

