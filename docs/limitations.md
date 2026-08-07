# Limitations and mitigation matrix

Phase 6 and 7 reduce the largest architectural gaps, but software cannot erase missing evidence, external infrastructure, or fundamental model uncertainty. The table distinguishes what is implemented from what must still be supplied.

| Area | Repository mitigation | Remaining boundary / required evidence |
|---|---|---|
| Limited development evidence | Revision-pinned public adapters, content hashes, group-safe splits, grouped bootstrap intervals, minimum-sample claim warnings, a 500-pair MMLU collection, and one persisted 100-record held-out routing evaluation | The held-out sample is still drawn from one MMLU development universe; external data, larger samples, repeated studies, domain review, and external validity remain required |
| Fake backend | Every response/artifact is explicitly fake and workflows match real interfaces | Fake quality, confidence, and latency are never model or capacity evidence |
| Model/hardware variance | Exact revisions/config/environment metadata, compatibility probes, warm/cold separation, real HTTP load artifacts, and a named RTX 3060/Qwen2.5 result | Quality and speed remain specific to weights, prompt, runtime, drivers, hardware, traffic, background load, and thermals |
| Confidence calibration | Delayed-label joins, chronological holdout, Brier/ECE gates, immutable promotion, change points, and rollback | Calibration can fail under future shift; representative recent labels and post-deployment monitoring remain mandatory |
| Drift | Numeric mean/quantile and category-distribution drift plus durable feature summaries | These do not prove semantic, label, factuality, abuse, or safety drift |
| Deterministic evaluation | Final-answer numeric extraction supports decimals/scientific notation/fractions/percentages; keyword matching respects phrases; bootstrap uncertainty is recorded | Exact metrics still miss open-ended usefulness, factuality, style, and safety; use domain experts or a validated optional judge |
| Batching and throughput | Actual batching/concurrency execute, benchmark throughput uses wall time, and load tests report p50/p95/p99, errors, overload, and claim boundaries | No speedup/capacity claim exists until measured on the target deployment; the autoscaling multiplier is heuristic |
| Retrieval scalability | Exact NumPy/FAISS plus configurable HNSW and revision-pinned semantic embeddings | Measure HNSW recall/latency, corpus freshness, licensing, and retrieval-induced answer quality on the target corpus |
| Retrieval usefulness | A paired, group-safe learned retrieval-benefit policy estimates whether context improves quality | Similarity/benefit scores are not causal guarantees; changed corpora/models require retraining and holdout evaluation |
| Replica coordination | SQLite WAL transactions provide same-host shared quotas, expiring global admission leases, counters, labels, reviews, and audit | SQLite is not multi-host/multi-region consensus; use a transactional PostgreSQL/Redis adapter for that topology |
| Durable scheduling | Global leases bound inflight work and recover on expiry; local queues are bounded and deadline-aware | Prompts are intentionally not persisted, so this is not a durable job queue and disconnected requests are not replayed |
| Tenant access | Environment-backed subject keys, scopes, constant-time comparison, and tenant-filtered records | No OIDC/MFA/user lifecycle, row encryption domain, billing, or compliance certification; use an IdP/secret manager and hardened datastore |
| Human review | Durable privacy-minimal cases, claim/resolve/versioning, idempotent labels, and audit events | No notification/SLA/ticketing UI or workforce integration; connect an external system when humans must actually review |
| Audit | Hash-chained events and continuity verification with bounded retention | A database administrator can replace the store; export to an independently controlled append-only/notarized sink for stronger assurance |
| Transport | Non-loopback config requires credentials and built-in TLS or explicit external termination | Edge certificates, network isolation, DDoS/WAF/service-mesh policy, and header trust are operator responsibilities |
| Content safety | Metadata bounds and configurable input/output literal rejection without echoing blocked content | Literal rules do not solve prompt injection, jailbreaks, malware, privacy leakage, factuality, or semantic policy enforcement |
| OpenAI-compatible runtimes | Safe URL/credential rules and exact single/batch compatibility artifacts | Vendors vary in seeds, logprobs, tokenization, batching, errors, and extensions; probe every exact runtime revision |
| Learned artifact supply chain | Metadata records source hashes, versions, splits, and label definitions | `joblib` loading is executable deserialization; accept artifacts only from trusted, integrity-verified build paths |

## Scientific interpretation

Bootstrap intervals quantify sampling variation for the observed independent groups; they do not correct selection bias or dataset mismatch. Minimum-sample warnings prevent silent overclaiming but do not make a threshold universally valid. Cache replay is excellent for comparing routing over fixed model outcomes and cannot reproduce live contention, batching, queueing, or thermal behavior.

## Operational interpretation

The supplied distributed profile is a realistic single-host control plane, not an internet-scale platform. It deliberately keeps prompts/answers out of operational SQLite state, which limits forensic and reviewer context but reduces retained sensitive data. Deployments must choose that trade-off explicitly and can integrate a separately governed content store if their threat model and retention policy allow it.

## Claims policy

Do not claim universal “production readiness,” solved prompt injection, model safety, semantic correctness, cost savings, HNSW scale, throughput, or latency from repository code or fake artifacts. Claims must name the exact dataset, sample size/groups, model and revision, prompt template, runtime, hardware, load shape, metric definition, uncertainty, and artifact location.
