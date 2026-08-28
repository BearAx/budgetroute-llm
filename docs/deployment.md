# Deployment

## Development

```bash
python -m budgetroute serve --config configs/serving/fake.yaml
```

Loopback fake mode may run without authentication. Replace the config only after installing required extras and running `doctor`, `validate-config`, and focused compatibility probes.

## Single-host replica deployment

`configs/serving/distributed.yaml` is the reference Phase 6 profile. It combines tenant scopes, trusted hosts, an explicit external TLS boundary, SQLite WAL coordination, durable feedback/reviews/audit, bounded batching, adaptation storage, and narrow content rules.

1. Provision a cryptographically random key of at least 32 bytes per subject in a secret manager. Inject the tenant JSON through `BUDGETROUTE_TENANT_KEYS_JSON`; never put it in YAML, images, `.env`, logs, or GitHub Actions output.
2. Assign every process a stable `BUDGETROUTE_REPLICA_ID`.
3. Put `data/state/budgetroute.db` and the calibration registry on persistent local storage accessible to every process on that host. Back them up consistently.
4. Replace `budgetroute.local` with the real DNS name in `api.trusted_hosts`.
5. Terminate HTTPS at a trusted proxy/load balancer and restrict direct access to the application port, or set an existing certificate/key pair in `api.tls_certfile`/`api.tls_keyfile`.
6. Run the exact-environment checks before starting:

```bash
python -m budgetroute validate-config --config configs/serving/distributed.yaml
python -m budgetroute security-check --config configs/serving/distributed.yaml
python -m budgetroute audit-check --config configs/serving/distributed.yaml
python -m budgetroute serve --config configs/serving/distributed.yaml
```

Health/readiness remain unauthenticated for orchestrator probes. Protected endpoints accept `Authorization: Bearer` or `X-API-Key`; prefer Bearer. The scoped tenant principal, not the client IP, controls shared quota and record visibility.

## Multi-host PostgreSQL deployment

`configs/serving/postgres.yaml` is the multi-host control-plane profile. Install the `postgres` extra, provision a dedicated PostgreSQL database with TLS, inject tenant credentials and the DSN through a secret manager, and replace the example trusted host. Prefer separate migration and runtime database roles as described in the [PostgreSQL operations runbook](postgres-operations.md).

The production profile disables automatic DDL. During deployment, expose the DDL-capable DSN only to a controlled migration job:

```bash
python -m budgetroute migrate-store --config configs/serving/postgres.yaml
```

Then expose the narrower runtime DSN to API replicas and run:

```bash
python -m budgetroute validate-config --config configs/serving/postgres.yaml
python -m budgetroute security-check --config configs/serving/postgres.yaml
python -m budgetroute audit-check --config configs/serving/postgres.yaml
python -m budgetroute serve --config configs/serving/postgres.yaml
```

Every replica verifies migration names/checksums before accepting traffic. `/readyz` queries both inference and operational storage and returns 503 if either is unavailable. The packaged Kubernetes reference uses a separate migration Job with a DDL DSN and API containers with only a runtime DSN; adapt its ingress/egress policy, resources, image digest, secrets, and model topology before use.

## Coordination semantics

The local queue is per process. The lease count is global across processes using the same SQLite database or every host using the same PostgreSQL database; active requests renew their leases and expiry recovers capacity from crashed processes. The tenant fixed-window quota is also transactional and shared. PostgreSQL uses database time to avoid replica clock skew. This bounds admitted requests; it does not persist prompt bodies or replay work after a disconnected client.

Use SQLite only on a reliable single host/filesystem and never mount it on an arbitrary network filesystem. PostgreSQL supports independent hosts inside one database consistency domain through atomic quota upserts, serialized lease admission/audit appends, idempotent constraints, and row-locked review transitions. It does not implement multi-region active/active consensus; cross-region failover, replication consistency, fencing, RPO/RTO, and split-brain prevention remain database/platform responsibilities.

## TLS and network policy

Non-loopback startup fails unless authentication is configured and one TLS mode is explicit. With external termination:

- expose only the proxy publicly;
- allow the proxy to reach the application on a private interface/network;
- validate forwarding/header policy and request-size limits at both layers;
- secure model-runtime ports so only BudgetRoute can reach them;
- set HSTS at the HTTPS edge and rotate certificates normally.

Built-in Uvicorn TLS is useful for controlled deployments but is not a replacement for edge DDoS protection, certificate automation, WAF, or service-mesh identity.

## Review, feedback, audit, and adaptation

Human-review responses create a durable case only when `api.review_enabled` is true. Review-scoped subjects can list their tenant cases, claim one, and resolve it with an optimistic `expected_version`; admins can operate across tenants. Resolution can create a correctness label without duplicating an existing feedback record.

`/v1/audit` returns retained application events and `/v1/audit/verify` checks their hash chain. Run `audit-check` regularly and export audit records to a separately controlled append-only archive if compliance or strong non-repudiation matters. Database-level compromise can replace the local chain.

Adaptation remains an operator action. Back up the registry, require the configured minimum labeled sample count, inspect candidate/holdout metrics and change points, promote only through the command, and test rollback. A promoted calibrator is not automatically wired into every separately configured policy; point the serving profile at the active artifact as part of the reviewed deployment release.

## Monitoring and load

`GET /metrics` includes local request/latency/drift/scheduler metrics and shared operational counters. `GET /v1/monitoring` includes mean/quantile/category drift and a shared recent-feature summary. Alerts should require sustained evidence and combine service, model-runtime, database/pool/lock/replication, host, and business-label telemetry.

Before setting replicas or queue limits, run an authorized steady-state load matrix against the exact deployment. Vary one relevant dimension at a time, include warm-up, preserve load artifacts, inspect overload and tail latency, then re-run after any scaling change. The generated replica multiplier is advisory only.

## Model runtimes

OpenAI-compatible profiles target the portable chat-completions subset. Loopback HTTP is permitted. Remote endpoints require explicit opt-in, HTTPS, and environment-backed credentials when applicable. Verify exact server/model revisions with `compatibility-matrix`; servers differ in seeds, tokenization, logprobs, batching, and extensions.

GPU deployments additionally require compatible driver/CUDA/PyTorch versions, appropriate wheels, explicit device allocation, sufficient VRAM, model licenses, cache volumes, and measured batch/quantization/compile settings. Configuration examples are not fit or speed claims.

## Container and repository controls

The supplied image is non-root and includes the PostgreSQL driver. `docker-compose.yml` remains the loopback SQLite example. `docker-compose.postgres.yml` starts a loopback-only API plus PostgreSQL using explicitly non-TLS local settings; it requires environment-injected tenant JSON and a URL-safe development password and is not a production secret/database topology.

`deploy/kubernetes/budgetroute.yaml` demonstrates three API replicas, a ClusterIP Service, readiness/liveness probes, PodDisruptionBudget, HPA, security contexts, and an ingress policy. It intentionally contains no Secret, database, ingress controller, or model server. HPA/resource values are starting placeholders until target-fleet load evidence exists. Managed database HA, TLS trust, backups/PITR, external secrets, image signing/scanning, egress policy, WAF/service-mesh controls, and rollback drills remain operator work.

GitHub workflows provide CodeQL, dependency review, Dependabot, packaging, offline tests, fake smoke, and dependency audit. Configure a `main` ruleset that blocks deletion/force push and requires the actual passing check names. Enable secret scanning/push protection and private vulnerability reporting. A solo owner should avoid approval rules that make legitimate merges impossible.

See [SECURITY.md](../SECURITY.md) and [limitations](limitations.md).
