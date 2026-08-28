# PostgreSQL operations runbook

The PostgreSQL backend is the shared operational control plane for API replicas on different hosts. It stores quotas, expiring admission leases, aggregate counters, privacy-minimal prediction features, correctness labels, review state, and application audit events. It never stores API credentials, prompts, retrieved text, or generated answers.

## Guarantees and mechanisms

| Concern | Mechanism | Boundary |
|---|---|---|
| Tenant fixed-window quota | Atomic `INSERT ... ON CONFLICT ... WHERE` update | One PostgreSQL primary/consistency domain; fixed windows can burst at boundaries |
| Global inflight cap | Expired-lease cleanup plus count/insert under a transaction advisory lock | Admission bound, not a durable queue or completed-work ledger |
| Crash recovery | Database-clock expiry and request heartbeat renewal | Work may continue after lease loss; prompts are not replayed |
| Idempotency | Tenant/request unique constraints for predictions, feedback, and review creation | Clients still own stable request IDs |
| Review concurrency | Row lock plus explicit version check | No human notification, SLA, or ticketing integration |
| Audit ordering | Short transaction advisory lock around head read and append | Database administrators can still replace the database; export externally for stronger assurance |
| Schema safety | Ordered packaged SQL, global migration lock, recorded SHA-256 checksum | Forward-only migrations; schema rollback requires an explicit restore/compatibility plan |

Normal requests never keep a database transaction open while a model runs. Admission stores a renewable lease, commits, and releases the connection before inference begins.

## Roles and secrets

Use a dedicated database/schema and two credentials:

- a migration role that owns the schema and runs `budgetroute migrate-store` during a controlled rollout;
- a runtime role with connect/usage, table `SELECT`/`INSERT`/`UPDATE`/`DELETE`, sequence `USAGE`/`SELECT`, and read access to `budgetroute_schema_migrations`, but no schema DDL.

The production profile sets `postgres_auto_migrate: false`. Its init job/container receives the migration DSN; API containers receive only the runtime DSN. Both values are injected through `BUDGETROUTE_POSTGRES_DSN` in their separate processes. Do not put either value in YAML, command arguments, images, `.env`, logs, artifacts, or pull requests.

Prefer `sslmode=verify-full` with a trusted CA and matching database hostname. `verify-ca` or `require` satisfy the application fail-closed check but provide weaker server-identity assurance. `sslmode=disable`, `allow`, and `prefer` are rejected when `postgres_require_tls: true`. The local Compose profile opts out explicitly and must remain loopback-only development infrastructure.

## Bootstrap and rollout

1. Provision the database, roles, TLS trust, network policy, monitoring, automated backups, and point-in-time recovery.
2. Set the migration-role DSN in the operator environment and run:

   ```bash
   python -m budgetroute migrate-store --config configs/serving/postgres.yaml
   ```

3. Replace the environment value with the runtime-role DSN plus tenant credentials, then run `validate-config`, `security-check`, and `audit-check` against the exact release environment.
4. Start one canary replica. Confirm `/readyz`, database connections, quota/lease behavior, error rate, and audit verification before increasing replicas.
5. Roll out gradually. Preserve the image digest, config hash, migration versions, and test/load evidence with the release record.

Every startup with automatic migration disabled verifies that all packaged migration names and checksums match the database. Missing, modified, or newer migrations fail startup. Migrations are transactional and serialized, so multiple controlled migration jobs are safe, though one explicit job is easier to observe.

## Pool and capacity budget

`postgres_pool_min_size` and `postgres_pool_max_size` apply per API process. The upper-bound connection budget is:

```text
maximum database connections = replicas × pool_max_size + migration/admin/monitoring reserve
```

Set that number below the database or proxy limit. Start with a small pool, observe checkout wait and query latency, then load-test the exact replica/model/database topology. A larger pool can increase contention and does not make inference faster. Use a managed connection proxy only after verifying transaction/advisory-lock compatibility.

## Monitoring and drills

Alert on readiness failures, pool checkout timeouts, connection saturation, transaction/lock latency, rejected quotas/admission, lease renewal failures, migration/checksum failures, replica restarts, database storage/replication lag, backup age, and audit verification. Centralize metadata-only logs and keep prompt content disabled.

Rehearse at least these cases in a non-production environment:

- terminate an API replica during generation and confirm its lease expires;
- stop database connectivity and confirm `/readyz` returns 503 and protected requests fail without silently using process-local state;
- run concurrent quota/lease/review operations and verify the configured caps and audit chain;
- rotate runtime and tenant credentials without logging either value;
- restore a backup to an isolated database, run checksum/audit verification, and measure recovery time/data loss;
- deploy a backward-compatible application change, canary it, and exercise the application rollback path.

## Backup, restore, and rollback

Use database-native encrypted backups and point-in-time recovery, define retention/RPO/RTO, test restores, and protect backup access separately from runtime credentials. The hash chain detects retained-row edits but is not a substitute for protected backups or an independently controlled append-only export.

Database migrations are forward-only. Do not edit an applied SQL file. Design future migrations to be expand/contract and compatible with the previous application during a rolling deployment. Because older binaries intentionally reject unknown migrations, rolling back across a new schema version requires a reviewed compatibility release or restoring the matching pre-migration database—not simply changing the image tag.
