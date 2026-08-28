# Security policy

Report vulnerabilities through [GitHub private vulnerability reporting](https://github.com/BearAx/budgetroute-llm/security/advisories/new). Include the affected version, impact, reproduction conditions, and a minimal proof of concept. Never include real credentials, private prompts, personal data, or unrelated system data. If the private form is unavailable, open a minimal public issue requesting a private channel without exploit details.

Acknowledgement is targeted within seven days. Security fixes target the latest released minor version; arbitrary downstream model servers and modified deployments are unsupported.

## Repository security baseline

- Dependabot covers Python, Docker, and GitHub Actions.
- CI runs CodeQL extended analysis, dependency review, packaging/tests, and `pip-audit`.
- Generated credentials, operational SQLite files, calibration registries, model/data caches, indexes, and experiment outputs are ignored; PostgreSQL DSNs are environment-only and omitted from sanitized config.
- Runtime configuration can be audited without printing secrets using `budgetroute security-check`.

Repository owners should enable private vulnerability reporting, Dependabot alerts/updates, secret scanning, and push protection. Protect `main` against force-push/deletion and require the real CI/package/security checks. Pin third-party Actions by immutable commit where practical and review automated dependency updates before merging.

## Service controls implemented here

- Constant-time legacy or tenant credential comparison; environment-only secret sources.
- Tenant principals and inference/feedback/review/admin scopes with tenant-scoped operational queries.
- Trusted Host validation, body/prompt/metadata bounds, defensive headers, non-reflective validation/config/domain errors, fixed-window tenant quotas, local queue bounds/deadlines, and renewable global admission leases.
- Required authentication plus an explicit built-in or external TLS boundary for non-loopback binding.
- Configurable literal prompt/metadata and complete serialized-response blocklists that reject without echoing content.
- Privacy-minimal predictions, delayed feedback, durable review transitions, shared metrics, and hash-chained application audit events.
- Optional PostgreSQL coordination uses bounded pools, database time, atomic/locked transactions, checksum-verified migrations, database-aware readiness, and a configurable separation between DDL migration and runtime credentials.
- PostgreSQL DSN parsing and connection failures suppress driver exception details and chained exceptions so startup diagnostics cannot echo secret-bearing connection strings.
- PostgreSQL profiles fail closed when required TLS is missing; `verify-full` is recommended even though `require` and `verify-ca` are accepted for platform compatibility.
- Remote OpenAI-compatible endpoints disabled by default; explicit opt-in requires HTTPS, and credentials are read from named environment variables.

## Required deployment controls

Use a secret manager and rotate tenant/database credentials. Expose only an HTTPS edge, restrict direct application/model/database access, define data retention and backup/PITR policy, centralize logs without prompts, monitor authentication/quota/audit/database failures, and rehearse restore/rollback/incident response. Use an external identity provider for user lifecycle/MFA and a separately controlled append-only audit sink when the threat model requires one. For PostgreSQL, separate DDL and runtime roles, prefer certificate/hostname verification, budget total pool connections across replicas, and protect backups separately.

The built-in literal rules are narrow defense in depth. They do not solve prompt injection, jailbreaks, sensitive-data inference, factuality, malware, or model-output safety. Add task-specific moderation/guardrails, sandboxing, egress controls, human review, red-team tests, and legal/privacy controls appropriate to the data.

Learned routing artifacts use `joblib`, which can execute code while loading. Train them locally or obtain them from a trusted, integrity-verified source; never load an untrusted artifact.

## Honest boundary

BudgetRoute-LLM is research software, not a certified authorization, safety, privacy, availability, or compliance product. SQLite/PostgreSQL tenant rows provide application-level isolation, not separate encryption domains or PostgreSQL row-level-security policy. The audit chain detects retained-row mutation but is not externally notarized. TLS termination, identity federation, DDoS/WAF protection, host hardening, managed-database HA/replication/backups, and multi-region resilience remain deployment responsibilities.
