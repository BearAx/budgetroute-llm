# ADR 0005: Secure local-runtime and API boundaries by default

## Decision

Allow authentication-free development only on loopback. Require API-key authentication for non-loopback binding, reject wildcard trusted hosts in the deployment audit, bound request bodies/rates/queues/deadlines, and keep secrets in named environment variables. Permit plaintext OpenAI-compatible endpoints only on loopback; remote endpoints require explicit opt-in and HTTPS.

## Consequences

Accidental public exposure and plaintext remote model traffic fail validation. Configuration/artifacts never contain secret values. These controls are appropriate to a single research-service process, not a substitute for TLS termination, user authorization, tenant isolation, shared quotas, policy enforcement, or a secret manager.

ADR 0006 later extends this boundary with scoped tenant credentials, an explicit TLS declaration, same-host shared quotas/admission, durable workflows, and audit records. External identity, edge/network protection, secret management, and multi-host coordination remain deployment responsibilities.
