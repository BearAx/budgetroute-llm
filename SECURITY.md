# Security policy

Use [GitHub private vulnerability reporting](https://github.com/BearAx/budgetroute-llm/security/advisories/new). Include the affected version, impact, reproduction conditions, and a minimal proof of concept. Do not include real credentials, private prompts, model outputs containing personal data, or unrelated system data.

You should receive acknowledgement within seven days. Please allow a reasonable coordinated-disclosure window while impact and remediation are verified. Do not open a public issue with exploit details. If the private form is unavailable, open a minimal issue requesting a private channel without disclosing the vulnerability.

Security fixes target the latest released minor version. Older snapshots and arbitrary downstream model servers are unsupported.

## Security baseline

- GitHub dependency alerts, automated security fixes, and private vulnerability reporting are enabled at repository level.
- Dependabot monitors Python, GitHub Actions, and Docker dependencies.
- CI runs CodeQL's extended Python suite, dependency review, and `pip-audit`.
- The API supports optional bearer/API-key authentication, bounded queues, deadlines, per-process rate limiting, body/prompt limits, trusted Host validation, sanitized errors/configuration, and defensive response headers. Rate-limit identities use a process-random HMAC so raw credentials are not retained as limiter keys.
- OpenAI-compatible runtime credentials are read only from named environment variables. Non-loopback model endpoints require explicit opt-in and HTTPS.

Run `python -m budgetroute security-check --config <serving-config>` before deployment. Keep secrets in an orchestrator or secret manager, terminate TLS at a trusted reverse proxy, restrict network access to model runtimes, and collect logs/metrics without prompt bodies.

## Important limitations

This remains research software, not an authorization or content-safety boundary. API keys provide service authentication, not per-resource authorization. Rate limits, drift windows, feedback aggregates, and queues are process-local. The project does not implement tenant isolation, distributed quotas, durable audit storage, prompt-injection defenses, malware scanning, model-output policy enforcement, or a durable human-review queue. Add controls appropriate to the data and threat model before untrusted or sensitive use.
