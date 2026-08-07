# Deployment

## Loopback development

```bash
python -m budgetroute serve --config configs/serving/fake.yaml
```

Replace the config with `cpu.yaml` or `gpu.yaml` after installing Transformers dependencies and validating with `doctor`. Model caches belong outside Git or under ignored `model-cache/`.

## Authenticated service

`configs/serving/secure.yaml` is a safe starting point for a non-loopback single-instance service. It requires a secret from `BUDGETROUTE_API_KEY`, rejects unknown Host headers, rate-limits identities, bounds body/prompt/queue sizes, uses request deadlines, and enables aggregate monitoring/feedback.

```powershell
$env:BUDGETROUTE_API_KEY = (New-Guid).Guid + (New-Guid).Guid
python -m budgetroute security-check --config configs/serving/secure.yaml
python -m budgetroute serve --config configs/serving/secure.yaml
```

Change `api.trusted_hosts` to real DNS names before deployment. Put TLS, IP/network policy, shared quotas, access logs, and user authorization at a trusted gateway. Store the API key in a secret manager or orchestrator, rotate it, and never place it in YAML, `.env`, shell history, images, or GitHub Actions logs.

Health/readiness remain unauthenticated for orchestrator probes. `/v1/*` and `/metrics` require authentication when enabled. The implementation accepts `Authorization: Bearer <key>` or `X-API-Key`; prefer Bearer. Error bodies are sanitized, and generated internal cascade answers are not exposed.

## Dynamic scheduling

Enable `batching.enabled` to use the bounded full-pipeline scheduler. Tune maximum batch size, first-item wait, queue capacity, admission timeout, and request deadline with a representative live load test. Short waits reduce tail latency; longer waits may improve accelerator utilization. A queue rejection returns HTTP 503 with `Retry-After`; a deadline returns 504. Clients should use bounded exponential backoff and idempotent request IDs.

Scheduler, rate limit, metrics, drift baseline, and feedback state are per process. Multiple workers or replicas require a gateway/shared limiter and centralized metrics. Do not interpret cache replay or fake timing as scheduling capacity.

## Local OpenAI-compatible runtimes

`configs/serving/local-openai-compatible.yaml` expects chat-completions servers at loopback `/v1` roots. It can target vLLM, llama.cpp server, Ollama's compatible endpoint, or another server implementing the required response subset. Start and secure those runtimes separately, align the configured model ID with each server, then run `doctor`.

Local HTTP is allowed only for loopback. Remote model endpoints require `allow_remote_endpoint: true`, HTTPS, and an explicit credential environment variable when needed. Remote use sends prompts across a new trust boundary; review retention, residency, provider authentication, TLS validation, and incident handling first.

## Monitoring and feedback

`GET /metrics` emits Prometheus text for request/error/review counts, latency sum, drift state, and scheduler counters. `GET /v1/monitoring` exposes a sanitized snapshot. No prompt, answer, feedback note, or request identifier is retained by `RuntimeMonitor`.

The default automatic drift baseline freezes at the configured minimum sample count. Production deployments should create a representative, reviewed baseline, alert on sustained—not single-window—changes, and require holdout evaluation plus rollback criteria before recalibration. `POST /v1/feedback` stores aggregate correctness only; use an authenticated durable system for labels needed in training.

## Docker

Set `BUDGETROUTE_API_KEY` in the invoking environment, then run `docker compose up --build`. Compose refuses a missing key. The CPU-safe fake service uses Python 3.12 slim, installs API/reporting/learned dependencies without model downloads, runs as an unprivileged user, binds the authenticated secure profile, and probes `/healthz`.

For real CPU execution, install the Transformers extra in a derived image and mount a read/write Hugging Face cache at runtime. For GPU, use an NVIDIA CUDA-compatible base/runtime, an appropriate CUDA-enabled PyTorch wheel, NVIDIA Container Toolkit, explicit device allocation, and models sized for VRAM. For model servers, use separate least-privilege containers/networks and expose runtime ports only to BudgetRoute.

## GitHub repository security

The repository contains Dependabot configuration plus CodeQL, dependency-review, and `pip-audit` workflows. Private vulnerability reporting and dependency alerts are enabled in repository settings. For a solo portfolio repository, review branch protection carefully: requiring another person's approval can prevent the owner from merging. At minimum, configure a ruleset for `main` that blocks force pushes/deletions and requires the CI, package, fake smoke, and security checks once their exact check names have completed successfully.

See the root `SECURITY.md` for coordinated disclosure and explicit security limitations.
