# Deployment

## Local service

```bash
python -m budgetroute serve --config configs/serving/fake.yaml
```

Replace the config with `cpu.yaml` or `gpu.yaml` after installing Transformers dependencies and validating with `doctor`. Model caches should live outside the Git repository or under ignored `model-cache/`.

## Docker

`docker compose up --build` creates a CPU-safe fake service. The image uses Python 3.12 slim, installs API/reporting/learned dependencies without model downloads, runs as an unprivileged user, and probes `/healthz`.

For real CPU execution, install the Transformers extra in a derived image and mount a read/write Hugging Face cache at runtime. For GPU, use an NVIDIA CUDA-compatible base/runtime, an appropriate CUDA-enabled PyTorch wheel, NVIDIA Container Toolkit, explicit device allocation, and models sized for VRAM. These choices are intentionally deployment-specific.

## Production gaps

The initial service has validation, request IDs, limits, structured domain errors, sanitized configuration, readiness, and clean lifespan. It does not provide authentication, authorization, TLS, quotas, distributed load shedding, persistent queues, multi-tenant model isolation, or safety policy enforcement. Put it behind appropriate infrastructure before untrusted access.

