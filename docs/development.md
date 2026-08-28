# Development

Use Python 3.11–3.13 and a repository-local virtual environment. `scripts/bootstrap.ps1` and `scripts/bootstrap.sh` install the editable development package. Do not commit `.venv`, caches, downloaded models/data, indexes, credentials, operational databases, calibration registries, or experiment outputs.

## Quality gate

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy src
python -m pytest
python -m pytest --cov=budgetroute --cov-report=term-missing --cov-fail-under=75
python -m build
python -m twine check dist/*
```

Use the narrowest test first, then the full gate. Tests under `real_model` are optional and excluded from CI unless deliberately selected. Default tests stay offline; the dedicated `postgres-contract` CI job supplies PostgreSQL and runs the `postgres` marker tests. To run those locally, point `BUDGETROUTE_TEST_POSTGRES_DSN` only at a disposable test database—the tests isolate their tables in a random schema and remove it afterward.

Run `python -m budgetroute security-check --config configs/serving/fake.yaml` locally and exercise the relevant SQLite or PostgreSQL profile with temporary environment-only credentials when changing service controls. Never print secret values in tests or artifacts. Changes to batching, operational transactions, migrations, tenant isolation, audit, adaptation, retrieval, monitoring, endpoint validation, or GitHub workflows require focused tests and corresponding architecture/deployment/security documentation.

## Change process

Read the relevant architecture, decision, and methodology documents before changing behavior. Substantial changes should explain scope, architecture impact, staged implementation, validation, risks, and final outcome in their issue or pull request. Preserve dependency injection and public schema compatibility. Add tests, update configuration/public documentation, and record methodology changes explicitly.

Benchmark claims need preserved real artifacts and compatible comparisons. Fake artifacts may validate workflows but never support performance claims.
