# Development

Use Python 3.11 or 3.12 and a repository-local virtual environment. `scripts/bootstrap.ps1` and `scripts/bootstrap.sh` install the editable development package. Do not commit `.venv`, caches, downloaded models/data, indexes, credentials, or experiment outputs.

## Quality gate

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy src
python -m pytest
python -m pytest --cov=budgetroute --cov-report=term-missing --cov-fail-under=75
python -m build
```

Use the narrowest test first, then the full gate. Tests under `real_model` are optional and excluded from CI unless deliberately selected. Default tests must stay offline.

## Change process

Read `AGENTS.md` and relevant architecture/method docs. Substantial changes require an execution plan under `docs/exec-plans/` according to `PLANS.md`. Preserve dependency injection and public schema compatibility. Add tests, update configuration and public documentation, and record methodology changes explicitly.

Benchmark claims need preserved real artifacts and compatible comparisons. Fake artifacts may validate workflows but never support performance claims.

