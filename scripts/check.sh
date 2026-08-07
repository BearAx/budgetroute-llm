#!/usr/bin/env bash
set -euo pipefail

if [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=.venv/bin/python
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

"$PYTHON_BIN" -m compileall -q src tests
"$PYTHON_BIN" -m ruff format --check .
"$PYTHON_BIN" -m ruff check .
"$PYTHON_BIN" -m mypy src
"$PYTHON_BIN" -m pytest
"$PYTHON_BIN" -m budgetroute security-check --config configs/serving/fake.yaml
"$PYTHON_BIN" -m build
