#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python command not found: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e '.[dev]'

echo "Bootstrap complete. The script cannot persist shell activation."
echo "Next: source .venv/bin/activate"
echo "Then: python -m budgetroute demo --config configs/serving/fake.yaml"

