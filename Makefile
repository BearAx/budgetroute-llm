PYTHON ?= python

.PHONY: install format format-check lint typecheck test test-unit test-integration coverage check doctor fake-demo fake-benchmark report serve-fake build clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check . --fix

format-check:
	$(PYTHON) -m ruff format --check .

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest

test-unit:
	$(PYTHON) -m pytest tests/unit

test-integration:
	$(PYTHON) -m pytest tests/integration

coverage:
	$(PYTHON) -m pytest --cov=budgetroute --cov-report=term-missing --cov-report=xml --cov-fail-under=75

check: format-check lint typecheck test build

doctor:
	$(PYTHON) -m budgetroute doctor --config configs/serving/fake.yaml

fake-demo:
	$(PYTHON) -m budgetroute demo --config configs/serving/fake.yaml

fake-benchmark:
	$(PYTHON) -m budgetroute benchmark --config configs/benchmarks/fake-smoke.yaml

report:
	$(PYTHON) -m budgetroute generate-report --latest

serve-fake:
	$(PYTHON) -m budgetroute serve --config configs/serving/fake.yaml

build:
	$(PYTHON) -m build

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('build'), pathlib.Path('dist'), pathlib.Path('.pytest_cache'), pathlib.Path('.mypy_cache'), pathlib.Path('.ruff_cache')]]"

