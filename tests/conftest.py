from __future__ import annotations

from pathlib import Path

import pytest

from budgetroute.config import AppConfig, load_config


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_config(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    monkeypatch.chdir(project_root)
    return load_config(project_root / "configs/serving/fake.yaml", apply_environment=False)
