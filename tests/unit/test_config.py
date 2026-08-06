from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from budgetroute.config import (
    AppConfig,
    BackendConfig,
    RoutingConfig,
    load_config,
    validate_runtime_config,
)
from budgetroute.exceptions import ConfigurationError


def test_composed_fake_config(fake_config: AppConfig) -> None:
    assert fake_config.mode == "fake"
    assert fake_config.retrieval.enabled
    assert fake_config.small_backend.quality < fake_config.large_backend.quality


def test_environment_override(
    project_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BUDGETROUTE_OUTPUT_DIR", str(tmp_path))
    config = load_config(project_root / "configs/serving/fake.yaml")
    assert config.output_dir == tmp_path


def test_invalid_fp16_cpu_has_actionable_error() -> None:
    with pytest.raises(ValidationError, match="fp16 on CPU"):
        BackendConfig(type="transformers", model_id="local", device="cpu", precision="fp16")


def test_missing_configuration() -> None:
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_config("not-present.yaml")


def test_runtime_validation_rejects_missing_learned_artifact(fake_config: AppConfig) -> None:
    learned = fake_config.model_copy(
        update={
            "routing": RoutingConfig(
                policy="learned", learned_model_path=Path("missing-router.joblib")
            )
        }
    )
    with pytest.raises(ConfigurationError, match="artifact does not exist"):
        validate_runtime_config(learned)
