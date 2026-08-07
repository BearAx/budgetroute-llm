from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from budgetroute.config import (
    AppConfig,
    BackendConfig,
    BenchmarkConfig,
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


def test_real_benchmark_can_require_revision_pins() -> None:
    backend = BackendConfig(type="transformers", model_id="local", device="cpu")
    with pytest.raises(ValidationError, match="requires pinned model revisions"):
        AppConfig(
            mode="cpu",
            small_backend=backend,
            large_backend=backend,
            benchmark=BenchmarkConfig(fake=False, require_pinned_revisions=True),
        )


def test_non_loopback_api_requires_authentication(fake_config: AppConfig) -> None:
    with pytest.raises(ValidationError, match="non-loopback API binding requires"):
        AppConfig.model_validate(
            {
                **fake_config.model_dump(mode="python"),
                "api": {**fake_config.api.model_dump(), "host": "0.0.0.0"},
            }
        )


def test_non_loopback_api_requires_an_explicit_tls_boundary(fake_config: AppConfig) -> None:
    with pytest.raises(ValidationError, match="built-in TLS or"):
        AppConfig.model_validate(
            {
                **fake_config.model_dump(mode="python"),
                "api": {
                    **fake_config.api.model_dump(),
                    "host": "0.0.0.0",
                    "require_api_key": True,
                },
            }
        )
    valid = AppConfig.model_validate(
        {
            **fake_config.model_dump(mode="python"),
            "api": {
                **fake_config.api.model_dump(),
                "host": "0.0.0.0",
                "require_api_key": True,
                "external_tls_termination": True,
            },
        }
    )
    assert valid.api.external_tls_termination is True
