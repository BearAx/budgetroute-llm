"""Typed YAML configuration loading, composition, and operational overrides."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from budgetroute.exceptions import ConfigurationError


class GenerationConfig(BaseModel):
    max_new_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)


class BackendConfig(BaseModel):
    type: Literal["fake", "transformers"] = "fake"
    model_id: str | None = None
    tokenizer_id: str | None = None
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    quantization: Literal["none", "int8", "int4"] = "none"
    compile: bool = False
    trust_remote_code: bool = False
    artificial_latency_ms: float = Field(default=0.0, ge=0.0)
    fail_on_substrings: list[str] = Field(default_factory=list)
    quality: float = Field(default=0.8, ge=0.0, le=1.0)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    @model_validator(mode="after")
    def validate_backend(self) -> BackendConfig:
        if self.type == "transformers" and not self.model_id:
            raise ValueError("transformers backend requires model_id")
        if self.type == "fake" and self.quantization != "none":
            raise ValueError("quantization is only valid for transformers backends")
        if self.device == "cpu" and self.precision == "fp16":
            raise ValueError("fp16 on CPU is unsupported; use fp32 or a supported bf16 setup")
        if self.device == "cpu" and self.quantization != "none":
            raise ValueError("quantization configuration currently requires a CUDA device")
        return self


class RetrievalConfig(BaseModel):
    enabled: bool = False
    corpus_path: Path = Path("data/sample_corpus")
    index_path: Path = Path("data/indexes/sample-index.npz")
    embedder: Literal["fake", "transformers"] = "fake"
    index_type: Literal["exact", "faiss"] = "exact"
    embedding_model_id: str | None = None
    embedding_dimension: int = Field(default=64, ge=8, le=4096)
    chunk_size: int = Field(default=500, ge=32)
    chunk_overlap: int = Field(default=50, ge=0)
    top_k: int = Field(default=3, ge=1, le=50)
    min_similarity: float = Field(default=0.12, ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def validate_retrieval(self) -> RetrievalConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("retrieval chunk_overlap must be smaller than chunk_size")
        if self.embedder == "transformers" and not self.embedding_model_id:
            raise ValueError("transformers embedder requires embedding_model_id")
        return self


class RoutingConfig(BaseModel):
    policy: Literal[
        "always_small",
        "always_large",
        "random",
        "heuristic",
        "learned",
        "retrieval_first",
        "cascade",
    ] = "heuristic"
    seed: int = 42
    small_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    large_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    difficulty_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    abstain_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    cascade_confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    retrieval_similarity_threshold: float = Field(default=0.12, ge=-1.0, le=1.0)
    learned_model_path: Path | None = None
    retain_initial_answer: bool = True

    @model_validator(mode="after")
    def validate_probabilities(self) -> RoutingConfig:
        if self.policy == "random" and self.small_probability + self.large_probability <= 0:
            raise ValueError("random routing probabilities must sum to more than zero")
        if self.policy == "learned" and self.learned_model_path is None:
            raise ValueError("learned routing requires learned_model_path")
        return self


class BatchingConfig(BaseModel):
    enabled: bool = False
    max_batch_size: int = Field(default=8, ge=1, le=1024)
    max_wait_ms: float = Field(default=10.0, ge=0.0, le=10_000.0)


class BenchmarkConfig(BaseModel):
    dataset_path: Path = Path("data/sample_benchmark.jsonl")
    policies: list[str] = Field(default_factory=lambda: ["always_small", "heuristic", "cascade"])
    warmup_runs: int = Field(default=1, ge=0, le=100)
    measured_runs: int = Field(default=1, ge=1, le=10_000)
    concurrency: int = Field(default=1, ge=1, le=1024)
    batch_size: int = Field(default=1, ge=1, le=1024)
    quality_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    fake: bool = False


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "info"
    max_prompt_chars: int = Field(default=50_000, ge=1)


class AppConfig(BaseModel):
    name: str = "budgetroute"
    mode: Literal["fake", "cpu", "gpu"] = "fake"
    seed: int = 42
    output_dir: Path = Path("outputs")
    small_backend: BackendConfig = Field(default_factory=BackendConfig)
    large_backend: BackendConfig = Field(default_factory=lambda: BackendConfig(quality=0.95))
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    batching: BatchingConfig = Field(default_factory=BatchingConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)

    @model_validator(mode="after")
    def validate_composition(self) -> AppConfig:
        retrieval_policies = {"retrieval_first"}
        if self.routing.policy in retrieval_policies and not self.retrieval.enabled:
            raise ValueError(f"{self.routing.policy} routing requires retrieval.enabled=true")
        if self.mode == "fake" and (
            self.small_backend.type != "fake" or self.large_backend.type != "fake"
        ):
            raise ValueError("fake mode requires both backends to use type=fake")
        return self

    def sanitized_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "small_backend": {
                "type": self.small_backend.type,
                "model_id": self.small_backend.model_id,
                "device": self.small_backend.device,
                "precision": self.small_backend.precision,
            },
            "large_backend": {
                "type": self.large_backend.type,
                "model_id": self.large_backend.model_id,
                "device": self.large_backend.device,
                "precision": self.large_backend.precision,
            },
            "retrieval": self.retrieval.model_dump(mode="json"),
            "routing": self.routing.model_dump(mode="json"),
            "batching": self.batching.model_dump(mode="json"),
        }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_include(include: str, current_file: Path) -> Path:
    candidate = Path(include)
    if candidate.is_absolute():
        return candidate
    relative = (current_file.parent / candidate).resolve()
    if relative.exists():
        return relative
    return (Path.cwd() / candidate).resolve()


def _load_yaml_tree(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    seen = set() if seen is None else seen
    if resolved in seen:
        raise ConfigurationError(f"cyclic configuration include detected at {resolved}")
    if not resolved.is_file():
        raise ConfigurationError(f"configuration file does not exist: {resolved}")
    seen.add(resolved)
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read YAML configuration {resolved}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"configuration root must be a mapping: {resolved}")
    includes = raw.pop("extends", [])
    if isinstance(includes, str):
        includes = [includes]
    if not isinstance(includes, list) or not all(isinstance(item, str) for item in includes):
        raise ConfigurationError("extends must be a string or list of YAML paths")
    merged: dict[str, Any] = {}
    for include in includes:
        merged = _deep_merge(merged, _load_yaml_tree(_resolve_include(include, resolved), seen))
    seen.remove(resolved)
    return _deep_merge(merged, raw)


def _apply_environment(data: dict[str, Any]) -> dict[str, Any]:
    result = dict(data)
    mappings: dict[str, tuple[str, ...]] = {
        "BUDGETROUTE_OUTPUT_DIR": ("output_dir",),
        "BUDGETROUTE_HOST": ("api", "host"),
        "BUDGETROUTE_PORT": ("api", "port"),
        "BUDGETROUTE_LOG_LEVEL": ("api", "log_level"),
    }
    for variable, path in mappings.items():
        value = os.environ.get(variable)
        if value is None:
            continue
        cursor = result
        for part in path[:-1]:
            nested = cursor.setdefault(part, {})
            if not isinstance(nested, dict):
                raise ConfigurationError(f"cannot apply {variable}: {part} is not a mapping")
            cursor = nested
        cursor[path[-1]] = int(value) if variable == "BUDGETROUTE_PORT" else value
    return result


def load_config(path: str | Path, *, apply_environment: bool = True) -> AppConfig:
    data = _load_yaml_tree(Path(path))
    if apply_environment:
        data = _apply_environment(data)
    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(f"invalid configuration {path}:\n{exc}") from exc


def dump_resolved_config(config: AppConfig) -> str:
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True)


def validate_runtime_config(config: AppConfig) -> None:
    """Reject explicit optional-runtime requests that this environment cannot honor."""
    transformer_backends = [
        backend
        for backend in (config.small_backend, config.large_backend)
        if backend.type == "transformers"
    ]
    if transformer_backends and importlib.util.find_spec("torch") is None:
        raise ConfigurationError(
            "Transformers execution is configured but PyTorch is not installed; "
            "install the transformers extra"
        )
    if transformer_backends:
        import torch

        if (
            any(backend.device == "cuda" for backend in transformer_backends)
            and not torch.cuda.is_available()
        ):
            raise ConfigurationError(
                "CUDA is explicitly configured but torch.cuda.is_available() is false"
            )
        if any(
            backend.device == "cuda" and backend.precision == "bf16"
            for backend in transformer_backends
        ) and not bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
            raise ConfigurationError("BF16 is configured but the CUDA device does not support it")
        if (
            any(backend.quantization != "none" for backend in transformer_backends)
            and importlib.util.find_spec("bitsandbytes") is None
        ):
            raise ConfigurationError("quantization is configured but bitsandbytes is not installed")
    if config.routing.policy == "learned":
        assert config.routing.learned_model_path is not None
        if not config.routing.learned_model_path.is_file():
            raise ConfigurationError(
                f"learned router artifact does not exist: {config.routing.learned_model_path}"
            )
    if config.retrieval.enabled:
        if config.retrieval.index_type == "faiss" and importlib.util.find_spec("faiss") is None:
            raise ConfigurationError("FAISS index requested but faiss is not installed")
        if not config.retrieval.index_path.is_file() and not config.retrieval.corpus_path.exists():
            raise ConfigurationError(
                "retrieval is enabled but neither a saved index nor corpus path is available"
            )
