"""Typer command-line interface for local development and experiments."""

from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
from typing import Any

import typer

from budgetroute import __version__
from budgetroute.config import AppConfig, load_config, validate_runtime_config
from budgetroute.environment import collect_environment, writable_directory
from budgetroute.evaluation.dataset import (
    dataset_summary,
    load_dataset,
    materialize_dataset,
    validate_dataset_manifest,
)
from budgetroute.experiments.replay import verify_replay
from budgetroute.experiments.runner import collect_baselines as collect_baseline_run
from budgetroute.experiments.runner import replay_benchmark as run_replay_benchmark
from budgetroute.experiments.runner import run_benchmark
from budgetroute.inference.service import build_service
from budgetroute.reporting.report import generate_report, latest_run_directory
from budgetroute.retrieval.base import EmbeddingProvider
from budgetroute.retrieval.embeddings import FakeEmbedder, TransformersEmbedder
from budgetroute.retrieval.index import ExactCosineIndex, FaissCosineIndex
from budgetroute.retrieval.service import RetrievalService
from budgetroute.routing.training import evaluate_router as evaluate_router_artifact
from budgetroute.routing.training import (
    train_backend_confidence_calibrator as train_confidence_artifact,
)
from budgetroute.routing.training import train_router as train_router_artifact
from budgetroute.schemas import GenerationRequest

app = typer.Typer(
    name="budgetroute",
    help="Quality-aware language-model routing, evaluation, and reporting.",
    no_args_is_help=True,
)


def _print_json(value: Any) -> None:
    typer.echo(json.dumps(value, indent=2, ensure_ascii=False, default=str))


@app.command()
def doctor(
    config: Path = typer.Option(Path("configs/serving/fake.yaml"), exists=False),
) -> None:
    """Inspect Python, optional dependencies, CUDA, configuration, and outputs."""
    environment = collect_environment()
    configuration: dict[str, Any]
    try:
        loaded = load_config(config)
        try:
            validate_runtime_config(loaded)
            runtime: dict[str, Any] = {"valid": True}
        except Exception as exc:
            runtime = {"valid": False, "error": str(exc)}
        configuration = {
            "valid": True,
            "path": str(config),
            "mode": loaded.mode,
            "output_writable": writable_directory(loaded.output_dir),
            "runtime": runtime,
        }
    except Exception as exc:
        configuration = {"valid": False, "path": str(config), "error": str(exc)}
    _print_json({"budgetroute_version": __version__, **environment, "configuration": configuration})


@app.command("validate-config")
def validate_config_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Resolve and validate a YAML configuration."""
    loaded = load_config(config)
    validate_runtime_config(loaded)
    _print_json({"valid": True, "path": str(config), "summary": loaded.sanitized_summary()})


@app.command("security-check")
def security_check(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Audit deployment-sensitive settings without printing secret values."""
    loaded = load_config(config)
    try:
        host_is_loopback = (
            loaded.api.host.lower() == "localhost"
            or ipaddress.ip_address(loaded.api.host).is_loopback
        )
    except ValueError:
        host_is_loopback = False
    checks: list[dict[str, object]] = [
        {
            "name": "non_loopback_authentication",
            "passed": host_is_loopback or loaded.api.require_api_key,
            "detail": "non-loopback bindings must require an API key",
        },
        {
            "name": "api_key_available",
            "passed": not loaded.api.require_api_key
            or bool(os.environ.get(loaded.api.api_key_env)),
            "detail": f"credential source: {loaded.api.api_key_env}",
        },
        {
            "name": "bounded_admission",
            "passed": loaded.batching.max_queue_size > 0 and loaded.batching.request_timeout_ms > 0,
            "detail": "queue capacity and request deadline are configured",
        },
        {
            "name": "trusted_hosts",
            "passed": bool(loaded.api.trusted_hosts) and "*" not in loaded.api.trusted_hosts,
            "detail": "wildcard Host headers are not allowed",
        },
        {
            "name": "repository_security_policy",
            "passed": Path("SECURITY.md").is_file()
            and Path(".github/workflows/security.yml").is_file()
            and Path(".github/dependabot.yml").is_file(),
            "detail": "security policy, scanning workflow, and dependency updates exist",
        },
    ]
    passed = all(bool(check["passed"]) for check in checks)
    _print_json({"passed": passed, "config": str(config), "checks": checks})
    if not passed:
        raise typer.Exit(code=1)


@app.command("inspect-data")
def inspect_data(
    path: Path = typer.Option(..., "--path", exists=True, dir_okay=False),
    manifest: Path | None = typer.Option(None, "--manifest", exists=True, dir_okay=False),
) -> None:
    """Validate a benchmark JSONL file and show category statistics."""
    records = load_dataset(path)
    manifest_value = (
        validate_dataset_manifest(path, manifest).model_dump(mode="json")
        if manifest is not None
        else None
    )
    _print_json(
        {
            "valid": True,
            "path": str(path),
            "manifest": manifest_value,
            **dataset_summary(records),
        }
    )


@app.command("materialize-dataset")
def materialize_dataset_command(
    spec: Path = typer.Option(..., "--spec", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", dir_okay=False),
    limit: int | None = typer.Option(None, min=1),
    offset: int = typer.Option(0, min=0),
    corpus_dir: Path | None = typer.Option(None, "--corpus-dir", file_okay=False),
) -> None:
    """Materialize a pinned public dataset and write its integrity manifest."""
    manifest = materialize_dataset(spec, output, limit=limit, offset=offset, corpus_dir=corpus_dir)
    _print_json(
        {
            "dataset": str(output),
            "manifest": str(output.with_suffix(".manifest.json")),
            "details": manifest.model_dump(mode="json"),
        }
    )


@app.command("build-index")
def build_index(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Build and persist the configured exact cosine retrieval index."""
    loaded = load_config(config)
    if not loaded.retrieval.enabled:
        raise typer.BadParameter("retrieval.enabled must be true")
    if loaded.retrieval.embedder == "fake":
        embedder: EmbeddingProvider = FakeEmbedder(loaded.retrieval.embedding_dimension)
    else:
        assert loaded.retrieval.embedding_model_id is not None
        embedder = TransformersEmbedder(
            loaded.retrieval.embedding_model_id, "cuda" if loaded.mode == "gpu" else "cpu"
        )
    index_class = FaissCosineIndex if loaded.retrieval.index_type == "faiss" else ExactCosineIndex
    service = RetrievalService(loaded.retrieval, index_class(embedder))
    service.initialize(rebuild=True, persist=True)
    _print_json(service.metadata())


@app.command()
def benchmark(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Run configured policies and save predictions, metrics, and a report."""
    run_directory = run_benchmark(load_config(config), Path.cwd())
    _print_json({"run_directory": str(run_directory), "report": str(run_directory / "report")})


@app.command("collect-baselines")
def collect_baselines_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Generate small/large baselines once and populate the content-addressed cache."""
    loaded = load_config(config)
    validate_runtime_config(loaded)
    run_directory = collect_baseline_run(loaded, Path.cwd())
    _print_json(
        {"run_directory": str(run_directory), "cache": str(loaded.benchmark.cache.directory)}
    )


@app.command("replay-benchmark")
def replay_benchmark_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Evaluate policies from cached generations without loading model weights."""
    run_directory = run_replay_benchmark(load_config(config), Path.cwd())
    _print_json({"run_directory": str(run_directory), "replay": True})


@app.command("verify-replay")
def verify_replay_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    sample_size: int = typer.Option(3, min=1, max=1000),
) -> None:
    """Regenerate a sample and verify exact semantic agreement with cache replay."""
    loaded = load_config(config)
    validate_runtime_config(loaded)
    result = verify_replay(loaded, sample_size)
    _print_json(result)
    if result["mismatch_count"]:
        raise typer.Exit(code=1)


@app.command("train-router")
def train_router(
    artifacts: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Option(Path("outputs/router/router.joblib")),
    quality_threshold: float = typer.Option(0.8, min=0.0, max=1.0),
    seed: int = typer.Option(42),
    target_selective_accuracy: float = typer.Option(0.8, min=0.0, max=1.0),
) -> None:
    """Train and persist a learned router from benchmark-derived labels."""
    result = train_router_artifact(
        artifacts, output, quality_threshold, seed, target_selective_accuracy
    )
    _print_json({"artifact": str(output), "training": result})


@app.command("evaluate-router")
def evaluate_router(
    router: Path = typer.Option(..., exists=True, dir_okay=False),
    artifacts: Path = typer.Option(..., exists=True, file_okay=False),
    quality_threshold: float = typer.Option(0.8, min=0.0, max=1.0),
) -> None:
    """Evaluate a learned router's labels, calibration, and route distribution."""
    _print_json(evaluate_router_artifact(router, artifacts, quality_threshold))


@app.command("calibrate-confidence")
def calibrate_confidence(
    artifacts: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Option(Path("outputs/router/small-confidence.json")),
    policy: str = typer.Option("always_small"),
    quality_threshold: float = typer.Option(0.8, min=0.0, max=1.0),
    target_selective_accuracy: float = typer.Option(0.8, min=0.0, max=1.0),
    seed: int = typer.Option(42),
) -> None:
    """Fit a portable correctness calibrator for small-backend confidence."""
    result = train_confidence_artifact(
        artifacts,
        output,
        policy=policy,
        quality_threshold=quality_threshold,
        target_selective_accuracy=target_selective_accuracy,
        seed=seed,
    )
    _print_json({"artifact": str(output), "calibration": result})


@app.command("generate-report")
def generate_report_command(
    run_directory: Path | None = typer.Option(None, "--run-directory", file_okay=False),
    latest: bool = typer.Option(False, "--latest"),
    output_root: Path = typer.Option(Path("outputs"), "--output-root"),
) -> None:
    """Generate Markdown, CSV, and PNG files from existing experiment artifacts."""
    selected = latest_run_directory(output_root) if latest else run_directory
    if selected is None:
        raise typer.BadParameter("no run directory was supplied or found")
    report = generate_report(selected)
    _print_json({"report_directory": str(report)})


@app.command()
def serve(
    config: Path = typer.Option(Path("configs/serving/fake.yaml"), exists=True),
    host: str | None = typer.Option(None),
    port: int | None = typer.Option(None, min=1, max=65535),
) -> None:
    """Start the FastAPI service for the selected fake, CPU, or GPU configuration."""
    try:
        import uvicorn

        from budgetroute.api.app import create_app
    except ImportError as exc:
        raise typer.BadParameter("serve requires `pip install -e .[api]`") from exc
    loaded = load_config(config)
    if host is not None or port is not None:
        data = loaded.model_dump(mode="python")
        data["api"] = {
            **loaded.api.model_dump(mode="python"),
            "host": host or loaded.api.host,
            "port": port or loaded.api.port,
        }
        loaded = AppConfig.model_validate(data)
    application = create_app(loaded)
    uvicorn.run(
        application,
        host=loaded.api.host,
        port=loaded.api.port,
        log_level=loaded.api.log_level,
    )


def _demo_case(label: str, config: AppConfig, request: GenerationRequest) -> None:
    service = build_service(config)
    try:
        service.initialize()
        response = service.generate(request)
        typer.echo(f"\n--- {label} ---")
        _print_json(response.model_dump(mode="json", exclude_none=True))
    finally:
        service.close()


@app.command()
def demo(
    config: Path = typer.Option(Path("configs/serving/fake.yaml"), exists=True),
) -> None:
    """Run five deterministic fake scenarios; output is not a real benchmark."""
    loaded = load_config(config)
    if loaded.mode != "fake":
        raise typer.BadParameter("demo requires a fake-mode configuration")
    typer.echo("FAKE MODE DEMO — outputs and timing are simulated, not model benchmark results.")
    heuristic = loaded.model_copy(
        update={"routing": loaded.routing.model_copy(update={"policy": "heuristic"})}
    )
    _demo_case(
        "easy -> small", heuristic, GenerationRequest(prompt="What is the capital of France?")
    )
    _demo_case(
        "difficult -> large",
        heuristic,
        GenerationRequest(
            prompt="Calculate the equation 17 * 6 + 5 step by step.",
            metadata={"fake_reference_answer": "107", "fake_small_success": False},
        ),
    )
    _demo_case(
        "retrieval -> small with context",
        heuristic,
        GenerationRequest(
            prompt="According to the project corpus, what is the internal project codename?",
            requires_retrieval=True,
            metadata={"fake_reference_answer": "Lantern", "fake_small_success": False},
        ),
    )
    cascade = loaded.model_copy(
        update={"routing": loaded.routing.model_copy(update={"policy": "cascade"})}
    )
    _demo_case(
        "cascade -> escalate",
        cascade,
        GenerationRequest(
            prompt="A difficult fake request that should escalate.",
            metadata={"fake_reference_answer": "Escalated answer", "fake_small_success": False},
        ),
    )
    _demo_case(
        "insufficient context -> abstain",
        heuristic,
        GenerationRequest(
            prompt="This is ambiguous and unanswerable without missing context.",
            metadata={"must_abstain": True},
        ),
    )
