from __future__ import annotations

import json
from pathlib import Path

from budgetroute.config import AppConfig
from budgetroute.experiments.runner import run_benchmark
from budgetroute.inference.service import build_service
from budgetroute.routing.training import evaluate_router, train_router
from budgetroute.schemas import GenerationRequest, RouteName


def test_benchmark_report_and_fake_router(
    fake_config: AppConfig, tmp_path: Path, project_root: Path
) -> None:
    benchmark_config = fake_config.model_copy(
        update={
            "output_dir": tmp_path / "outputs",
            "benchmark": fake_config.benchmark.model_copy(
                update={"policies": ["always_small", "heuristic", "cascade"], "fake": True}
            ),
        }
    )
    run_dir = run_benchmark(benchmark_config, project_root)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["fake"] is True
    assert metrics["failed_requests"] == 0
    assert (run_dir / "report/figures/quality-latency.png").is_file()
    assert "NOT REAL MODEL PERFORMANCE" in (run_dir / "report/report.md").read_text(
        encoding="utf-8"
    )

    router_path = tmp_path / "router.joblib"
    metadata = train_router(run_dir, router_path, quality_threshold=0.8)
    assert router_path.is_file()
    assert metadata["label_definition"] == "small_model_quality >= 0.8"
    evaluation = evaluate_router(router_path, run_dir, threshold=0.8)
    assert 0 <= evaluation["accuracy"] <= 1

    learned_config = fake_config.model_copy(
        update={
            "routing": fake_config.routing.model_copy(
                update={"policy": "learned", "learned_model_path": router_path}
            )
        }
    )
    service = build_service(learned_config)
    try:
        response = service.generate(GenerationRequest(prompt="What is the capital of France?"))
        assert response.router.policy == "learned"
        assert response.route in {RouteName.SMALL, RouteName.LARGE}
    finally:
        service.close()
