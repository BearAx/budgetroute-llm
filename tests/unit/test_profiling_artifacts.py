from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from budgetroute.experiments.artifacts import ArtifactWriter, create_run_directory
from budgetroute.profiling.latency import summarize_latencies
from budgetroute.reporting.report import generate_report, latest_run_directory


def test_latency_summary_and_percentile_warning() -> None:
    summary = summarize_latencies([3.0, 1.0, 2.0])
    assert summary.p50_ms == 2.0
    assert summary.p95_ms is not None and summary.p95_ms >= summary.p50_ms
    assert summary.p99_ms is None
    assert summary.warning is not None


def test_artifact_creation_and_empty_report(tmp_path: Path) -> None:
    run_id, run_dir = create_run_directory(tmp_path, "fake")
    writer = ArtifactWriter(run_dir)
    writer.write_json("run.json", {"run_id": run_id})
    assert json.loads((run_dir / "run.json").read_text())["run_id"] == run_id
    report_dir = generate_report(run_dir)
    report = (report_dir / "report.md").read_text(encoding="utf-8")
    assert "no performance values or figures were invented" in report


def test_latest_run_ignores_non_experiment_directories(tmp_path: Path) -> None:
    (tmp_path / "router").mkdir()
    run = tmp_path / "20260101T000000Z_fake-smoke"
    run.mkdir()
    (run / "run.json").write_text("{}", encoding="utf-8")
    (run / "metrics.json").write_text("{}", encoding="utf-8")
    assert latest_run_directory(tmp_path) == run


def test_reporting_imports_in_a_clean_process(project_root: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from budgetroute.reporting.report import latest_run_directory; print('ok')",
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
