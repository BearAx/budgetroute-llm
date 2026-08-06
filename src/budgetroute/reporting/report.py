"""Report orchestration that reads only existing experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from budgetroute.experiments.artifacts import ArtifactWriter
from budgetroute.reporting.markdown import render_empty_report, render_report
from budgetroute.reporting.plots import generate_figures
from budgetroute.reporting.tables import policy_csv


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def generate_report(run_directory: Path) -> Path:
    report_dir = run_directory / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    writer = ArtifactWriter(report_dir)
    metrics_path = run_directory / "metrics.json"
    run_path = run_directory / "run.json"
    if not metrics_path.is_file() or not run_path.is_file():
        writer._atomic_text("report.md", render_empty_report("metrics.json or run.json is missing"))
        writer._atomic_text("metrics.csv", "policy,requests,mean_quality\n")
        return report_dir
    metrics = _read_json(metrics_path)
    run = _read_json(run_path)
    writer._atomic_text("report.md", render_report(metrics, run))
    writer._atomic_text("metrics.csv", policy_csv(metrics))
    timings = _read_jsonl(run_directory / "timings.jsonl")
    predictions = _read_jsonl(run_directory / "predictions.jsonl")
    generate_figures(metrics, timings, report_dir / "figures", predictions)
    return report_dir


def latest_run_directory(output_root: Path) -> Path | None:
    if not output_root.is_dir():
        return None
    candidates = sorted(
        (
            path
            for path in output_root.iterdir()
            if path.is_dir() and (path / "run.json").is_file() and (path / "metrics.json").is_file()
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    return candidates[0] if candidates else None
