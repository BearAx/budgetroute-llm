"""Honest Markdown experiment reports."""

from __future__ import annotations

from typing import Any


def _format(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def render_report(metrics: dict[str, Any], run: dict[str, Any]) -> str:
    fake = bool(run.get("fake") or metrics.get("fake"))
    label = "FAKE SMOKE BENCHMARK - NOT REAL MODEL PERFORMANCE" if fake else "REAL EXECUTION REPORT"
    lines = [
        f"# BudgetRoute-LLM report: {run.get('run_id', 'unknown run')}",
        "",
        f"> **{label}**",
        "",
        "This report was generated only from the artifact files in this experiment directory.",
        "",
        "## Run context",
        "",
        f"- Timestamp (UTC): `{run.get('timestamp_utc', 'unknown')}`",
        f"- Dataset hash: `{run.get('dataset_hash', 'unknown')}`",
        f"- Configuration hash: `{run.get('configuration_hash', 'unknown')}`",
        f"- Warm-up runs: {run.get('warmup_runs', 'unknown')}",
        f"- Measured runs: {run.get('measured_runs', 'unknown')}",
        f"- Concurrency: {run.get('concurrency', 'unknown')}",
        f"- Batch size: {run.get('batch_size', 'unknown')}",
        "",
        "## Policy comparison",
        "",
        "| Policy | Requests | Mean quality | p50 ms | p95 ms | Peak RSS MiB | Escalations | Abstentions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    policies = metrics.get("policies", {})
    if not policies:
        lines.append("| No policy results available | 0 | n/a | n/a | n/a | n/a | 0 | 0 |")
    for name, values in sorted(policies.items()):
        latency = values.get("latency", {})
        lines.append(
            f"| {name} | {values.get('request_count', 0)} | {_format(values.get('mean_quality'))} "
            f"| {_format(latency.get('p50_ms'))} | {_format(latency.get('p95_ms'))} "
            f"| {_format(values.get('process_rss_mb_peak'), 1)} "
            f"| {values.get('escalations', 0)} | {values.get('abstentions', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "The bundled sample is designed to validate architecture and metrics, not to establish scientific conclusions. Latency percentiles from small samples are unstable. Compare runs only when hardware, model, warm-up, concurrency, batching, and configuration are compatible.",
            "",
            "## Generated files",
            "",
            "- `metrics.csv`: machine-readable policy summary",
            "- `figures/quality-latency.png`: observed quality/latency trade-off",
            "- `figures/route-distribution.png`: route shares by policy",
            "- `figures/latency-distribution.png`: per-request latency observations",
            "- `figures/calibration.png`: confidence reliability observations",
            "- `figures/threshold-sweep.png`: confidence/coverage/quality sweep",
            "- `figures/routing-confusion-matrix.png`: backend-routing confusion matrix when available",
            "",
        ]
    )
    return "\n".join(lines)


def render_empty_report(reason: str) -> str:
    return (
        "# BudgetRoute-LLM report\n\n"
        "No benchmark metrics are available, so no performance values or figures were invented.\n\n"
        f"Reason: {reason}\n\n"
        "Run `python -m budgetroute benchmark --config configs/benchmarks/fake-smoke.yaml` "
        "for an explicitly fake pipeline smoke test, or use a real-model benchmark configuration.\n"
    )
