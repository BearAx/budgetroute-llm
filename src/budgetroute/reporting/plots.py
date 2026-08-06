"""Headless Matplotlib figures built from experiment artifact values."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def generate_figures(
    metrics: dict[str, Any],
    timings: list[dict[str, Any]],
    destination: Path,
    predictions: list[dict[str, Any]] | None = None,
) -> list[Path]:
    if importlib.util.find_spec("matplotlib") is None:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    policies = metrics.get("policies", {})
    if policies:
        names = sorted(policies)
        qualities = [policies[name].get("mean_quality") or 0.0 for name in names]
        latencies = [policies[name].get("latency", {}).get("p50_ms") or 0.0 for name in names]
        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.scatter(latencies, qualities)
        for name, x, y in zip(names, latencies, qualities, strict=True):
            axis.annotate(name, (x, y), xytext=(4, 4), textcoords="offset points")
        frontier_x: list[float] = []
        frontier_y: list[float] = []
        best_quality = -1.0
        for latency, policy_quality in sorted(zip(latencies, qualities, strict=True)):
            if policy_quality > best_quality:
                frontier_x.append(latency)
                frontier_y.append(policy_quality)
                best_quality = policy_quality
        if len(frontier_x) > 1:
            axis.plot(
                frontier_x,
                frontier_y,
                linestyle="--",
                alpha=0.6,
                label="Pareto frontier",
            )
            axis.legend(fontsize="small")
        axis.set_xlabel("p50 latency (ms)")
        axis.set_ylabel("Mean deterministic quality")
        axis.set_title("Observed quality-latency trade-off")
        axis.grid(alpha=0.25)
        path = destination / "quality-latency.png"
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        created.append(path)

        all_routes = sorted(
            {route for values in policies.values() for route in values.get("route_counts", {})}
        )
        figure, axis = plt.subplots(figsize=(8, 4.5))
        bottoms = [0.0] * len(names)
        for route in all_routes:
            shares = [
                policies[name].get("route_distribution", {}).get(route, 0.0) for name in names
            ]
            axis.bar(names, shares, bottom=bottoms, label=route)
            bottoms = [bottom + share for bottom, share in zip(bottoms, shares, strict=True)]
        axis.set_ylim(0, 1)
        axis.set_ylabel("Request share")
        axis.set_title("Route distribution")
        axis.legend(loc="upper right", fontsize="small")
        axis.tick_params(axis="x", rotation=25)
        path = destination / "route-distribution.png"
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        created.append(path)

    values = [float(item["total_ms"]) for item in timings if item.get("total_ms") is not None]
    if values:
        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.hist(values, bins=min(20, max(5, len(values))), edgecolor="white")
        axis.set_xlabel("Total latency (ms)")
        axis.set_ylabel("Request count")
        axis.set_title("Latency observations")
        path = destination / "latency-distribution.png"
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        created.append(path)

    reliability = metrics.get("reliability", [])
    populated = [item for item in reliability if item.get("count")]
    if populated:
        confidence = [float(item["mean_confidence"]) for item in populated]
        accuracy = [float(item["accuracy"]) for item in populated]
        figure, axis = plt.subplots(figsize=(5, 5))
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
        axis.plot(confidence, accuracy, marker="o", label="observed")
        axis.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Mean confidence",
            ylabel="Accuracy",
        )
        axis.set_title("Reliability diagram")
        axis.legend()
        path = destination / "calibration.png"
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        created.append(path)

    usable_predictions = [item for item in (predictions or []) if not item.get("error")]
    if usable_predictions:
        thresholds = [index / 20 for index in range(21)]
        coverage: list[float] = []
        selective_quality: list[float] = []
        for threshold in thresholds:
            selected = [
                item
                for item in usable_predictions
                if float(item.get("confidence", 0.0)) >= threshold
            ]
            coverage.append(len(selected) / len(usable_predictions))
            selective_quality.append(
                sum(float(item["quality_score"]) for item in selected) / len(selected)
                if selected
                else float("nan")
            )
        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.plot(thresholds, coverage, label="coverage")
        axis.plot(thresholds, selective_quality, label="selective quality")
        axis.set(
            xlabel="Confidence threshold",
            ylabel="Observed fraction / quality",
            ylim=(0, 1.05),
        )
        axis.set_title("Confidence threshold sweep")
        axis.legend()
        path = destination / "threshold-sweep.png"
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        created.append(path)

    routing_values = next(
        (
            values.get("routing", {})
            for _, values in sorted(policies.items())
            if values.get("routing", {}).get("confusion_matrix")
        ),
        None,
    )
    if routing_values is not None:
        matrix = routing_values["confusion_matrix"]
        figure, axis = plt.subplots(figsize=(5, 4.5))
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row][column]), ha="center", va="center")
        axis.set_xticks([0, 1], labels=["predict large", "predict small"])
        axis.set_yticks([0, 1], labels=["needs large", "small succeeds"])
        axis.set_title("Routing confusion matrix")
        figure.colorbar(image, ax=axis)
        path = destination / "routing-confusion-matrix.png"
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        created.append(path)
    return created
