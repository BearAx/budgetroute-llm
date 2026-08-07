"""Operator-invoked runtime compatibility probes with immutable evidence."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from budgetroute.config import dump_resolved_config, load_config, validate_runtime_config
from budgetroute.environment import collect_environment
from budgetroute.experiments.artifacts import ArtifactWriter, create_run_directory
from budgetroute.inference.service import build_service
from budgetroute.schemas import GenerationRequest


def run_compatibility_matrix(config_paths: list[Path], output_root: Path) -> Path:
    if not config_paths:
        raise ValueError("at least one configuration is required")
    run_id, run_directory = create_run_directory(output_root, "compatibility")
    writer = ArtifactWriter(run_directory)
    results: list[dict[str, Any]] = []
    source_material: list[bytes] = []
    for path in config_paths:
        started = time.perf_counter_ns()
        result: dict[str, Any] = {"config_path": str(path), "status": "failed"}
        service = None
        try:
            config = load_config(path)
            resolved = dump_resolved_config(config).encode("utf-8")
            source_material.append(resolved)
            result["resolved_config_hash"] = hashlib.sha256(resolved).hexdigest()
            result["config"] = config.sanitized_summary()
            validate_runtime_config(config)
            service = build_service(config)
            service.initialize()
            request = GenerationRequest(
                prompt="Compatibility probe: return the word ready.",
                metadata={
                    "fake_reference_answer": "ready",
                    "fake_small_success": True,
                },
            )
            single = service.generate(request)
            batch = service.generate_batch(
                [
                    request.model_copy(update={"request_id": "compatibility-batch-1"}),
                    request.model_copy(update={"request_id": "compatibility-batch-2"}),
                ]
            )
            result.update(
                {
                    "status": "passed",
                    "single": {
                        "route": single.route.value,
                        "backend": (
                            single.execution.final_backend.value
                            if single.execution.final_backend is not None
                            else None
                        ),
                        "fake": single.fake,
                        "confidence_method": (
                            single.confidence_signals.method
                            if single.confidence_signals is not None
                            else None
                        ),
                    },
                    "batch": {
                        "request_count": len(batch),
                        "order_preserved": [item.request_id for item in batch]
                        == ["compatibility-batch-1", "compatibility-batch-2"],
                        "reported_batch_sizes": [item.timing.batch_size for item in batch],
                    },
                    "metadata": service.metadata(),
                }
            )
        except Exception as exc:
            if len(source_material) < len(results) + 1:
                try:
                    source_material.append(path.read_bytes())
                except OSError:
                    source_material.append(str(path).encode("utf-8"))
            result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            if service is not None:
                service.close()
        result["probe_wall_ms"] = (time.perf_counter_ns() - started) / 1_000_000
        results.append(result)
    source_hash = hashlib.sha256(
        b"".join(len(item).to_bytes(8, "big") + item for item in source_material)
    ).hexdigest()
    artifact = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_epoch": time.time(),
        "source_hash": source_hash,
        "environment": collect_environment(),
        "results": results,
        "passed": all(result["status"] == "passed" for result in results),
        "claim_boundary": (
            "Compatibility probes validate only these exact configurations and this environment; "
            "they are not performance comparisons."
        ),
    }
    writer.write_json("compatibility.json", artifact)
    writer.write_json(
        "run.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "kind": "runtime_compatibility",
            "status": "passed" if artifact["passed"] else "completed_with_failures",
            "source_hash": source_hash,
        },
    )
    return run_directory
