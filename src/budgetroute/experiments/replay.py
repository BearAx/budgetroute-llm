"""Live-versus-cache agreement validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from budgetroute.config import AppConfig
from budgetroute.evaluation.dataset import load_dataset
from budgetroute.evaluation.harness import request_for_record
from budgetroute.inference.service import build_service
from budgetroute.schemas import GenerationResponse


def response_signature(response: GenerationResponse) -> dict[str, Any]:
    """Fields that cache replay must reproduce exactly; timings are replay estimates."""

    return {
        "request_id": response.request_id,
        "answer": response.answer,
        "route": response.route.value,
        "router_policy": response.router.policy,
        "execution": response.execution.model_dump(mode="json"),
        "retrieval": [item.model_dump(mode="json") for item in response.retrieval],
        "usage": response.usage.model_dump(mode="json"),
        "backend_confidence": response.backend_confidence,
        "backend_confidence_raw": response.backend_confidence_raw,
        "backend_confidence_calibrated": response.backend_confidence_calibrated,
        "confidence_signals": (
            response.confidence_signals.model_dump(mode="json")
            if response.confidence_signals is not None
            else None
        ),
        "fake": response.fake,
    }


def verify_replay(config: AppConfig, sample_size: int = 3) -> dict[str, Any]:
    if sample_size < 1:
        raise ValueError("replay verification sample_size must be at least one")
    records = load_dataset(config.benchmark.dataset_path)[:sample_size]
    comparisons: list[dict[str, Any]] = []
    for policy_name in config.benchmark.policies:
        routing = config.routing.model_copy(update={"policy": policy_name})
        live_benchmark = config.benchmark.model_copy(
            update={"cache": config.benchmark.cache.model_copy(update={"mode": "refresh"})}
        )
        replay_benchmark = config.benchmark.model_copy(
            update={"cache": config.benchmark.cache.model_copy(update={"mode": "read_only"})}
        )
        live_service = build_service(
            config.model_copy(update={"routing": routing, "benchmark": live_benchmark})
        )
        replay_service = build_service(
            config.model_copy(update={"routing": routing, "benchmark": replay_benchmark})
        )
        try:
            live_service.initialize()
            for record in records:
                request = request_for_record(record, fake=config.benchmark.fake)
                live = live_service.generate(request)
                replay_service.initialize()
                replay = replay_service.generate(request)
                live_signature = response_signature(live)
                replay_signature = response_signature(replay)
                comparisons.append(
                    {
                        "policy": policy_name,
                        "example_id": record.id,
                        "match": live_signature == replay_signature,
                        "live": live_signature if live_signature != replay_signature else None,
                        "replay": replay_signature if live_signature != replay_signature else None,
                    }
                )
        finally:
            live_service.close()
            replay_service.close()
    mismatches = [item for item in comparisons if not item["match"]]
    return {
        "sample_size": len(records),
        "comparison_count": len(comparisons),
        "matches": len(comparisons) - len(mismatches),
        "mismatch_count": len(mismatches),
        "agreement": (len(comparisons) - len(mismatches)) / len(comparisons)
        if comparisons
        else None,
        "mismatches": mismatches,
        "cache_directory": str(Path(config.benchmark.cache.directory).resolve()),
    }
