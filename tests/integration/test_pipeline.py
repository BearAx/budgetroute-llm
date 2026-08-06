from __future__ import annotations

from budgetroute.config import AppConfig
from budgetroute.inference.service import build_service
from budgetroute.schemas import GenerationRequest, RouteName


def test_fake_request_through_pipeline(fake_config: AppConfig) -> None:
    service = build_service(fake_config)
    try:
        response = service.generate(GenerationRequest(prompt="What is the capital of France?"))
        assert response.answer == "Paris"
        assert response.route == RouteName.SMALL
        assert response.fake
        assert response.timing.total_ms > 0
    finally:
        service.close()


def test_retrieval_aware_route(fake_config: AppConfig) -> None:
    service = build_service(fake_config)
    try:
        response = service.generate(
            GenerationRequest(
                prompt="According to the project corpus, what is the internal project codename?",
                requires_retrieval=True,
                metadata={"fake_reference_answer": "Lantern", "fake_small_success": False},
            )
        )
        assert response.route == RouteName.SMALL_WITH_RETRIEVAL
        assert response.answer == "Lantern"
        assert response.retrieval
        assert response.timing.retrieval_ms > 0
    finally:
        service.close()


def test_cascade_escalation(fake_config: AppConfig) -> None:
    cascade = fake_config.model_copy(
        update={"routing": fake_config.routing.model_copy(update={"policy": "cascade"})}
    )
    service = build_service(cascade)
    try:
        response = service.generate(
            GenerationRequest(
                prompt="A difficult fake prompt",
                metadata={"fake_reference_answer": "large answer", "fake_small_success": False},
            )
        )
        assert response.execution.escalated
        assert response.execution.final_backend == "large"
        assert response.answer == "large answer"
        assert response.timing.escalation_ms > 0
    finally:
        service.close()
