from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from budgetroute.api.app import create_app
from budgetroute.config import AppConfig


def test_api_health_readiness_route_and_generate(fake_config: AppConfig) -> None:
    with TestClient(create_app(fake_config)) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        readiness = client.get("/readyz")
        assert readiness.status_code == 200
        assert readiness.json()["ready"] is True
        route = client.post("/v1/route", json={"prompt": "What is the capital of France?"})
        assert route.status_code == 200
        assert route.json()["route"] == "small"
        generated = client.post("/v1/generate", json={"prompt": "What is the capital of France?"})
        assert generated.status_code == 200
        assert generated.json()["answer"] == "Paris"
        assert "initial_answer" not in generated.json()["execution"]
        assert "fail_on_substrings" not in client.get("/v1/config").text
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "budgetroute_requests_total 1" in metrics.text
        assert generated.headers["x-content-type-options"] == "nosniff"


def test_api_authentication_rate_limit_feedback_and_body_limit(
    fake_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BUDGETROUTE_API_KEY", "test-only-secret")
    secure = fake_config.model_copy(
        update={
            "api": fake_config.api.model_copy(
                update={
                    "require_api_key": True,
                    "rate_limit_requests": 3,
                    "max_request_bytes": 1024,
                    "feedback_enabled": True,
                }
            ),
            "batching": fake_config.batching.model_copy(update={"enabled": True}),
        }
    )
    with TestClient(create_app(secure)) as client:
        assert client.get("/v1/config").status_code == 401
        headers = {"Authorization": "Bearer test-only-secret"}
        assert client.get("/v1/config", headers=headers).status_code == 200
        feedback = client.post(
            "/v1/feedback",
            headers=headers,
            json={"request_id": "request-1", "correct": True, "notes": "not retained"},
        )
        assert feedback.status_code == 202
        assert feedback.json()["retained_fields"] == ["correct"]
        assert client.get("/v1/monitoring", headers=headers).status_code == 200
        limited = client.get("/v1/config", headers=headers)
        assert limited.status_code == 429

    body_config = secure.model_copy(
        update={
            "api": secure.api.model_copy(
                update={"rate_limit_requests": 100, "max_request_bytes": 1024}
            )
        }
    )
    with TestClient(create_app(body_config)) as client:
        oversized = client.post(
            "/v1/generate",
            headers={"Authorization": "Bearer test-only-secret"},
            json={"prompt": "x" * 2000},
        )
        assert oversized.status_code == 413
