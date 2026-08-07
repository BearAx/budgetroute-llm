from __future__ import annotations

import json
from pathlib import Path

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
        config_response = client.get("/v1/config").text
        assert "fail_on_substrings" not in config_response
        assert "prompt_blocklist" not in config_response
        assert "database_path" not in config_response
        assert "registry_path" not in config_response
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "budgetroute_requests_total 1" in metrics.text
        assert generated.headers["x-content-type-options"] == "nosniff"
        secret_value = "must-not-be-reflected"
        invalid = client.post(
            "/v1/generate",
            json={"prompt": "valid", "max_new_tokens": secret_value},
        )
        assert invalid.status_code == 422
        assert secret_value not in invalid.text


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
        assert feedback.json()["retained_fields"] == ["request_id", "correct"]
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


def test_tenant_scopes_durable_review_and_audit(
    fake_config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    credentials = {
        "tenants": [
            {
                "tenant_id": "team-a",
                "subject": "generator",
                "api_key": "generator-secret-0001",
                "scopes": ["inference"],
            },
            {
                "tenant_id": "team-a",
                "subject": "reviewer",
                "api_key": "reviewer-secret-0001",
                "scopes": ["review"],
            },
            {
                "tenant_id": "operations",
                "subject": "administrator",
                "api_key": "administrator-secret-0001",
                "scopes": ["admin"],
            },
        ]
    }
    monkeypatch.setenv("BUDGETROUTE_TENANTS", json.dumps(credentials))
    secure = fake_config.model_copy(
        update={
            "small_backend": fake_config.small_backend.model_copy(
                update={"input_cost_units_per_1k_tokens": 1.0}
            ),
            "large_backend": fake_config.large_backend.model_copy(
                update={"input_cost_units_per_1k_tokens": 1.0}
            ),
            "routing": fake_config.routing.model_copy(
                update={
                    "policy": "budget_aware",
                    "max_estimated_cost_units": 0.0,
                    "human_review_enabled": True,
                    "human_review_threshold": 0.0,
                }
            ),
            "api": fake_config.api.model_copy(
                update={
                    "require_api_key": True,
                    "tenant_keys_env": "BUDGETROUTE_TENANTS",
                    "review_enabled": True,
                }
            ),
            "operations": fake_config.operations.model_copy(
                update={"backend": "sqlite", "database_path": tmp_path / "operations.db"}
            ),
        }
    )
    generator_headers = {"Authorization": "Bearer generator-secret-0001"}
    reviewer_headers = {"Authorization": "Bearer reviewer-secret-0001"}
    admin_headers = {"Authorization": "Bearer administrator-secret-0001"}
    with TestClient(create_app(secure)) as client:
        response = client.post(
            "/v1/generate",
            headers=generator_headers,
            json={"request_id": "review-request-1", "prompt": "A difficult request"},
        )
        assert response.status_code == 200
        assert response.json()["route"] == "human_review"
        assert client.get("/v1/reviews", headers=generator_headers).status_code == 403

        reviews = client.get("/v1/reviews", headers=reviewer_headers).json()
        assert len(reviews) == 1
        case_id = reviews[0]["case_id"]
        claimed = client.post(f"/v1/reviews/{case_id}/claim", headers=reviewer_headers).json()
        resolved = client.post(
            f"/v1/reviews/{case_id}/resolve",
            headers=reviewer_headers,
            json={
                "outcome": "approved",
                "correct": True,
                "expected_version": claimed["version"],
            },
        )
        assert resolved.status_code == 200
        assert resolved.json()["status"] == "resolved"

        audit = client.get("/v1/audit", headers=admin_headers)
        assert audit.status_code == 200
        assert [item["action"] for item in audit.json()["events"]] == [
            "review.created",
            "review.claimed",
            "review.resolved",
        ]
