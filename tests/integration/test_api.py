from __future__ import annotations

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
