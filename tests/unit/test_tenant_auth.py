from __future__ import annotations

import json

import pytest

from budgetroute.config import ApiConfig
from budgetroute.exceptions import ConfigurationError
from budgetroute.operations.auth import TenantAuthenticator, required_scope


def test_tenant_authentication_and_scopes_use_environment_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "tenants": [
            {
                "tenant_id": "team-a",
                "subject": "automation-a",
                "api_key": "tenant-secret-0001",
                "scopes": ["inference", "feedback"],
            }
        ]
    }
    monkeypatch.setenv("BUDGETROUTE_TENANTS", json.dumps(payload))
    monkeypatch.setenv("BUDGETROUTE_API_KEY", "inherited-legacy-secret")
    authenticator = TenantAuthenticator(
        ApiConfig(require_api_key=True, tenant_keys_env="BUDGETROUTE_TENANTS")
    )

    principal = authenticator.authenticate("tenant-secret-0001")
    assert principal is not None
    assert principal.tenant_id == "team-a"
    assert principal.permits("inference") is True
    assert principal.permits("review") is False
    assert authenticator.authenticate("wrong-secret") is None
    assert authenticator.authenticate("inherited-legacy-secret") is None
    assert authenticator.sanitized_summary() == {
        "required": True,
        "credential_count": 1,
        "tenant_count": 1,
    }
    assert "tenant-secret-0001" not in repr(authenticator.__dict__)


def test_invalid_tenant_credential_document_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "BUDGETROUTE_TENANTS",
        json.dumps({"tenants": [{"tenant_id": "team-a", "api_key": "short"}]}),
    )
    with pytest.raises(ConfigurationError, match="at least 16"):
        TenantAuthenticator(ApiConfig(require_api_key=True, tenant_keys_env="BUDGETROUTE_TENANTS"))


def test_endpoint_scope_mapping() -> None:
    assert required_scope("GET", "/healthz") is None
    assert required_scope("POST", "/v1/generate") == "inference"
    assert required_scope("POST", "/v1/feedback") == "feedback"
    assert required_scope("GET", "/v1/reviews") == "review"
    assert required_scope("GET", "/metrics") == "admin"
