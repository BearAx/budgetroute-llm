"""Environment-only tenant credentials and scope authorization."""

from __future__ import annotations

import json
import os
import re
import secrets
from dataclasses import dataclass, field
from typing import Any, Literal

from budgetroute.config import ApiConfig
from budgetroute.exceptions import ConfigurationError
from budgetroute.operations.models import Principal

_VALID_SCOPES = frozenset({"inference", "feedback", "review", "admin"})


@dataclass(frozen=True)
class _Credential:
    tenant_id: str
    subject: str
    scopes: frozenset[str]
    secret: str = field(repr=False)


class TenantAuthenticator:
    """Verify legacy or tenant keys without exposing them to configuration or audit state."""

    def __init__(self, config: ApiConfig) -> None:
        self.required = config.require_api_key
        self._credentials: list[_Credential] = []
        # Tenant mode is exclusive so an inherited legacy environment variable cannot
        # accidentally become a global all-scopes bypass.
        legacy = os.environ.get(config.api_key_env) if config.tenant_keys_env is None else None
        if legacy:
            self._credentials.append(
                _Credential(
                    tenant_id="legacy",
                    subject="legacy-api-key",
                    scopes=_VALID_SCOPES,
                    secret=legacy,
                )
            )
        if config.tenant_keys_env:
            raw = os.environ.get(config.tenant_keys_env)
            if raw:
                self._credentials.extend(self._parse_tenants(raw, config.tenant_keys_env))
        if self.required and not self._credentials:
            raise ConfigurationError("authentication is required but no API credentials are loaded")

    @staticmethod
    def _parse_tenants(raw: str, source: str) -> list[_Credential]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"{source} must contain valid JSON") from exc
        tenants = payload.get("tenants") if isinstance(payload, dict) else None
        if not isinstance(tenants, list) or not tenants:
            raise ConfigurationError(f"{source} must contain a non-empty tenants list")
        credentials: list[_Credential] = []
        seen_subjects: set[tuple[str, str]] = set()
        seen_secrets: set[str] = set()
        for index, item in enumerate(tenants):
            if not isinstance(item, dict):
                raise ConfigurationError(f"{source} tenant {index} must be an object")
            tenant_id = item.get("tenant_id")
            secret = item.get("api_key")
            subject = item.get("subject", tenant_id)
            scopes_value = item.get("scopes", ["inference"])
            if not isinstance(tenant_id, str) or not re.fullmatch(
                r"[A-Za-z0-9_.-]{1,100}", tenant_id
            ):
                raise ConfigurationError(f"{source} tenant {index} has an invalid tenant_id")
            if not isinstance(subject, str) or not subject or len(subject) > 200:
                raise ConfigurationError(f"{source} tenant {index} has an invalid subject")
            if not isinstance(secret, str) or len(secret) < 16:
                raise ConfigurationError(
                    f"{source} tenant {index} api_key must contain at least 16 characters"
                )
            if not isinstance(scopes_value, list) or not all(
                isinstance(value, str) for value in scopes_value
            ):
                raise ConfigurationError(f"{source} tenant {index} scopes must be a string list")
            scopes = frozenset(scopes_value)
            unknown = scopes - _VALID_SCOPES
            if unknown:
                raise ConfigurationError(
                    f"{source} tenant {index} contains unknown scopes: {sorted(unknown)}"
                )
            identity = (tenant_id, subject)
            if identity in seen_subjects or secret in seen_secrets:
                raise ConfigurationError(f"{source} contains duplicate tenant subjects or API keys")
            seen_subjects.add(identity)
            seen_secrets.add(secret)
            credentials.append(_Credential(tenant_id, subject, scopes, secret))
        return credentials

    def authenticate(self, presented: str | None) -> Principal | None:
        if not self.required:
            return Principal(
                tenant_id="local",
                subject="unauthenticated-loopback",
                scopes=_VALID_SCOPES,
                authentication_method="local",
            )
        candidate = presented or ""
        matched: _Credential | None = None
        for credential in self._credentials:
            if secrets.compare_digest(candidate, credential.secret):
                matched = credential
        if matched is None:
            return None
        method: Literal["legacy_api_key", "tenant_api_key"] = (
            "legacy_api_key" if matched.tenant_id == "legacy" else "tenant_api_key"
        )
        return Principal(
            tenant_id=matched.tenant_id,
            subject=matched.subject,
            scopes=matched.scopes,
            authentication_method=method,
        )

    def sanitized_summary(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "credential_count": len(self._credentials),
            "tenant_count": len(
                {item.tenant_id for item in self._credentials if item.tenant_id != "legacy"}
            ),
        }


def required_scope(method: str, path: str) -> str | None:
    if path in {"/healthz", "/readyz", "/docs", "/openapi.json"}:
        return None
    if path == "/metrics" or path == "/v1/monitoring" or path.startswith("/v1/audit"):
        return "admin"
    if path.startswith("/v1/reviews"):
        return "review"
    if path == "/v1/feedback":
        return "feedback"
    if path.startswith("/v1/"):
        return "inference"
    return None
