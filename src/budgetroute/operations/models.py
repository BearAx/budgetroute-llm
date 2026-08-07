"""Typed operational records shared by API and offline control-plane commands."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Principal(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=100)
    subject: str = Field(min_length=1, max_length=200)
    scopes: frozenset[str] = Field(default_factory=frozenset)
    authentication_method: Literal["local", "legacy_api_key", "tenant_api_key"]

    def permits(self, scope: str) -> bool:
        return "admin" in self.scopes or scope in self.scopes


class QuotaDecision(BaseModel):
    allowed: bool
    remaining: int = Field(ge=0)
    retry_after_seconds: float = Field(ge=0.0)


class AdmissionLease(BaseModel):
    lease_id: str
    tenant_id: str
    replica_id: str
    expires_at_epoch: float


class ReviewStatus(StrEnum):
    OPEN = "open"
    CLAIMED = "claimed"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class ReviewOutcome(StrEnum):
    APPROVED = "approved"
    CORRECTED = "corrected"
    REJECTED = "rejected"
    NOT_ACTIONABLE = "not_actionable"


class ReviewCase(BaseModel):
    sequence: int = Field(ge=1)
    case_id: str
    tenant_id: str
    request_id: str
    reason: str
    status: ReviewStatus
    assignee: str | None = None
    outcome: ReviewOutcome | None = None
    correct: bool | None = None
    version: int = Field(ge=1)
    created_at: str
    updated_at: str


class ReviewResolution(BaseModel):
    outcome: ReviewOutcome
    correct: bool | None = None
    expected_version: int | None = Field(default=None, ge=1)


class PredictionObservation(BaseModel):
    tenant_id: str
    request_id: str
    route: str
    backend_confidence_raw: float | None = Field(default=None, ge=0.0, le=1.0)
    backend_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    features: dict[str, float] = Field(default_factory=dict)
    created_at_epoch: float | None = None


class LabeledObservation(BaseModel):
    tenant_id: str
    request_id: str
    route: str
    backend_confidence_raw: float
    backend_confidence: float | None = None
    correct: bool
    features: dict[str, float] = Field(default_factory=dict)
    predicted_at_epoch: float
    labeled_at_epoch: float


class AuditEvent(BaseModel):
    sequence: int = Field(ge=1)
    event_id: str
    occurred_at: str
    tenant_id: str
    actor: str
    action: str
    target_type: str
    target_id: str
    details: dict[str, Any] = Field(default_factory=dict)
    previous_hash: str
    event_hash: str
