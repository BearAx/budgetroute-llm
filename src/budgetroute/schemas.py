"""Typed public schemas shared by inference, evaluation, CLI, and API layers."""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class RouteName(StrEnum):
    SMALL = "small"
    LARGE = "large"
    SMALL_WITH_RETRIEVAL = "small_with_retrieval"
    CASCADE = "cascade"
    ABSTAIN = "abstain"


class BackendName(StrEnum):
    SMALL = "small"
    LARGE = "large"


class EvaluationType(StrEnum):
    EXACT_MATCH = "exact_match"
    TOKEN_F1 = "token_f1"
    NUMERIC = "numeric"
    CLASSIFICATION = "classification"
    ABSTENTION = "abstention"
    KEYWORD = "keyword"


class GenerationRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    prompt: str = Field(min_length=1, max_length=50_000)
    max_new_tokens: int | None = Field(default=None, ge=1, le=4096)
    requires_retrieval: bool = False
    allow_abstention: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt")
    @classmethod
    def prompt_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must contain non-whitespace text")
        return value


class RequestFeatures(BaseModel):
    char_count: int
    word_count: int
    token_count: int
    line_count: int
    sentence_count: int
    digit_ratio: float
    punctuation_count: int
    question_mark_count: int
    has_code_block: bool
    has_math_symbols: bool
    has_url: bool
    requests_long_output: bool
    category: str
    retrieval_similarity: float = 0.0
    retrieval_margin: float = 0.0
    relevant_context_found: bool = False

    def numeric_snapshot(self) -> dict[str, float]:
        data = self.model_dump(exclude={"category"})
        return {key: float(value) for key, value in data.items()}


class RouteDecision(BaseModel):
    route: RouteName
    confidence: float = Field(ge=0.0, le=1.0)
    difficulty_score: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str
    policy: str
    features: dict[str, Any] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)


class RetrievalHit(BaseModel):
    document_id: str
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RouterTrace(BaseModel):
    policy: str
    difficulty_score: float | None = None
    confidence: float
    reason: str


class ExecutionTrace(BaseModel):
    initial_backend: BackendName | None = None
    final_backend: BackendName | None = None
    retrieval_used: bool = False
    escalated: bool = False
    abstained: bool = False
    escalation_reason: str | None = None
    initial_answer: str | None = None


class UsageStats(BaseModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    escalation_input_tokens: int = Field(default=0, ge=0)
    escalation_output_tokens: int = Field(default=0, ge=0)


class TimingStats(BaseModel):
    total_ms: float = Field(ge=0.0)
    routing_ms: float = Field(default=0.0, ge=0.0)
    retrieval_ms: float = Field(default=0.0, ge=0.0)
    generation_ms: float = Field(default=0.0, ge=0.0)
    escalation_ms: float = Field(default=0.0, ge=0.0)
    time_to_first_token_ms: float | None = Field(default=None, ge=0.0)


class MemoryStats(BaseModel):
    process_rss_mb: float = Field(ge=0.0)
    peak_cuda_mb: float | None = Field(default=None, ge=0.0)


class GenerationResponse(BaseModel):
    request_id: str
    answer: str
    route: RouteName
    router: RouterTrace
    execution: ExecutionTrace
    usage: UsageStats
    timing: TimingStats
    memory: MemoryStats
    retrieval: list[RetrievalHit] = Field(default_factory=list)
    backend_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    fake: bool = False


class BackendGeneration(BaseModel):
    text: str
    backend: BackendName
    input_tokens: int
    output_tokens: int
    confidence: float = Field(ge=0.0, le=1.0)
    generation_ms: float = Field(ge=0.0)
    time_to_first_token_ms: float | None = Field(default=None, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkRecord(BaseModel):
    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    reference_answer: str
    evaluation_type: EvaluationType
    requires_retrieval: bool = False
    must_abstain: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class PredictionRecord(BaseModel):
    example_id: str
    category: str
    policy: str
    route: RouteName
    backend: BackendName | None
    raw_answer: str
    normalized_answer: str
    reference: str
    quality_score: float = Field(ge=0.0, le=1.0)
    metric_name: str
    latency_ms: float = Field(ge=0.0)
    generation_ms: float = Field(default=0.0, ge=0.0)
    retrieval_ms: float = Field(default=0.0, ge=0.0)
    time_to_first_token_ms: float | None = Field(default=None, ge=0.0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    process_rss_mb: float | None = Field(default=None, ge=0.0)
    peak_cuda_mb: float | None = Field(default=None, ge=0.0)
    retrieval_used: bool
    escalated: bool
    abstained: bool
    confidence: float = Field(ge=0.0, le=1.0)
    error: str | None = None
