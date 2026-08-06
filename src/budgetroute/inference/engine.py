"""End-to-end routing, retrieval, generation, escalation, and trace orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass

from budgetroute.backends.base import GenerationBackend
from budgetroute.config import RoutingConfig
from budgetroute.exceptions import RetrievalError
from budgetroute.features.request_features import RequestFeatureExtractor
from budgetroute.profiling.memory import current_memory
from budgetroute.profiling.system import Timer
from budgetroute.retrieval.service import RetrievalService
from budgetroute.routing.base import RoutingPolicy
from budgetroute.routing.cascade import escalation_reason
from budgetroute.schemas import (
    BackendGeneration,
    BackendName,
    ExecutionTrace,
    GenerationRequest,
    GenerationResponse,
    RetrievalHit,
    RouteDecision,
    RouteName,
    RouterTrace,
    TimingStats,
    UsageStats,
)


@dataclass(frozen=True)
class RouteResult:
    decision: RouteDecision
    retrieval_ms: float
    hits: list[RetrievalHit]


class InferenceEngine:
    def __init__(
        self,
        small_backend: GenerationBackend,
        large_backend: GenerationBackend,
        policy: RoutingPolicy,
        routing_config: RoutingConfig,
        retriever: RetrievalService | None = None,
        feature_extractor: RequestFeatureExtractor | None = None,
    ) -> None:
        self.small_backend = small_backend
        self.large_backend = large_backend
        self.policy = policy
        self.routing_config = routing_config
        self.retriever = retriever
        self.feature_extractor = feature_extractor or RequestFeatureExtractor()

    def route(self, request: GenerationRequest) -> RouteResult:
        hits: list[RetrievalHit] = []
        retrieval_ms = 0.0
        should_retrieve = request.requires_retrieval or self.policy.name == "retrieval_first"
        if should_retrieve:
            if self.retriever is None or not self.retriever.ready:
                if not request.allow_abstention:
                    raise RetrievalError(
                        "request requires retrieval but no ready retriever is configured"
                    )
                features = self.feature_extractor.extract(request)
                decision = RouteDecision(
                    route=RouteName.ABSTAIN,
                    confidence=1.0,
                    difficulty_score=None,
                    reason="Retrieval is required but no ready retrieval service is configured",
                    policy=self.policy.name,
                    features=features.model_dump(mode="json"),
                )
                return RouteResult(decision, retrieval_ms, hits)
            hits = self.retriever.search(request.prompt)
            retrieval_ms = self.retriever.last_retrieval_ms
        features = self.feature_extractor.extract(request, hits)
        with Timer() as timer:
            decision = self.policy.decide(request, features)
        decision.thresholds["routing_ms"] = timer.elapsed_ms
        return RouteResult(decision, retrieval_ms, hits)

    def _request_with_context(
        self, request: GenerationRequest, hits: list[RetrievalHit]
    ) -> GenerationRequest:
        context = "\n\n".join(f"[{hit.chunk_id}; score={hit.score:.3f}] {hit.text}" for hit in hits)
        return request.model_copy(
            update={"prompt": f"{request.prompt}\n\nReference context:\n{context}"}
        )

    def _generate(
        self,
        backend: GenerationBackend,
        request: GenerationRequest,
        hits: list[RetrievalHit],
        use_retrieval: bool,
    ) -> BackendGeneration:
        effective_request = self._request_with_context(request, hits) if use_retrieval else request
        return backend.generate(effective_request)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        total_start = time.perf_counter_ns()
        route_result = self.route(request)
        decision = route_result.decision
        hits = route_result.hits
        routing_ms = float(decision.thresholds.pop("routing_ms", 0.0))
        execution = ExecutionTrace()
        generation_ms = 0.0
        escalation_ms = 0.0
        initial: BackendGeneration | None = None
        final: BackendGeneration | None = None

        if decision.route == RouteName.ABSTAIN:
            execution.abstained = True
            answer = "ABSTAIN: insufficient confidence or context."
            usage = UsageStats(input_tokens=0, output_tokens=0)
            confidence: float | None = None
        elif decision.route in {RouteName.SMALL, RouteName.SMALL_WITH_RETRIEVAL}:
            execution.initial_backend = BackendName.SMALL
            execution.final_backend = BackendName.SMALL
            execution.retrieval_used = decision.route == RouteName.SMALL_WITH_RETRIEVAL
            if execution.retrieval_used and not hits:
                raise RetrievalError(
                    "routing selected retrieval but no context passed the threshold"
                )
            final = self._generate(self.small_backend, request, hits, execution.retrieval_used)
            generation_ms = final.generation_ms
            answer = final.text
            confidence = final.confidence
            usage = UsageStats(input_tokens=final.input_tokens, output_tokens=final.output_tokens)
        elif decision.route == RouteName.LARGE:
            execution.initial_backend = BackendName.LARGE
            execution.final_backend = BackendName.LARGE
            final = self._generate(self.large_backend, request, hits, False)
            generation_ms = final.generation_ms
            answer = final.text
            confidence = final.confidence
            usage = UsageStats(input_tokens=final.input_tokens, output_tokens=final.output_tokens)
        elif decision.route == RouteName.CASCADE:
            execution.initial_backend = BackendName.SMALL
            initial = self._generate(
                self.small_backend, request, hits, request.requires_retrieval and bool(hits)
            )
            generation_ms = initial.generation_ms
            reason = escalation_reason(
                initial.confidence, self.routing_config.cascade_confidence_threshold
            )
            if reason is None:
                final = initial
                answer = initial.text
                confidence = initial.confidence
                execution.final_backend = BackendName.SMALL
                usage = UsageStats(
                    input_tokens=initial.input_tokens, output_tokens=initial.output_tokens
                )
            else:
                execution.escalated = True
                execution.escalation_reason = reason
                if self.routing_config.retain_initial_answer:
                    execution.initial_answer = initial.text
                with Timer() as escalation_timer:
                    final = self._generate(self.large_backend, request, hits, False)
                escalation_ms = escalation_timer.elapsed_ms
                answer = final.text
                confidence = final.confidence
                execution.final_backend = BackendName.LARGE
                usage = UsageStats(
                    input_tokens=initial.input_tokens,
                    output_tokens=initial.output_tokens,
                    escalation_input_tokens=final.input_tokens,
                    escalation_output_tokens=final.output_tokens,
                )
        else:  # pragma: no cover - enums and policy validation make this unreachable
            raise ValueError(f"unsupported route: {decision.route}")

        total_ms = (time.perf_counter_ns() - total_start) / 1_000_000
        ttft = final.time_to_first_token_ms if final is not None else None
        fake = bool(
            final.metadata.get("fake")
            if final is not None
            else self.small_backend.metadata().get("fake")
            and self.large_backend.metadata().get("fake")
        )
        return GenerationResponse(
            request_id=request.request_id,
            answer=answer,
            route=decision.route,
            router=RouterTrace(
                policy=decision.policy,
                difficulty_score=decision.difficulty_score,
                confidence=decision.confidence,
                reason=decision.reason,
            ),
            execution=execution,
            usage=usage,
            timing=TimingStats(
                total_ms=total_ms,
                routing_ms=routing_ms,
                retrieval_ms=route_result.retrieval_ms,
                generation_ms=generation_ms,
                escalation_ms=escalation_ms,
                time_to_first_token_ms=ttft,
            ),
            memory=current_memory(),
            retrieval=hits if execution.retrieval_used else [],
            backend_confidence=confidence,
            fake=fake,
        )
