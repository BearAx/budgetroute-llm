"""Transparent request-difficulty and retrieval-aware heuristics."""

from __future__ import annotations

from budgetroute.config import RoutingConfig
from budgetroute.routing.base import feature_snapshot
from budgetroute.schemas import GenerationRequest, RequestFeatures, RouteDecision, RouteName


def difficulty_score(features: RequestFeatures) -> float:
    score = 0.0
    score += min(features.word_count / 200.0, 0.22)
    score += min(features.sentence_count / 10.0, 0.10)
    score += 0.18 if features.has_code_block else 0.0
    score += 0.16 if features.has_math_symbols else 0.0
    score += 0.12 if features.requests_long_output else 0.0
    score += 0.12 if features.category in {"numeric_reasoning", "code"} else 0.0
    score += 0.10 if features.has_url else 0.0
    score += min(features.digit_ratio * 0.5, 0.10)
    return min(1.0, score)


class HeuristicPolicy:
    name = "heuristic"

    def __init__(self, config: RoutingConfig) -> None:
        self.config = config

    def decide(self, request: GenerationRequest, features: RequestFeatures) -> RouteDecision:
        difficulty = difficulty_score(features)
        thresholds = {
            "difficulty": self.config.difficulty_threshold,
            "abstain": self.config.abstain_threshold,
            "retrieval_similarity": self.config.retrieval_similarity_threshold,
        }
        if request.allow_abstention and (
            request.metadata.get("must_abstain") or features.category == "ambiguous"
        ):
            return RouteDecision(
                route=RouteName.ABSTAIN,
                confidence=0.9,
                difficulty_score=difficulty,
                reason="Request is marked unanswerable or has insufficiently specified context",
                policy=self.name,
                features=feature_snapshot(features),
                thresholds=thresholds,
            )
        if request.requires_retrieval and features.relevant_context_found:
            return RouteDecision(
                route=RouteName.SMALL_WITH_RETRIEVAL,
                confidence=min(1.0, max(0.5, features.retrieval_similarity)),
                difficulty_score=difficulty,
                reason="Request requires retrieval and relevant context was found",
                policy=self.name,
                features=feature_snapshot(features),
                thresholds=thresholds,
            )
        route = (
            RouteName.LARGE if difficulty >= self.config.difficulty_threshold else RouteName.SMALL
        )
        margin = abs(difficulty - self.config.difficulty_threshold)
        return RouteDecision(
            route=route,
            confidence=min(1.0, 0.5 + margin),
            difficulty_score=difficulty,
            reason=(
                f"Difficulty {difficulty:.3f} is "
                f"{'above' if route == RouteName.LARGE else 'below'} the configured threshold"
            ),
            policy=self.name,
            features=feature_snapshot(features),
            thresholds=thresholds,
        )


class RetrievalFirstPolicy:
    name = "retrieval_first"

    def __init__(self, config: RoutingConfig) -> None:
        self.config = config

    def decide(self, request: GenerationRequest, features: RequestFeatures) -> RouteDecision:
        difficulty = difficulty_score(features)
        threshold = self.config.retrieval_similarity_threshold
        if features.relevant_context_found and features.retrieval_similarity >= threshold:
            return RouteDecision(
                route=RouteName.SMALL_WITH_RETRIEVAL,
                confidence=min(1.0, max(0.5, features.retrieval_similarity)),
                difficulty_score=difficulty,
                reason="Pre-routing retrieval found context above the similarity threshold",
                policy=self.name,
                features=feature_snapshot(features),
                thresholds={"retrieval_similarity": threshold},
            )
        route = (
            RouteName.LARGE if difficulty >= self.config.difficulty_threshold else RouteName.SMALL
        )
        return RouteDecision(
            route=route,
            confidence=0.6,
            difficulty_score=difficulty,
            reason="Retrieval was not sufficiently relevant; selected backend by difficulty",
            policy=self.name,
            features=feature_snapshot(features),
            thresholds={
                "retrieval_similarity": threshold,
                "difficulty": self.config.difficulty_threshold,
            },
        )
