"""Deterministic, interpretable request feature extraction."""

from __future__ import annotations

import re
import string

from budgetroute.schemas import GenerationRequest, RequestFeatures, RetrievalHit


class RequestFeatureExtractor:
    _url_pattern = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    _sentence_pattern = re.compile(r"[.!?]+(?:\s|$)")
    _token_pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def category(self, prompt: str) -> str:
        lowered = prompt.lower()
        if "```" in prompt or any(term in lowered for term in ("function", "class ", "code")):
            return "code"
        if any(
            term in lowered for term in ("calculate", "how many", "equation", "sum", "multiply")
        ):
            return "numeric_reasoning"
        if any(term in lowered for term in ("classify", "sentiment", "label")):
            return "classification"
        if any(term in lowered for term in ("summarize", "summary", "in detail", "essay")):
            return "long_form"
        if any(
            term in lowered for term in ("according to", "document", "corpus", "project codename")
        ):
            return "retrieval"
        if any(term in lowered for term in ("ambiguous", "unknown", "unanswerable")):
            return "ambiguous"
        return "factual"

    def extract(
        self, request: GenerationRequest, retrieval_hits: list[RetrievalHit] | None = None
    ) -> RequestFeatures:
        prompt = request.prompt
        words = re.findall(r"\b\w+\b", prompt, flags=re.UNICODE)
        digits = sum(character.isdigit() for character in prompt)
        non_space = sum(not character.isspace() for character in prompt)
        punctuation_count = sum(character in string.punctuation for character in prompt)
        sentences = self._sentence_pattern.findall(prompt)
        hits = retrieval_hits or []
        top_score = hits[0].score if hits else 0.0
        second_score = hits[1].score if len(hits) > 1 else 0.0
        lowered = prompt.lower()
        return RequestFeatures(
            char_count=len(prompt),
            word_count=len(words),
            token_count=len(self._token_pattern.findall(prompt)),
            line_count=prompt.count("\n") + 1,
            sentence_count=max(1, len(sentences)),
            digit_ratio=digits / max(non_space, 1),
            punctuation_count=punctuation_count,
            question_mark_count=prompt.count("?"),
            has_code_block="```" in prompt,
            has_math_symbols=bool(re.search(r"[=+*/^√∑∫<>]", prompt)),
            has_url=bool(self._url_pattern.search(prompt)),
            requests_long_output=any(
                marker in lowered
                for marker in ("in detail", "comprehensive", "essay", "step by step", "long answer")
            ),
            category=self.category(prompt),
            retrieval_similarity=top_score,
            retrieval_margin=top_score - second_score,
            relevant_context_found=bool(hits),
        )
