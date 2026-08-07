"""Deterministic deployment-specific request and response policy controls."""

from __future__ import annotations

import json
from typing import Any

from budgetroute.config import ContentPolicyConfig
from budgetroute.exceptions import ContentPolicyError
from budgetroute.schemas import GenerationRequest, GenerationResponse


def _metadata_shape(value: Any, depth: int = 1) -> tuple[int, int]:
    if isinstance(value, dict):
        nested = [_metadata_shape(item, depth + 1) for item in value.values()]
        return (
            len(value) + sum(item[0] for item in nested),
            max([depth, *(item[1] for item in nested)]),
        )
    if isinstance(value, list):
        nested = [_metadata_shape(item, depth + 1) for item in value]
        return (sum(item[0] for item in nested), max([depth, *(item[1] for item in nested)]))
    return (0, depth)


class ContentPolicy:
    def __init__(self, config: ContentPolicyConfig) -> None:
        self.config = config
        self._prompt_patterns = [item.casefold() for item in config.prompt_blocklist]
        self._output_patterns = [item.casefold() for item in config.output_blocklist]

    def validate_request(self, request: GenerationRequest) -> None:
        serialized = json.dumps(request.metadata, ensure_ascii=False, default=str)
        key_count, depth = _metadata_shape(request.metadata)
        if len(serialized) > self.config.max_metadata_json_chars:
            raise ContentPolicyError("request metadata exceeds the configured size policy")
        if key_count > self.config.max_metadata_keys:
            raise ContentPolicyError("request metadata exceeds the configured key-count policy")
        if depth > self.config.max_metadata_depth:
            raise ContentPolicyError("request metadata exceeds the configured nesting policy")
        if self.config.enabled:
            input_text = f"{request.prompt}\n{serialized}".casefold()
            if any(pattern in input_text for pattern in self._prompt_patterns):
                raise ContentPolicyError("request matched a configured input policy rule")

    def validate_response(self, response: GenerationResponse) -> None:
        if not self.config.enabled:
            return
        outward = json.dumps(
            response.model_dump(mode="json"), ensure_ascii=False, default=str
        ).casefold()
        if any(pattern in outward for pattern in self._output_patterns):
            raise ContentPolicyError("generated response matched a configured output policy rule")
