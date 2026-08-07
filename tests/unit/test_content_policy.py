from __future__ import annotations

import pytest

from budgetroute.config import ContentPolicyConfig
from budgetroute.exceptions import ContentPolicyError
from budgetroute.inference.content_policy import ContentPolicy
from budgetroute.schemas import (
    ExecutionTrace,
    GenerationRequest,
    GenerationResponse,
    MemoryStats,
    RouteName,
    RouterTrace,
    TimingStats,
    UsageStats,
)


def test_content_policy_blocks_configured_input_and_output_without_echoing_content() -> None:
    policy = ContentPolicy(
        ContentPolicyConfig(
            enabled=True,
            prompt_blocklist=["blocked input"],
            output_blocklist=["blocked output"],
        )
    )
    with pytest.raises(ContentPolicyError, match="input policy") as input_error:
        policy.validate_request(GenerationRequest(prompt="This has BLOCKED INPUT content"))
    assert "BLOCKED INPUT" not in str(input_error.value)
    with pytest.raises(ContentPolicyError, match="input policy"):
        policy.validate_request(
            GenerationRequest(prompt="safe", metadata={"attachment": "BLOCKED INPUT"})
        )

    response = GenerationResponse(
        request_id="request-1",
        answer="A blocked output value",
        route=RouteName.ABSTAIN,
        router=RouterTrace(policy="test", confidence=1.0, reason="test"),
        execution=ExecutionTrace(abstained=True),
        usage=UsageStats(input_tokens=0, output_tokens=0),
        timing=TimingStats(total_ms=0.0),
        memory=MemoryStats(process_rss_mb=0.0),
    )
    with pytest.raises(ContentPolicyError, match="output policy"):
        policy.validate_response(response)
    with pytest.raises(ContentPolicyError, match="output policy"):
        policy.validate_response(
            response.model_copy(
                update={
                    "answer": "safe",
                    "router": response.router.model_copy(update={"reason": "blocked output"}),
                }
            )
        )


def test_metadata_shape_limits_apply_even_when_blocklists_are_disabled() -> None:
    policy = ContentPolicy(
        ContentPolicyConfig(enabled=False, max_metadata_depth=2, max_metadata_keys=10)
    )
    with pytest.raises(ContentPolicyError, match="nesting"):
        policy.validate_request(
            GenerationRequest(prompt="test", metadata={"outer": {"inner": {"value": 1}}})
        )
