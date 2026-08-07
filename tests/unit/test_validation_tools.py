from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from budgetroute.validation.compatibility import run_compatibility_matrix
from budgetroute.validation.load import run_load_test


def test_multi_endpoint_load_artifact_uses_wall_time_and_hides_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "load-test-secret-0001"
    monkeypatch.setenv("LOAD_TEST_KEY", secret)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {secret}"
        return httpx.Response(
            200,
            json={"route": "small", "request_id": "response"},
            headers={"content-type": "application/json"},
        )

    run_directory = asyncio.run(
        run_load_test(
            ["https://service-a.example", "https://service-b.example"],
            request_count=20,
            concurrency=4,
            output_root=tmp_path,
            api_key_env="LOAD_TEST_KEY",
            transport=httpx.MockTransport(handler),
        )
    )
    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    assert summary["success_count"] == 20
    assert summary["throughput_requests_per_second"] > 0
    assert summary["targets"] == [
        "https://service-a.example",
        "https://service-b.example",
    ]
    assert secret not in "".join(
        path.read_text(encoding="utf-8") for path in run_directory.iterdir()
    )


def test_fake_runtime_compatibility_matrix_is_explicitly_non_comparative(
    tmp_path: Path,
) -> None:
    run_directory = run_compatibility_matrix([Path("configs/serving/fake.yaml")], tmp_path)
    artifact = json.loads((run_directory / "compatibility.json").read_text(encoding="utf-8"))
    assert artifact["passed"] is True
    assert artifact["results"][0]["batch"]["order_preserved"] is True
    assert "not performance comparisons" in artifact["claim_boundary"]
