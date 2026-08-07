from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from budgetroute.cli import app


def test_cli_validation_inspection_doctor_and_demo(project_root: Path) -> None:
    runner = CliRunner()
    config = project_root / "configs/serving/fake.yaml"
    benchmark = project_root / "data/sample_benchmark.jsonl"

    validated = runner.invoke(app, ["validate-config", "--config", str(config)])
    assert validated.exit_code == 0, validated.output
    assert '"valid": true' in validated.output

    inspected = runner.invoke(app, ["inspect-data", "--path", str(benchmark)])
    assert inspected.exit_code == 0, inspected.output
    assert '"records": 12' in inspected.output

    doctor = runner.invoke(app, ["doctor", "--config", str(config)])
    assert doctor.exit_code == 0, doctor.output
    assert '"budgetroute_version": "0.4.0"' in doctor.output

    security = runner.invoke(app, ["security-check", "--config", str(config)])
    assert security.exit_code == 0, security.output
    assert '"passed": true' in security.output

    demo = runner.invoke(app, ["demo", "--config", str(config)])
    assert demo.exit_code == 0, demo.output
    assert "FAKE MODE DEMO" in demo.output
    assert "cascade -> escalate" in demo.output

    unsafe_bind = runner.invoke(app, ["serve", "--config", str(config), "--host", "0.0.0.0"])
    assert unsafe_bind.exit_code != 0
    assert "non-loopback API binding requires" in str(unsafe_bind.exception)
