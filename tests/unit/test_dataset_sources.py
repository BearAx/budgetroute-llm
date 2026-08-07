from __future__ import annotations

import json
from pathlib import Path

import pytest

from budgetroute.evaluation.dataset import (
    DatasetManifest,
    convert_dataset_row,
    dataset_hash,
    load_dataset_spec,
    validate_dataset_manifest,
)
from budgetroute.exceptions import BudgetRouteError
from budgetroute.schemas import EvaluationType


def test_pinned_public_dataset_specs(project_root: Path) -> None:
    expected = {
        "gsm8k.yaml": ("gsm8k", "MIT"),
        "mmlu.yaml": ("mmlu", "MIT"),
        "hotpotqa.yaml": ("hotpot_qa", "CC-BY-SA-4.0"),
    }
    for filename, (adapter, license_name) in expected.items():
        spec = load_dataset_spec(project_root / "configs/datasets" / filename)
        assert spec.adapter == adapter
        assert spec.license == license_name
        assert len(spec.revision) == 40
        int(spec.revision, 16)


def test_dataset_converters_are_deterministic(project_root: Path) -> None:
    gsm = load_dataset_spec(project_root / "configs/datasets/gsm8k.yaml")
    gsm_record, gsm_corpus = convert_dataset_row(
        gsm,
        {"question": "If two plus three?", "answer": "Reasoning\n#### 5"},
        4,
    )
    assert gsm_record.reference_answer == "5"
    assert gsm_record.evaluation_type == EvaluationType.NUMERIC
    assert gsm_corpus is None

    mmlu = load_dataset_spec(project_root / "configs/datasets/mmlu.yaml")
    mmlu_record, _ = convert_dataset_row(
        mmlu,
        {
            "question": "Which value is even?",
            "choices": ["1", "2", "3", "5"],
            "answer": 1,
            "subject": "elementary_mathematics",
        },
        9,
    )
    assert mmlu_record.reference_answer == "B"
    assert "B. 2" in mmlu_record.prompt
    assert mmlu_record.evaluation_type == EvaluationType.CLASSIFICATION

    hotpot = load_dataset_spec(project_root / "configs/datasets/hotpotqa.yaml")
    hotpot_record, corpus = convert_dataset_row(
        hotpot,
        {
            "question": "Who created the project?",
            "answer": "Ada",
            "context": {"title": ["Project"], "sentences": [["Ada created it."]]},
        },
        2,
    )
    assert hotpot_record.requires_retrieval is True
    assert corpus == "Project\nAda created it."
    assert hotpot_record.group_id is not None


def test_manifest_detects_dataset_tampering(tmp_path: Path) -> None:
    dataset = tmp_path / "benchmark.jsonl"
    row = {
        "id": "one",
        "category": "factual",
        "prompt": "Question?",
        "reference_answer": "Answer",
        "evaluation_type": "exact_match",
    }
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = DatasetManifest(
        name="fixture",
        adapter="gsm8k",
        source="fixture/source",
        config_name=None,
        source_revision="a" * 40,
        source_split="test",
        license="MIT",
        homepage="https://example.test",
        prompt_version="v1",
        record_count=1,
        records_sha256=dataset_hash(dataset),
        selection={"offset": 0, "limit": 1},
        datasets_version="test",
    )
    manifest_path = tmp_path / "benchmark.manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    assert validate_dataset_manifest(dataset, manifest_path).record_count == 1
    dataset.write_text(dataset.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(BudgetRouteError, match="does not match manifest"):
        validate_dataset_manifest(dataset, manifest_path)
