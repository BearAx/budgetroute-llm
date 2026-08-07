from __future__ import annotations

import json
from pathlib import Path

import pytest

from budgetroute.evaluation.dataset import (
    DatasetManifest,
    convert_dataset_row,
    dataset_hash,
    load_dataset_spec,
    materialize_router_test_split,
    select_source_indices,
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


def test_mmlu_stratified_selection_is_balanced_and_deterministic(project_root: Path) -> None:
    spec = load_dataset_spec(project_root / "configs/datasets/mmlu.yaml")
    upstream = [
        {"subject": subject, "question": f"{subject}-{index}"}
        for subject in ("algebra", "history", "physics")
        for index in range(4)
    ]

    selected = select_source_indices(
        upstream,
        spec,
        limit=6,
        offset=0,
        sampling="stratified",
        seed=42,
    )
    repeated = select_source_indices(
        upstream,
        spec,
        limit=6,
        offset=0,
        sampling="stratified",
        seed=42,
    )

    assert selected == repeated
    assert len(set(selected)) == 6
    assert {
        subject: sum(upstream[index]["subject"] == subject for index in selected)
        for subject in {"algebra", "history", "physics"}
    } == {
        "algebra": 2,
        "history": 2,
        "physics": 2,
    }


def test_stratified_selection_rejects_unsupported_dataset(project_root: Path) -> None:
    spec = load_dataset_spec(project_root / "configs/datasets/gsm8k.yaml")
    with pytest.raises(BudgetRouteError, match="supports only MMLU"):
        select_source_indices(
            [{"question": "one"}],
            spec,
            limit=1,
            offset=0,
            sampling="stratified",
            seed=42,
        )


def test_materialize_router_test_split_preserves_persisted_id_order(tmp_path: Path) -> None:
    dataset = tmp_path / "benchmark.jsonl"
    rows = [
        {
            "id": record_id,
            "category": "factual",
            "prompt": f"Question {record_id}?",
            "reference_answer": "A",
            "evaluation_type": "classification",
        }
        for record_id in ("one", "two", "three")
    ]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    source_manifest = DatasetManifest(
        name="fixture",
        adapter="mmlu",
        source="fixture/source",
        config_name="all",
        source_revision="a" * 40,
        source_split="test",
        license="MIT",
        homepage="https://example.test",
        prompt_version="v1",
        record_count=3,
        records_sha256=dataset_hash(dataset),
        selection={"method": "head", "offset": 0, "limit": 3},
        datasets_version="test",
    )
    source_manifest_path = tmp_path / "benchmark.manifest.json"
    source_manifest_path.write_text(
        source_manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    router_metadata = tmp_path / "router.metadata.json"
    router_metadata.write_text(
        json.dumps({"seed": 42, "splits": {"test": {"ids": ["three", "one"]}}}),
        encoding="utf-8",
    )
    output = tmp_path / "router-test.jsonl"

    manifest = materialize_router_test_split(router_metadata, dataset, source_manifest_path, output)

    assert [json.loads(line)["id"] for line in output.read_text().splitlines()] == [
        "three",
        "one",
    ]
    assert manifest.record_count == 2
    assert manifest.selection["method"] == "router_test_split"
    assert manifest.selection["parent_records_sha256"] == source_manifest.records_sha256
    assert validate_dataset_manifest(output, output.with_suffix(".manifest.json")) == manifest
