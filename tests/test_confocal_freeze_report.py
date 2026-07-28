from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sarcomere_analysis.confocal_freeze_report import (
    MANUAL_SPOT_CHECK_VERDICTS,
    build_confocal_freeze_report,
    write_confocal_freeze_report,
)


def freeze_config(tmp_path: Path) -> dict:
    return {"paths": {"output_dir": str(tmp_path / "results")}}


def write_freeze_inputs(root: Path, include_review_index: bool = True) -> tuple[Path, Path, Path, Path]:
    pipeline = root / "pipeline"
    audit = root / "audit"
    endpoint = root / "endpoint"
    review = root / "review"
    for path in [pipeline, audit, endpoint, review]:
        path.mkdir(parents=True, exist_ok=True)

    write_json(
        pipeline / "confocal_pipeline_summary.json",
        {
            "images_processed": 42,
            "errors": 0,
            "calibrated_images": 42,
            "total_patches": 40362,
            "selected_candidate_patches": 10546,
            "valid_selected_spacing_patches": 786,
            "median_selected_oop": 0.7068,
            "median_selected_spacing_um": 2.2396,
            "widefield_calibration_used": False,
        },
    )
    write_json(
        audit / "confocal_larger_audit_summary.json",
        {
            "selected_spacing_valid_fraction": 0.0745,
            "image_count_by_interpretation_class": {
                "spacing_moderate": 7,
                "spacing_robust": 0,
            },
        },
    )
    write_json(
        endpoint / "confocal_endpoint_summary.json",
        {
            "oop_reportable_image_count": 41,
            "spacing_reportable_image_count": 7,
            "endpoint_class_counts": {
                "spacing_eligible_moderate": 7,
                "spacing_eligible_low_confidence": 25,
                "oop_only_spacing_low_yield": 5,
                "low_candidate_review_needed": 5,
            },
        },
    )
    pd.DataFrame(
        [
            {
                "confocal_image_id": "8A793",
                "filename": "8A793.tif",
                "spacing_reportable": True,
            },
            {
                "confocal_image_id": "E0ABF",
                "filename": "E0ABF.tif",
                "spacing_reportable": True,
            },
        ]
    ).to_csv(endpoint / "confocal_endpoint_per_image.csv", index=False)
    write_json(
        review / "confocal_endpoint_review_pack_summary.json",
        {
            "images_included": 17,
            "review_group_counts": {"spacing_reportable": 7},
            "review_image_files_copied": 102,
            "missing_preview_count": 0,
            "zip_path": "results/confocal_endpoint_review_pack/confocal_endpoint_review_pack.zip",
        },
    )
    if include_review_index:
        pd.DataFrame({"filename": ["8A793.tif"], "review_group": ["spacing_reportable"]}).to_csv(
            review / "confocal_endpoint_review_index.csv", index=False
        )
    return pipeline, audit, endpoint, review


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_freeze_report_loads_required_input_summaries(tmp_path: Path) -> None:
    pipeline, audit, endpoint, review = write_freeze_inputs(tmp_path)

    report = build_confocal_freeze_report(pipeline, audit, endpoint, review)

    assert report["larger_dataset"]["images"] == 42
    assert report["endpoint_result"]["oop_reportable_images"] == 41
    assert report["endpoint_review_pack"]["images_included"] == 17


def test_manual_review_verdicts_are_recorded(tmp_path: Path) -> None:
    pipeline, audit, endpoint, review = write_freeze_inputs(tmp_path)

    report = build_confocal_freeze_report(pipeline, audit, endpoint, review)

    assert report["manual_visual_spot_check"]["reviewed_in_chat_count"] == 6
    assert report["manual_visual_spot_check"]["reviewed_pass_count"] == 6
    assert any(item["filename"] == "E0ABF.tif" for item in MANUAL_SPOT_CHECK_VERDICTS)
    assert "E0ABF.tif" in report["manual_visual_spot_check"]["pending_visual_confirmation"]


def test_final_decisions_are_present(tmp_path: Path) -> None:
    pipeline, audit, endpoint, review = write_freeze_inputs(tmp_path)

    report = build_confocal_freeze_report(pipeline, audit, endpoint, review)

    decisions = report["final_frozen_interpretation"]
    assert "Confocal OOP/coherence is broadly reportable." in decisions
    assert "Confocal spacing is not universal." in decisions
    assert report["primary_gate"] == "moderate"


def test_allowed_and_not_allowed_downstream_uses_are_present(tmp_path: Path) -> None:
    pipeline, audit, endpoint, review = write_freeze_inputs(tmp_path)

    report = build_confocal_freeze_report(pipeline, audit, endpoint, review)

    assert "Report spacing only for spacing-reportable selected regions." in report["allowed_downstream_use"]
    assert "Whole-cohort spacing claims." in report["not_allowed_claims"]
    assert "Relaxed gate as primary." in report["not_allowed_claims"]


def test_json_serializable(tmp_path: Path) -> None:
    pipeline, audit, endpoint, review = write_freeze_inputs(tmp_path)

    report = build_confocal_freeze_report(pipeline, audit, endpoint, review)

    json.dumps(report)


def test_missing_optional_review_index_handled_gracefully(tmp_path: Path) -> None:
    pipeline, audit, endpoint, review = write_freeze_inputs(tmp_path, include_review_index=False)

    report = build_confocal_freeze_report(pipeline, audit, endpoint, review)

    assert report["source_summary_presence"]["review_pack_index"] is False
    assert report["endpoint_review_pack"]["review_index_rows"] == 0


def test_no_input_files_modified(tmp_path: Path) -> None:
    pipeline, audit, endpoint, review = write_freeze_inputs(tmp_path)
    source = endpoint / "confocal_endpoint_per_image.csv"
    before = source.read_bytes()

    write_confocal_freeze_report(
        freeze_config(tmp_path),
        pipeline_dir=pipeline,
        audit_dir=audit,
        endpoint_dir=endpoint,
        review_pack_dir=review,
        output_directory=tmp_path / "out",
        docs_directory=tmp_path / "docs",
    )

    assert source.read_bytes() == before
    assert (tmp_path / "out" / "confocal_freeze_summary.json").exists()
    assert (tmp_path / "docs" / "CONFOCAL_FREEZE_REPORT.md").exists()
