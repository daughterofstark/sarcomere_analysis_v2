from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sarcomere_analysis.confocal_endpoint_report import (
    build_endpoint_per_image,
    build_endpoint_summary,
    classify_endpoint,
    write_confocal_endpoint_report,
)


def endpoint_config(tmp_path: Path) -> dict:
    return {"paths": {"output_dir": str(tmp_path / "results")}}


def synthetic_triage() -> pd.DataFrame:
    return pd.DataFrame(
        [
            triage_row("spacing", "spacing.tif", 100, 0.40, 20, 0.20, 2.0, 0.8),
            triage_row("lowconf", "lowconf.tif", 100, 0.40, 8, 0.08, 2.1, 0.7),
            triage_row("ooponly", "ooponly.tif", 100, 0.40, 2, 0.02, 2.2, 0.6),
            triage_row("lowcandidate", "lowcandidate.tif", 100, 0.04, 0, 0.0, None, 0.5),
            triage_row("uncalibrated", "uncalibrated.tif", 100, 0.40, 20, 0.20, 2.0, 0.8),
            triage_row("error", "error.tif", 100, 0.40, 20, 0.20, 2.0, 0.8),
        ]
    )


def triage_row(
    image_id: str,
    filename: str,
    total: int,
    candidate_fraction: float,
    spacing_count: int,
    spacing_fraction: float,
    spacing_median: float | None,
    oop: float,
) -> dict:
    candidates = int(total * candidate_fraction)
    return {
        "confocal_image_id": image_id,
        "filename": filename,
        "pixel_size_x_um": 0.08,
        "pixel_size_y_um": 0.08,
        "total_patches": total,
        "selected_candidate_patches": candidates,
        "selected_candidate_fraction": candidate_fraction,
        "valid_selected_spacing_patches": spacing_count,
        "selected_spacing_valid_fraction": spacing_fraction,
        "selected_spacing_median_um": spacing_median,
        "selected_spacing_iqr_um": 0.2,
        "selected_median_oop": oop,
        "selected_vs_all_oop_difference": 0.1,
    }


def synthetic_pipeline_image() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"confocal_image_id": "spacing", "pixel_size_available": True, "processing_status": "ok", "error_message": ""},
            {"confocal_image_id": "lowconf", "pixel_size_available": True, "processing_status": "ok", "error_message": ""},
            {"confocal_image_id": "ooponly", "pixel_size_available": True, "processing_status": "ok", "error_message": ""},
            {"confocal_image_id": "lowcandidate", "pixel_size_available": True, "processing_status": "ok", "error_message": ""},
            {
                "confocal_image_id": "uncalibrated",
                "pixel_size_available": False,
                "processing_status": "ok",
                "error_message": "",
            },
            {"confocal_image_id": "error", "pixel_size_available": True, "processing_status": "error", "error_message": "bad"},
        ]
    )


def write_inputs(root: Path) -> tuple[Path, Path]:
    audit = root / "audit"
    pipeline = root / "pipeline"
    audit.mkdir(parents=True)
    pipeline.mkdir(parents=True)
    synthetic_triage().to_csv(audit / "confocal_larger_image_triage.csv", index=False)
    pd.DataFrame([{"images_processed": 6}]).to_csv(audit / "confocal_larger_cohort_summary.csv", index=False)
    (audit / "confocal_larger_audit_summary.json").write_text(
        json.dumps(
            {
                "image_count_by_interpretation_class": {"spacing_robust": 0, "spacing_moderate": 1},
                "selected_spacing_valid_fraction": 0.1,
                "median_selected_oop": 0.7,
                "median_selected_spacing_um": 2.0,
            }
        ),
        encoding="utf-8",
    )
    synthetic_pipeline_image().to_csv(pipeline / "confocal_pipeline_per_image.csv", index=False)
    (pipeline / "confocal_pipeline_summary.json").write_text(
        json.dumps(
            {
                "images_processed": 6,
                "errors": 1,
                "calibrated_images": 5,
                "selected_candidate_patches": 204,
                "valid_selected_spacing_patches": 70,
            }
        ),
        encoding="utf-8",
    )
    return audit, pipeline


def test_endpoint_classes_assigned_correctly() -> None:
    per_image = build_endpoint_per_image(synthetic_triage(), synthetic_pipeline_image())
    classes = dict(zip(per_image["confocal_image_id"], per_image["endpoint_class"]))

    assert classes["spacing"] == "spacing_eligible_moderate"
    assert classes["lowconf"] == "spacing_eligible_low_confidence"
    assert classes["ooponly"] == "oop_only_spacing_low_yield"
    assert classes["lowcandidate"] == "low_candidate_review_needed"
    assert classes["uncalibrated"] == "failed_or_unusable"
    assert classes["error"] == "failed_or_unusable"


def test_spacing_reportable_requires_count_and_fraction() -> None:
    assert classify_endpoint(True, 40, 0.4, 10, 0.10)[0] == "spacing_eligible_moderate"
    assert classify_endpoint(True, 40, 0.4, 9, 0.10)[0] != "spacing_eligible_moderate"
    assert classify_endpoint(True, 40, 0.4, 10, 0.09)[0] != "spacing_eligible_moderate"


def test_oop_reportable_uses_candidate_count_and_calibration() -> None:
    per_image = build_endpoint_per_image(synthetic_triage(), synthetic_pipeline_image())
    flags = dict(zip(per_image["confocal_image_id"], per_image["oop_reportable"]))

    assert bool(flags["ooponly"]) is True
    assert bool(flags["lowcandidate"]) is False
    assert bool(flags["uncalibrated"]) is False


def test_low_candidate_review_flag_works() -> None:
    per_image = build_endpoint_per_image(synthetic_triage(), synthetic_pipeline_image())
    row = per_image.loc[per_image["confocal_image_id"] == "lowcandidate"].iloc[0]

    assert bool(row["review_needed"]) is True
    assert row["endpoint_class"] == "low_candidate_review_needed"


def test_summary_counts_classes_correctly() -> None:
    per_image = build_endpoint_per_image(synthetic_triage(), synthetic_pipeline_image())
    summary = build_endpoint_summary(
        per_image,
        pd.DataFrame(),
        {"image_count_by_interpretation_class": {"spacing_robust": 0, "spacing_moderate": 1}},
        {"images_processed": 6, "errors": 1, "calibrated_images": 5},
    )

    assert summary["endpoint_class_counts"]["spacing_eligible_moderate"] == 1
    assert summary["spacing_reportable_image_count"] == 1
    assert summary["spacing_moderate_image_count_from_cohort_audit"] == 1


def test_output_json_serializable(tmp_path: Path) -> None:
    audit, pipeline = write_inputs(tmp_path)

    _, summary, paths = write_confocal_endpoint_report(
        endpoint_config(tmp_path),
        audit_dir=audit,
        pipeline_dir=pipeline,
        output_directory=tmp_path / "out",
        docs_directory=tmp_path / "docs",
    )

    json.dumps(summary)
    assert json.loads(paths["summary_json"].read_text(encoding="utf-8"))["mode"] == "confocal_endpoint_classification_report"
    assert paths["markdown"].exists()


def test_existing_outputs_not_modified(tmp_path: Path) -> None:
    audit, pipeline = write_inputs(tmp_path)
    source = audit / "confocal_larger_image_triage.csv"
    before = source.read_bytes()

    write_confocal_endpoint_report(
        endpoint_config(tmp_path),
        audit_dir=audit,
        pipeline_dir=pipeline,
        output_directory=tmp_path / "out",
        docs_directory=tmp_path / "docs",
    )

    assert source.read_bytes() == before
