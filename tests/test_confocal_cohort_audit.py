from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sarcomere_analysis.confocal_cohort_audit import (
    audit_confocal_cohort,
    build_cohort_summary_table,
    build_image_triage,
    select_review_images,
)


def audit_config(tmp_path: Path) -> dict:
    return {"paths": {"output_dir": str(tmp_path / "results")}}


def synthetic_per_image() -> pd.DataFrame:
    return pd.DataFrame(
        [
            image_row("robust", "robust.tif", 60, 30, 0.30, 2.0, 0.80),
            image_row("moderate", "moderate.tif", 60, 12, 0.20, 2.1, 0.70),
            image_row("ooponly", "ooponly.tif", 60, 2, 0.033333, 2.2, 0.60),
            image_row("lowcand", "lowcand.tif", 4, 0, 0.0, None, 0.55),
            image_row("broad", "broad.tif", 80, 50, 0.625, 2.6, 0.90),
            image_row("error", "error.tif", 50, 20, 0.40, 2.0, 0.50, status="error", error="bad image"),
        ]
    )


def image_row(
    image_id: str,
    filename: str,
    candidate_count: int,
    valid_spacing_count: int,
    spacing_fraction: float,
    median_spacing: float | None,
    oop: float,
    status: str = "ok",
    error: str = "",
) -> dict:
    return {
        "confocal_image_id": image_id,
        "filename": filename,
        "pixel_size_x_um": 0.08,
        "pixel_size_y_um": 0.08,
        "total_patches": 100,
        "selected_candidate_patches": candidate_count,
        "selected_candidate_fraction": candidate_count / 100,
        "valid_selected_spacing_patches": valid_spacing_count,
        "selected_spacing_valid_fraction": spacing_fraction,
        "selected_spacing_median_um": median_spacing,
        "selected_spacing_iqr_um": 0.2,
        "selected_spacing_range_um": "1.6-2.3" if median_spacing is not None else "",
        "selected_median_oop": oop,
        "all_region_median_oop": oop - 0.1,
        "selected_vs_all_oop_difference": 0.1,
        "selected_median_coherence": 0.7,
        "processing_status": status,
        "error_message": error,
    }


def synthetic_per_patch() -> pd.DataFrame:
    rows = []
    for image_id, filename in [("robust", "robust.tif"), ("moderate", "moderate.tif"), ("ooponly", "ooponly.tif")]:
        for idx in range(3):
            rows.append(
                {
                    "confocal_image_id": image_id,
                    "filename": filename,
                    "patch_id": f"{image_id}_p{idx}",
                    "candidate_striation_region": True,
                    "patch_oop": 0.5 + idx * 0.1,
                    "spacing_estimate_um": 1.7 + idx * 0.1,
                    "spacing_confidence": 0.2,
                    "spacing_valid": idx != 2,
                }
            )
    return pd.DataFrame(rows)


def synthetic_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "confocal_image_id": synthetic_per_image()["confocal_image_id"],
            "filename": synthetic_per_image()["filename"],
            "pixel_size_available": True,
        }
    )


def write_pipeline_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    synthetic_manifest().to_csv(root / "confocal_pipeline_manifest.csv", index=False)
    synthetic_per_image().to_csv(root / "confocal_pipeline_per_image.csv", index=False)
    synthetic_per_patch().to_csv(root / "confocal_pipeline_per_patch.csv", index=False)
    (root / "confocal_pipeline_summary.json").write_text(
        json.dumps(
            {
                "images_processed": 6,
                "errors": 1,
                "calibrated_images": 6,
                "selected_candidate_patches": 314,
                "valid_selected_spacing_patches": 114,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_image_triage_classes_assigned_correctly() -> None:
    triage = build_image_triage(synthetic_per_image())
    classes = dict(zip(triage["confocal_image_id"], triage["interpretation_class"]))

    assert classes["robust"] == "spacing_robust"
    assert classes["moderate"] == "spacing_moderate"
    assert classes["ooponly"] == "oop_only_low_spacing"
    assert classes["lowcand"] == "low_candidate_fraction_review"
    assert classes["broad"] == "broad_candidate_fraction_review"
    assert classes["error"] == "failed_or_error"


def test_cohort_summary_counts_classes_correctly() -> None:
    triage = build_image_triage(synthetic_per_image())
    summary = build_cohort_summary_table(triage, synthetic_per_patch(), synthetic_manifest(), {})

    counts = json.loads(summary.iloc[0]["image_count_by_interpretation_class"])
    assert counts["spacing_robust"] == 1
    assert counts["spacing_moderate"] == 1
    assert counts["oop_only_low_spacing"] == 1
    assert int(summary.iloc[0]["selected_candidate_patches"]) == 314


def test_selected_spacing_valid_fraction_computed_when_missing() -> None:
    per_image = synthetic_per_image().drop(columns=["selected_spacing_valid_fraction"])
    triage = build_image_triage(per_image)
    robust = triage.loc[triage["confocal_image_id"] == "robust"].iloc[0]

    assert robust["selected_spacing_valid_fraction"] == 0.5


def test_pilot_comparison_handles_missing_pilot_gracefully(tmp_path: Path) -> None:
    pipeline_dir = write_pipeline_dir(tmp_path / "pipeline")

    _, _, _, summary, _ = audit_confocal_cohort(
        audit_config(tmp_path),
        pipeline_dir=pipeline_dir,
        pilot_dir=tmp_path / "missing_pilot",
        output_directory=tmp_path / "audit",
    )

    assert summary["pilot_comparison"]["pilot_available"] is False


def test_top_and_bottom_images_identified() -> None:
    triage = build_image_triage(synthetic_per_image())
    selection = select_review_images(triage)

    assert selection["top_spacing_yield"].iloc[0]["confocal_image_id"] == "broad"
    assert "lowcand" in selection["bottom_spacing_yield"]["confocal_image_id"].tolist()


def test_spacing_outside_range_flagged(tmp_path: Path) -> None:
    pipeline_dir = write_pipeline_dir(tmp_path / "pipeline")

    _, _, _, summary, _ = audit_confocal_cohort(
        audit_config(tmp_path),
        pipeline_dir=pipeline_dir,
        output_directory=tmp_path / "audit",
    )

    flagged = {row["confocal_image_id"] for row in summary["spacing_median_outside_expected_range_images"]}
    assert "broad" in flagged


def test_summary_json_serializable(tmp_path: Path) -> None:
    pipeline_dir = write_pipeline_dir(tmp_path / "pipeline")

    _, _, _, summary, paths = audit_confocal_cohort(
        audit_config(tmp_path),
        pipeline_dir=pipeline_dir,
        output_directory=tmp_path / "audit",
    )

    json.dumps(summary)
    assert json.loads(paths["summary_json"].read_text(encoding="utf-8"))["mode"] == "confocal_larger_cohort_audit"


def test_existing_pipeline_outputs_are_not_modified(tmp_path: Path) -> None:
    pipeline_dir = write_pipeline_dir(tmp_path / "pipeline")
    source = pipeline_dir / "confocal_pipeline_per_image.csv"
    before = source.read_bytes()

    audit_confocal_cohort(audit_config(tmp_path), pipeline_dir=pipeline_dir, output_directory=tmp_path / "audit")

    assert source.read_bytes() == before
