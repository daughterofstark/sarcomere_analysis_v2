from __future__ import annotations

import json
from pathlib import Path

from sarcomere_analysis.final_validation_report import build_final_validation_report, write_final_validation_report


def report_config(tmp_path: Path) -> dict:
    return {
        "paths": {"raw_tiff_dir": str(tmp_path / "raw"), "output_dir": str(tmp_path / "results")},
        "outputs": {"manifest_csv": str(tmp_path / "results" / "tables" / "manifest.csv")},
        "calibration": {
            "pixel_size_um": 0.1299,
            "expected_sarcomere_spacing_um": {"min": 1.5, "max": 2.4},
        },
        "filename_pattern": {"regex": r"^(?P<donor_id>\d+\.\d+)-(?P<region_id>\d+)$"},
        "run": {"include_extensions": [".tif", ".tiff"], "recursive": False},
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_report_summaries(tmp_path: Path) -> None:
    validation = tmp_path / "results" / "validation"
    tables = tmp_path / "results" / "tables"
    write_json(
        validation / "synthetic_oop_validation_summary.json",
        {
            "synthetic_examples": 72,
            "clean_case_median_angular_error_deg": 0.318,
            "clean_case_max_angular_error_deg": 0.318,
            "oop_monotonicity_low_gt_medium_gt_high": True,
            "recovered_oop_median_by_disorder_level": {"low": 0.998, "medium": 0.909, "high": 0.844},
            "degradation_failure_modes": ["high_noise_increases_error"],
        },
    )
    write_json(
        validation / "expert_annotation_validation" / "expert_annotation_validation_summary.json",
        {
            "audit": {
                "total_rows": 75,
                "annotations_matched_to_internal_key": 75,
                "completed_striations_visible_count": 75,
                "completed_organisation_score_count": 51,
                "completed_confidence_score_count": 53,
                "manual_sarcomere_length_completed_count": 0,
            },
            "visibility_vs_automated_oop": {"oop_medians": {"yes": 0.0865, "unclear": 0.0702, "no": 0.0583}},
            "organisation_score_vs_automated_oop": {
                "spearman": {"computed": True, "n": 51, "rho": 0.0124, "p_value": 0.931}
            },
            "confidence_filtered": {
                "organisation_oop_spearman": {"computed": True, "n": 41, "rho": 0.00157, "p_value": 0.9922}
            },
            "orientation": {"dominant_orientation_used_as_primary": False},
        },
    )
    write_json(
        validation / "expert_feature_audit" / "expert_feature_audit_summary.json",
        {
            "audit": {"numeric_automated_features_considered": 19},
            "top_organisation_features_by_abs_spearman": [{"feature": "intensity_mean", "spearman_rho": -0.21}],
            "oop_specific_statement": {"statement": "patch_oop alone had near-zero organisation correlation."},
        },
    )
    write_json(
        validation / "full_image_patch_mask_validation_summary.json",
        {
            "total_automated_patches_in_annotated_images": 2700,
            "n_orientation_pairs": 183,
            "median_axial_error_deg": 47.71,
        },
    )
    write_json(
        tables / "feature_assembly_summary.json",
        {
            "per_patch_rows": 32625,
            "per_image_rows": 145,
            "per_donor_rows": 29,
            "donor_count": 29,
            "spacing_global_status": "exploratory_low_yield",
            "total_valid_spacing_patches": 14,
        },
    )


def test_report_handles_missing_optional_summaries(tmp_path: Path) -> None:
    report = build_final_validation_report(report_config(tmp_path))

    assert report["synthetic_oop_implementation_validation"]["status"] == "missing"
    assert report["source_summary_presence"]["expert_annotation"] is False
    assert report["final_interpretation"]["real_tissue_oop_as_expert_organisation_endpoint"] == "not_validated"


def test_report_includes_synthetic_validation_status(tmp_path: Path) -> None:
    write_report_summaries(tmp_path)
    report = build_final_validation_report(report_config(tmp_path))

    synthetic = report["synthetic_oop_implementation_validation"]
    assert synthetic["status"] == "implementation_validated_on_controlled_synthetic_data"
    assert synthetic["synthetic_examples"] == 72
    assert synthetic["oop_monotonicity_low_gt_medium_gt_high"] is True


def test_report_includes_expert_annotation_negative_validation(tmp_path: Path) -> None:
    write_report_summaries(tmp_path)
    report = build_final_validation_report(report_config(tmp_path))

    expert = report["manual_expert_validation_summary"]["blinded_expert_annotations"]
    assert expert["matched_rows"] == 75
    assert expert["organisation_vs_oop_spearman"]["rho"] == 0.0124
    assert report["final_interpretation"]["real_tissue_oop_as_expert_organisation_endpoint"] == "not_validated"


def test_report_includes_spacing_exploratory_low_yield(tmp_path: Path) -> None:
    write_report_summaries(tmp_path)
    report = build_final_validation_report(report_config(tmp_path))

    spacing = report["sarcgraph_zdisc_detection_interpretation"]
    assert spacing["spacing_status"] == "exploratory_low_yield"
    assert spacing["valid_spacing_patches"] == 14
    assert report["final_interpretation"]["sarcomere_spacing"] == "not_validated_exploratory_low_yield"


def test_report_includes_allowed_and_not_allowed_claims(tmp_path: Path) -> None:
    write_report_summaries(tmp_path)
    report = build_final_validation_report(report_config(tmp_path))

    assert any("pipeline processes" in claim for claim in report["claims_allowed"])
    assert any("OOP is validated as expert-rated" in claim for claim in report["claims_not_allowed"])


def test_output_json_serializable(tmp_path: Path) -> None:
    write_report_summaries(tmp_path)
    report = build_final_validation_report(report_config(tmp_path))

    json.dumps(report)


def test_markdown_doc_written(tmp_path: Path) -> None:
    write_report_summaries(tmp_path)
    report, paths = write_final_validation_report(
        report_config(tmp_path),
        docs_directory=tmp_path / "docs",
    )

    assert paths["json"].exists()
    assert paths["txt"].exists()
    assert paths["markdown"].exists()
    assert "# Final Validation Interpretation" in paths["markdown"].read_text(encoding="utf-8")
    assert report["final_interpretation"]["oop_orientation_implementation"] == "validated_on_synthetic_controlled_data"


def test_no_production_tables_modified(tmp_path: Path) -> None:
    write_report_summaries(tmp_path)
    production_table = tmp_path / "results" / "tables" / "features_per_patch.csv"
    production_table.write_text("image_id,patch_id,patch_oop\n2.007-1,0,0.1\n", encoding="utf-8")
    before = production_table.read_bytes()

    write_final_validation_report(report_config(tmp_path), docs_directory=tmp_path / "docs")

    assert production_table.read_bytes() == before
