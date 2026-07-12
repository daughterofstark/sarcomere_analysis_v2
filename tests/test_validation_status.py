from __future__ import annotations

import json
from pathlib import Path

from sarcomere_analysis.validation_status import build_validation_status, write_validation_status


def status_config(tmp_path: Path) -> dict:
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


def write_synthetic_summaries(tmp_path: Path) -> None:
    validation = tmp_path / "results" / "validation"
    tables = tmp_path / "results" / "tables"
    write_json(
        validation / "synthetic_oop_validation_summary.json",
        {
            "synthetic_examples": 72,
            "clean_case_median_angular_error_deg": 0.318,
            "clean_case_max_angular_error_deg": 0.318,
            "oop_monotonicity_low_gt_medium_gt_high": True,
            "recovered_oop_median_by_disorder_level": {"low": 0.99, "medium": 0.9, "high": 0.8},
            "degradation_failure_modes": ["ok"],
        },
    )
    write_json(
        validation / "zdisc_mask_validation_summary.json",
        {
            "total_annotation_masks": 40,
            "masks_with_zdisc_labels": 12,
            "n_orientation_pairs": 12,
            "median_axial_error_deg": 35.0,
            "oop_medians_by_annotation_status": {"zdisc_labeled": 0.1, "empty": 0.2},
        },
    )
    write_json(
        validation / "full_image_zdisc_mask_validation_summary.json",
        {
            "total_full_image_annotations": 12,
            "images_with_zdisc_labels": 7,
            "n_orientation_pairs": 7,
            "median_axial_error_deg": 55.84,
            "oop_medians_by_annotation_status": {"zdisc_labeled": 0.4, "empty": 0.2},
        },
    )
    write_json(
        validation / "full_image_patch_mask_validation_summary.json",
        {
            "total_automated_patches_in_annotated_images": 2700,
            "patches_with_manual_zdisc_labels": 261,
            "n_orientation_pairs": 183,
            "median_axial_error_deg": 47.71,
            "oop_medians_by_manual_patch_status": {"zdisc_labeled": 0.07, "empty": 0.07},
            "spearman_zdisc_fraction_vs_patch_oop": {"computed": True, "rho": -0.027},
        },
    )
    write_json(
        tables / "feature_assembly_summary.json",
        {"spacing_global_status": "exploratory_low_yield", "total_valid_spacing_patches": 14},
    )


def test_status_handles_missing_optional_validation_summaries(tmp_path: Path) -> None:
    status = build_validation_status(status_config(tmp_path))

    assert status["synthetic_oop_validation"]["status"] == "missing"
    assert status["manual_crop_zdisc_mask_validation"]["status"] == "missing"
    assert status["spacing"]["status"] == "exploratory_low_yield"


def test_synthetic_status_is_recorded_when_summary_exists(tmp_path: Path) -> None:
    write_synthetic_summaries(tmp_path)
    status = build_validation_status(status_config(tmp_path))

    assert status["synthetic_oop_validation"]["status"] == "controlled_implementation_validated"
    assert status["synthetic_oop_validation"]["synthetic_example_count"] == 72


def test_manual_mask_statuses_are_pilot_only(tmp_path: Path) -> None:
    write_synthetic_summaries(tmp_path)
    status = build_validation_status(status_config(tmp_path))

    assert status["manual_crop_zdisc_mask_validation"]["status"] == "pilot_only_not_confirmatory"
    assert status["manual_full_image_zdisc_mask_validation"]["status"] == "pilot_only_not_confirmatory"
    assert status["manual_full_image_patch_mask_validation"]["status"] == "pilot_only_not_confirmatory"


def test_spacing_status_remains_exploratory_low_yield(tmp_path: Path) -> None:
    write_synthetic_summaries(tmp_path)
    status = build_validation_status(status_config(tmp_path))

    assert status["spacing"]["status"] == "exploratory_low_yield"
    assert status["spacing"]["valid_spacing_patches"] == 14


def test_overall_decision_mentions_real_tissue_oop_unresolved(tmp_path: Path) -> None:
    write_synthetic_summaries(tmp_path)
    status = build_validation_status(status_config(tmp_path))

    assert status["overall_validation_decision"]["real_tissue_oop"] == "not_expert_validated_unresolved"
    assert "real-tissue oop therefore remains unresolved" in status["overall_validation_decision"]["plain_language_summary"].lower()


def test_output_json_is_serializable(tmp_path: Path) -> None:
    write_synthetic_summaries(tmp_path)
    status = build_validation_status(status_config(tmp_path))

    json.dumps(status)


def test_markdown_doc_is_written(tmp_path: Path) -> None:
    write_synthetic_summaries(tmp_path)
    status, paths = write_validation_status(status_config(tmp_path), docs_directory=tmp_path / "docs")

    assert paths["summary_json"].exists()
    assert paths["summary_txt"].exists()
    assert paths["markdown"].exists()
    assert "# Validation Status" in paths["markdown"].read_text(encoding="utf-8")
    assert status["synthetic_oop_validation"]["status"] == "controlled_implementation_validated"
