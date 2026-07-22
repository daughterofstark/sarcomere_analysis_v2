from __future__ import annotations

import json
from pathlib import Path

from sarcomere_analysis.confocal_pilot_report import build_confocal_pilot_report, write_confocal_pilot_report


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


def write_confocal_summaries(tmp_path: Path) -> None:
    root = tmp_path / "results"
    write_json(
        root / "confocal_baseline" / "confocal_baseline_summary.json",
        {
            "confocal_image_count": 11,
            "processed_ok": 11,
            "processed_error": 0,
            "filenames": ["5138.tif", "6052-CLEAR_STRIPES.tif", "3112.tif"],
            "expected_positive_examples": [
                {"confocal_image_id": "5138", "filename": "5138.tif"},
                {"confocal_image_id": "6052-CLEAR_STRIPES", "filename": "6052-CLEAR_STRIPES.tif"},
            ],
            "noted_complex_examples": [{"confocal_image_id": "3112", "filename": "3112.tif"}],
            "patch_rows": 2475,
            "orientation_summary": {
                "valid_orientation_patch_count_total": 2,
                "valid_orientation_patch_fraction_median": 0.0,
                "image_oop_median": 0.5509,
            },
            "spacing_calibration_status": "confocal_pixel_size_unknown_spacing_um_not_reported",
        },
    )
    write_json(
        root / "confocal_striation_mask" / "confocal_striation_mask_summary.json",
        {"candidate_patch_fraction": 0.83975},
    )
    write_json(
        root / "confocal_striation_sensitivity" / "confocal_striation_sensitivity_summary.json",
        {
            "default_assessment": {"variant_id": "default_current", "classification": "too_broad"},
            "why_default_was_too_broad": "The default/current gate was too broad.",
            "best_plausible_variants": [
                {
                    "variant_id": "moderate",
                    "classification": "plausible_for_review",
                    "candidate_fraction_5138": 0.3215,
                    "candidate_fraction_6052": 0.2924,
                    "candidate_fraction_3112": 0.0968,
                }
            ],
        },
    )
    write_json(
        root / "confocal_selective_analysis" / "confocal_selective_summary.json",
        {
            "selected_variant": "moderate",
            "candidate_patch_count": 2330,
            "candidate_fraction_by_image": [
                {"confocal_image_id": "7028", "candidate_patch_fraction": 0.6223},
            ],
            "selected_vs_all_comparison": {
                "median_selected_region_coherence": 0.7260,
                "median_all_region_coherence": 0.6812,
                "median_selected_region_gradient_energy": 0.00538,
                "median_all_region_gradient_energy": 0.00309,
            },
        },
    )
    write_json(
        root / "confocal_same_grid_oop" / "confocal_same_grid_oop_summary.json",
        {
            "same_grid_patch_rows": 10571,
            "patches_processed_ok": 10571,
            "patches_error": 0,
            "candidate_patch_count": 2330,
            "selected_vs_all_oop_summary": {
                "median_selected_region_oop_128": 0.7085,
                "median_all_region_oop_128": 0.6140,
                "median_selected_vs_all_oop_difference_128": 0.0653,
                "median_selected_region_coherence_128": 0.6771,
                "median_all_region_coherence_128": 0.6395,
            },
            "selected_region_summaries": [
                {
                    "confocal_image_id": "5138",
                    "filename": "5138.tif",
                    "candidate_patch_fraction": 0.3215,
                    "selected_region_median_oop_128": 0.8391,
                    "all_region_median_oop_128": 0.7773,
                },
                {
                    "confocal_image_id": "6052-CLEAR_STRIPES",
                    "filename": "6052-CLEAR_STRIPES.tif",
                    "candidate_patch_fraction": 0.2924,
                    "selected_region_median_oop_128": 0.6793,
                    "all_region_median_oop_128": 0.6140,
                },
                {
                    "confocal_image_id": "3112",
                    "filename": "3112.tif",
                    "candidate_patch_fraction": 0.0968,
                    "selected_region_median_oop_128": 0.6834,
                    "all_region_median_oop_128": 0.6763,
                },
                {
                    "confocal_image_id": "7028",
                    "filename": "7028.tif",
                    "candidate_patch_fraction": 0.6223,
                    "interpretation_flag": "broad_candidate_fraction_review_needed",
                },
            ],
            "spacing_status": "not_computed_in_microns_confocal_pixel_size_unknown",
        },
    )


def test_handles_missing_optional_summaries(tmp_path: Path) -> None:
    report = build_confocal_pilot_report(report_config(tmp_path))

    assert report["source_summary_presence"]["baseline"] is False
    assert report["confocal_dataset_intake"]["image_count"] is None
    assert report["final_confocal_pilot_classification"] == "selective_region_analysis_feasible_exploratory_needs_manual_review"


def test_includes_baseline_transfer_audit(tmp_path: Path) -> None:
    write_confocal_summaries(tmp_path)
    report = build_confocal_pilot_report(report_config(tmp_path))

    baseline = report["baseline_transfer_audit"]
    assert baseline["valid_orientation_patch_count_total"] == 2
    assert baseline["conclusion"] == "widefield_qc_not_transferable_unchanged"
    assert "2/2475" in baseline["interpretation"]


def test_includes_moderate_gate_classification(tmp_path: Path) -> None:
    write_confocal_summaries(tmp_path)
    report = build_confocal_pilot_report(report_config(tmp_path))

    mask = report["selective_confident_striation_mask"]
    assert mask["default_gate_assessment"] == "too_broad"
    assert mask["moderate_gate_classification"] == "plausible_for_review"
    assert mask["moderate_candidate_fractions"]["5138"] == 0.3215


def test_includes_same_grid_oop_selected_vs_all_result(tmp_path: Path) -> None:
    write_confocal_summaries(tmp_path)
    report = build_confocal_pilot_report(report_config(tmp_path))

    same_grid = report["same_grid_selected_region_oop"]
    assert same_grid["same_grid_patch_rows"] == 10571
    assert same_grid["selected_vs_all_oop_summary"]["median_selected_vs_all_oop_difference_128"] == 0.0653
    assert same_grid["positive_examples"]["5138"]["selected_region_median_oop_128"] == 0.8391


def test_includes_allowed_and_not_allowed_claims(tmp_path: Path) -> None:
    write_confocal_summaries(tmp_path)
    report = build_confocal_pilot_report(report_config(tmp_path))

    assert any("moderate selective mask is plausible" in claim for claim in report["claims_allowed"])
    assert any("Spacing in microns is measured" in claim for claim in report["claims_not_allowed"])


def test_output_json_serializable(tmp_path: Path) -> None:
    write_confocal_summaries(tmp_path)
    report = build_confocal_pilot_report(report_config(tmp_path))

    json.dumps(report)


def test_markdown_written(tmp_path: Path) -> None:
    write_confocal_summaries(tmp_path)
    report, paths = write_confocal_pilot_report(report_config(tmp_path), docs_directory=tmp_path / "docs")

    assert paths["json"].exists()
    assert paths["txt"].exists()
    assert paths["markdown"].exists()
    assert "# Confocal Pilot Interpretation" in paths["markdown"].read_text(encoding="utf-8")
    assert report["answer_to_natalia"]["short_answer"] == "yes_feasible_exploratory"
