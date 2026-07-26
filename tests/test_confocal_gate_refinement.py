from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sarcomere_analysis.confocal_gate_refinement import (
    DEFAULT_FOCUS_IMAGES,
    run_confocal_gate_refinement,
)


def refinement_config(tmp_path: Path) -> dict:
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


def write_refinement_inputs(tmp_path: Path, broad: bool = False) -> None:
    root = tmp_path / "results"
    mask_dir = root / "confocal_striation_mask"
    oop_dir = root / "confocal_same_grid_oop"
    spacing_dir = root / "confocal_spacing_audit"
    metadata_dir = root / "confocal_metadata"
    for directory in [mask_dir, oop_dir, spacing_dir, metadata_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_id in DEFAULT_FOCUS_IMAGES:
        expected = image_id in {"5138", "6052-CLEAR_STRIPES"}
        complex_flag = image_id == "3112"
        for idx in range(10):
            moderate = idx < 2
            near = 2 <= idx < (9 if broad else 4)
            rows.append(
                {
                    "confocal_image_id": image_id,
                    "filename": f"{image_id}.tif",
                    "patch_id": f"{image_id}_p{idx:05d}",
                    "y0": 0,
                    "x0": idx * 4,
                    "y1": 4,
                    "x1": idx * 4 + 4,
                    "center_y": 2,
                    "center_x": idx * 4 + 2,
                    "signal_fraction": 0.5 if moderate or near else 0.01,
                    "gradient_energy": 0.9 if moderate else (0.48 if near else 0.01),
                    "orientation_coherence": 0.9 if moderate else (0.48 if near else 0.01),
                    "intensity_mean": 0.5,
                    "intensity_std": 0.9 if moderate else (0.48 if near else 0.01),
                    "contrast": 0.9 if moderate else (0.48 if near else 0.01),
                    "saturation_fraction": 0.0,
                    "candidate_striation_region": moderate,
                    "candidate_reason": "ok" if moderate else "",
                    "rejection_reason": "ok" if moderate else "relaxed_candidate",
                    "expected_positive_example": expected,
                    "noted_complex_example": complex_flag,
                }
            )
    mask = pd.DataFrame(rows)
    mask.to_csv(mask_dir / "confocal_striation_mask_per_patch.csv", index=False)
    oop = mask[["confocal_image_id", "filename", "patch_id", "y0", "x0", "y1", "x1", "center_y", "center_x"]].copy()
    oop["candidate_striation_region"] = mask["candidate_striation_region"]
    oop["patch_oop_128"] = np.where(mask["candidate_striation_region"], 0.8, 0.5)
    oop["patch_mean_orientation_deg_128"] = 0.0
    oop["patch_orientation_coherence_mean_128"] = np.where(mask["candidate_striation_region"], 0.85, 0.55)
    oop.to_csv(oop_dir / "confocal_same_grid_oop_per_patch.csv", index=False)
    spacing = mask[["confocal_image_id", "filename", "patch_id"]].copy()
    spacing["spacing_valid"] = mask["candidate_striation_region"]
    spacing["spacing_estimate_um"] = np.where(mask["candidate_striation_region"], 1.8, np.nan)
    spacing["spacing_confidence"] = np.where(mask["candidate_striation_region"], 0.4, 0.0)
    spacing.to_csv(spacing_dir / "confocal_spacing_per_patch.csv", index=False)
    metadata = pd.DataFrame(
        [
            {
                "confocal_image_id": image_id,
                "source_path": str(tmp_path / f"{image_id}.tif"),
                "pixel_size_x_um": 0.06,
                "pixel_size_available": True,
                "spacing_um_enabled": True,
            }
            for image_id in DEFAULT_FOCUS_IMAGES
        ]
    )
    metadata.to_csv(metadata_dir / "confocal_metadata_calibration.csv", index=False)


def test_moderate_reference_preserved(tmp_path: Path) -> None:
    write_refinement_inputs(tmp_path)
    variants, _, summary, _ = run_confocal_gate_refinement(refinement_config(tmp_path), output_directory=tmp_path / "out")

    moderate = variants.loc[variants["variant_name"] == "moderate_reference"].iloc[0]
    assert moderate["total_candidate_patches"] == 8
    assert moderate["classification"] == "conservative_reference"
    assert summary["current_reference_variant"] == "moderate"


def test_relaxed_variant_adds_patches_relative_to_moderate(tmp_path: Path) -> None:
    write_refinement_inputs(tmp_path)
    variants, _, _, _ = run_confocal_gate_refinement(refinement_config(tmp_path), output_directory=tmp_path / "out")

    added = variants.loc[variants["variant_name"] == "moderate_relaxed_combined", "added_vs_moderate_count"].iloc[0]
    assert added > 0


def test_candidate_fraction_computed_correctly(tmp_path: Path) -> None:
    write_refinement_inputs(tmp_path)
    _, per_image, _, _ = run_confocal_gate_refinement(refinement_config(tmp_path), output_directory=tmp_path / "out")

    row = per_image.loc[
        (per_image["confocal_image_id"] == "5138") & (per_image["variant_name"] == "moderate_reference")
    ].iloc[0]
    assert row["candidate_fraction"] == 0.2


def test_broad_selection_warning_works(tmp_path: Path) -> None:
    write_refinement_inputs(tmp_path, broad=True)
    variants, per_image, _, _ = run_confocal_gate_refinement(refinement_config(tmp_path), output_directory=tmp_path / "out")

    assert "too_broad" in set(variants["classification"])
    flag = per_image.loc[
        (per_image["confocal_image_id"] == "7028") & (per_image["variant_name"] == "moderate_relaxed_combined"),
        "review_flag",
    ].iloc[0]
    assert "broad_selection_risk" in flag


def test_spacing_summaries_from_selected_patches(tmp_path: Path) -> None:
    write_refinement_inputs(tmp_path)
    variants, _, _, _ = run_confocal_gate_refinement(refinement_config(tmp_path), output_directory=tmp_path / "out")

    moderate = variants.loc[variants["variant_name"] == "moderate_reference"].iloc[0]
    assert moderate["selected_valid_spacing_count"] == 8
    assert moderate["selected_median_spacing_um"] == 1.8


def test_focus_image_reporting_works(tmp_path: Path) -> None:
    write_refinement_inputs(tmp_path)
    _, _, summary, _ = run_confocal_gate_refinement(refinement_config(tmp_path), output_directory=tmp_path / "out")

    assert set(summary["focus_image_summaries"]) == set(DEFAULT_FOCUS_IMAGES)
    assert len(summary["focus_image_summaries"]["3112"]) == 5


def test_summary_json_serializable(tmp_path: Path) -> None:
    write_refinement_inputs(tmp_path)
    _, _, summary, paths = run_confocal_gate_refinement(refinement_config(tmp_path), output_directory=tmp_path / "out")

    json.dumps(summary)
    assert paths["summary_json"].exists()


def test_preview_writing_disabled_by_default(tmp_path: Path) -> None:
    write_refinement_inputs(tmp_path)
    _, _, summary, paths = run_confocal_gate_refinement(refinement_config(tmp_path), output_directory=tmp_path / "out")

    assert summary["previews_written"] is False
    assert not paths["previews"].exists()


def test_existing_confocal_outputs_are_not_modified(tmp_path: Path) -> None:
    write_refinement_inputs(tmp_path)
    source = tmp_path / "results" / "confocal_spacing_audit" / "confocal_spacing_per_patch.csv"
    before = source.read_bytes()

    run_confocal_gate_refinement(refinement_config(tmp_path), output_directory=tmp_path / "out")

    assert source.read_bytes() == before
