from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sarcomere_analysis.confocal_selective_analysis import run_confocal_selective_analysis


def selective_config(tmp_path: Path) -> dict:
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


def synthetic_patches() -> pd.DataFrame:
    rows = []
    specs = [
        ("5138", "5138.tif", True, False, [0.9, 0.8, 0.2, 0.1]),
        ("3112", "3112.tif", False, True, [0.7, 0.1, 0.1, 0.1]),
    ]
    for image_id, filename, expected, complex_flag, coherences in specs:
        for idx, coherence in enumerate(coherences):
            good = coherence >= 0.5
            rows.append(
                {
                    "confocal_image_id": image_id,
                    "filename": filename,
                    "patch_id": f"{image_id}_p{idx:05d}",
                    "y0": idx * 4,
                    "x0": idx * 4,
                    "y1": idx * 4 + 4,
                    "x1": idx * 4 + 4,
                    "center_y": idx * 4 + 2,
                    "center_x": idx * 4 + 2,
                    "orientation_coherence": coherence,
                    "gradient_energy": 0.01 if good else 0.0001,
                    "intensity_std": 0.2 if good else 0.01,
                    "contrast": 0.5 if good else 0.02,
                    "signal_fraction": 0.5 if good else 0.01,
                    "saturation_fraction": 0.0,
                    "candidate_striation_region": good,
                    "expected_positive_example": expected,
                    "noted_complex_example": complex_flag,
                }
            )
    return pd.DataFrame(rows)


def synthetic_baseline_patches(patches: pd.DataFrame | None = None, include_oop: bool = True, coordinate_shift: int = 0) -> pd.DataFrame:
    patches = synthetic_patches() if patches is None else patches
    baseline = patches[["confocal_image_id", "filename", "patch_id", "y0", "x0", "y1", "x1"]].copy()
    if coordinate_shift:
        baseline["x0"] = baseline["x0"] + coordinate_shift
        baseline["x1"] = baseline["x1"] + coordinate_shift
    if include_oop:
        baseline["patch_oop"] = pd.to_numeric(patches["orientation_coherence"], errors="coerce") / 2
        baseline["patch_mean_orientation_deg"] = 45.0
        baseline["patch_orientation_weight_sum"] = 1.0
        baseline["patch_orientation_valid_pixels"] = 16
    return baseline


def synthetic_variants() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant_id": "moderate",
                "min_gradient_energy": 0.001,
                "min_orientation_coherence": 0.5,
                "min_intensity_std": 0.05,
                "min_contrast": 0.1,
                "min_signal_fraction": 0.05,
                "max_saturation_fraction": 0.1,
            }
        ]
    )


def synthetic_sensitivity_per_image() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant_id": "moderate",
                "confocal_image_id": "5138",
                "filename": "5138.tif",
                "candidate_patch_fraction": 0.5,
                "expected_positive_example": True,
                "noted_complex_example": False,
            },
            {
                "variant_id": "moderate",
                "confocal_image_id": "3112",
                "filename": "3112.tif",
                "candidate_patch_fraction": 0.25,
                "expected_positive_example": False,
                "noted_complex_example": True,
            },
        ]
    )


def write_inputs(
    tmp_path: Path,
    patches: pd.DataFrame | None = None,
    baseline: pd.DataFrame | None = None,
) -> tuple[Path, Path, Path, Path]:
    patch_path = tmp_path / "patches.csv"
    baseline_path = tmp_path / "baseline.csv"
    variant_path = tmp_path / "variants.csv"
    sensitivity_image_path = tmp_path / "sensitivity_per_image.csv"
    patches = patches if patches is not None else synthetic_patches()
    patches.to_csv(patch_path, index=False)
    (baseline if baseline is not None else synthetic_baseline_patches(patches)).to_csv(baseline_path, index=False)
    synthetic_variants().to_csv(variant_path, index=False)
    synthetic_sensitivity_per_image().to_csv(sensitivity_image_path, index=False)
    return patch_path, baseline_path, variant_path, sensitivity_image_path


def run_synthetic(
    tmp_path: Path,
    patches: pd.DataFrame | None = None,
    baseline: pd.DataFrame | None = None,
    **kwargs,
):
    patch_path, baseline_path, variant_path, sensitivity_path = write_inputs(tmp_path, patches, baseline)
    return run_confocal_selective_analysis(
        selective_config(tmp_path),
        selected_variant="moderate",
        patch_table=patch_path,
        baseline_patch_table=baseline_path,
        sensitivity_variants=variant_path,
        sensitivity_per_image=sensitivity_path,
        output_directory=tmp_path / "out",
        **kwargs,
    )


def test_selected_variant_filters_candidate_patches_correctly(tmp_path: Path) -> None:
    per_patch, _, _, _ = run_synthetic(tmp_path)

    selected = per_patch.loc[per_patch["candidate_striation_region"]]
    assert len(selected) == 3
    assert set(selected["confocal_image_id"]) == {"5138", "3112"}


def test_per_image_candidate_fraction_computed_correctly(tmp_path: Path) -> None:
    _, per_image, _, _ = run_synthetic(tmp_path)

    fraction = per_image.loc[per_image["confocal_image_id"] == "5138", "candidate_patch_fraction"].iloc[0]
    assert fraction == 0.5


def test_selected_region_medians_use_only_candidates(tmp_path: Path) -> None:
    _, per_image, _, _ = run_synthetic(tmp_path)

    median = per_image.loc[per_image["confocal_image_id"] == "5138", "selected_region_median_coherence"].iloc[0]
    assert round(median, 3) == 0.85


def test_all_region_medians_use_all_patches(tmp_path: Path) -> None:
    _, per_image, _, _ = run_synthetic(tmp_path)

    median = per_image.loc[per_image["confocal_image_id"] == "5138", "all_region_median_coherence"].iloc[0]
    assert median == 0.5


def test_selected_vs_all_oop_difference_computed(tmp_path: Path) -> None:
    _, per_image, _, _ = run_synthetic(tmp_path)

    diff = per_image.loc[per_image["confocal_image_id"] == "5138", "selected_vs_all_oop_difference"].iloc[0]
    assert round(diff, 3) == 0.175


def test_baseline_patch_features_join_by_image_and_patch_id(tmp_path: Path) -> None:
    per_patch, _, summary, _ = run_synthetic(tmp_path)

    assert summary["baseline_patch_join_audit"]["matched_rows"] == len(per_patch)
    assert summary["baseline_patch_join_audit"]["oop_columns_found"] == [
        "patch_oop",
        "patch_mean_orientation_deg",
        "patch_orientation_weight_sum",
        "patch_orientation_valid_pixels",
    ]
    assert per_patch["baseline_patch_oop"].notna().all()


def test_too_few_candidates_flag_works(tmp_path: Path) -> None:
    _, per_image, _, _ = run_synthetic(tmp_path, min_candidate_patches=3)

    flag = per_image.loc[per_image["confocal_image_id"] == "3112", "interpretation_flag"].iloc[0]
    assert "too_few_candidates" in flag


def test_expected_positive_and_complex_flags_preserved(tmp_path: Path) -> None:
    _, per_image, summary, _ = run_synthetic(tmp_path)

    assert bool(per_image.loc[per_image["confocal_image_id"] == "5138", "expected_positive_example"].iloc[0])
    assert bool(per_image.loc[per_image["confocal_image_id"] == "3112", "noted_complex_example"].iloc[0])
    assert len(summary["selected_region_summaries"]) == 2


def test_summary_json_serializable(tmp_path: Path) -> None:
    _, _, summary, paths = run_synthetic(tmp_path)

    json.dumps(summary)
    assert paths["summary_json"].exists()


def test_missing_baseline_oop_columns_handled_gracefully(tmp_path: Path) -> None:
    patches = synthetic_patches()
    baseline = synthetic_baseline_patches(patches, include_oop=False)
    _, per_image, summary, _ = run_synthetic(tmp_path, patches, baseline)

    assert per_image["selected_region_median_oop"].isna().all()
    assert summary["selected_vs_all_comparison"]["oop_available"] is False
    assert summary["baseline_patch_join_audit"]["oop_summary_reason"] == "no_oop_columns_found"


def test_unmatched_baseline_rows_reported(tmp_path: Path) -> None:
    patches = synthetic_patches()
    baseline = synthetic_baseline_patches(patches).iloc[:2].copy()
    _, _, summary, _ = run_synthetic(tmp_path, patches, baseline)

    assert summary["baseline_patch_join_audit"]["matched_rows"] == 2
    assert summary["baseline_patch_join_audit"]["unmatched_rows"] == len(patches) - 2


def test_coordinate_mismatch_reported(tmp_path: Path) -> None:
    patches = synthetic_patches()
    baseline = synthetic_baseline_patches(patches, coordinate_shift=1)
    _, per_image, summary, _ = run_synthetic(tmp_path, patches, baseline)

    assert summary["baseline_patch_join_audit"]["coordinate_mismatch_count"] == len(patches)
    assert summary["baseline_patch_join_audit"]["oop_summary_reason"] == "baseline_patch_grid_coordinate_mismatch"
    assert per_image["selected_region_median_oop"].isna().all()


def test_previews_disabled_by_default(tmp_path: Path) -> None:
    _, _, summary, paths = run_synthetic(tmp_path)

    assert summary["previews_written"] is False
    assert summary["preview_paths"] == []
    assert not paths["previews"].exists()


def test_existing_outputs_are_not_modified(tmp_path: Path) -> None:
    patch_path, baseline_path, variant_path, sensitivity_path = write_inputs(tmp_path)
    before_patch = patch_path.read_bytes()
    before_baseline = baseline_path.read_bytes()
    before_variant = variant_path.read_bytes()
    before_sensitivity = sensitivity_path.read_bytes()

    run_confocal_selective_analysis(
        selective_config(tmp_path),
        selected_variant="moderate",
        patch_table=patch_path,
        baseline_patch_table=baseline_path,
        sensitivity_variants=variant_path,
        sensitivity_per_image=sensitivity_path,
        output_directory=tmp_path / "out",
    )

    assert patch_path.read_bytes() == before_patch
    assert baseline_path.read_bytes() == before_baseline
    assert variant_path.read_bytes() == before_variant
    assert sensitivity_path.read_bytes() == before_sensitivity
