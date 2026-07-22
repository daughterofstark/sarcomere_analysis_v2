from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sarcomere_analysis.confocal_striation_sensitivity import (
    classify_variant,
    generate_threshold_variants,
    run_confocal_striation_sensitivity,
)


def sensitivity_config(tmp_path: Path) -> dict:
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


def synthetic_patch_table() -> pd.DataFrame:
    rows = []
    specs = {
        "5138": {"good": 8, "bad": 2, "expected": True, "complex": False},
        "6052-CLEAR_STRIPES": {"good": 7, "bad": 3, "expected": True, "complex": False},
        "3112": {"good": 2, "bad": 8, "expected": False, "complex": True},
        "4049": {"good": 9, "bad": 1, "expected": False, "complex": False},
    }
    for image_id, spec in specs.items():
        for idx in range(spec["good"] + spec["bad"]):
            is_good = idx < spec["good"]
            rows.append(
                {
                    "confocal_image_id": image_id,
                    "filename": f"{image_id}.tif",
                    "patch_id": f"{image_id}_p{idx:05d}",
                    "y0": idx * 2,
                    "x0": idx * 2,
                    "y1": idx * 2 + 2,
                    "x1": idx * 2 + 2,
                    "gradient_energy": 0.01 if is_good else 0.00001,
                    "orientation_coherence": 0.8 if is_good else 0.1,
                    "intensity_std": 0.2 if is_good else 0.01,
                    "contrast": 0.5 if is_good else 0.02,
                    "signal_fraction": 0.5 if is_good else 0.01,
                    "saturation_fraction": 0.0,
                    "candidate_striation_region": is_good,
                    "expected_positive_example": spec["expected"],
                    "noted_complex_example": spec["complex"],
                }
            )
    return pd.DataFrame(rows)


def write_inputs(tmp_path: Path, patches: pd.DataFrame | None = None) -> tuple[Path, Path]:
    patches = synthetic_patch_table() if patches is None else patches
    patch_path = tmp_path / "mask" / "confocal_striation_mask_per_patch.csv"
    image_path = tmp_path / "mask" / "confocal_striation_mask_per_image.csv"
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patches.to_csv(patch_path, index=False)
    images = (
        patches.groupby(["confocal_image_id", "filename"], as_index=False)
        .agg(
            total_patches=("patch_id", "size"),
            candidate_patch_count=("candidate_striation_region", "sum"),
            expected_positive_example=("expected_positive_example", "any"),
            noted_complex_example=("noted_complex_example", "any"),
        )
    )
    images["candidate_patch_fraction"] = images["candidate_patch_count"] / images["total_patches"]
    images.to_csv(image_path, index=False)
    return patch_path, image_path


def test_variants_are_generated() -> None:
    variants = generate_threshold_variants(synthetic_patch_table())

    assert [variant["variant_id"] for variant in variants] == ["lenient", "default_current", "moderate", "strict", "very_strict"]


def test_candidate_fractions_computed_correctly(tmp_path: Path) -> None:
    patch_path, image_path = write_inputs(tmp_path)
    variants, per_image, _, _ = run_confocal_striation_sensitivity(
        sensitivity_config(tmp_path),
        patch_table=patch_path,
        image_table=image_path,
        output_directory=tmp_path / "out",
    )

    default = variants.loc[variants["variant_id"] == "default_current"].iloc[0]
    image_default = per_image.loc[(per_image["variant_id"] == "default_current") & (per_image["confocal_image_id"] == "5138")].iloc[0]
    assert default["candidate_fraction_5138"] == image_default["candidate_patch_fraction"]


def test_too_broad_classification_works() -> None:
    per_image = pd.DataFrame(
        {
            "candidate_patch_fraction": [0.95, 0.92, 0.91, 0.93],
            "expected_positive_example": [True, True, False, False],
            "noted_complex_example": [False, False, True, False],
        }
    )

    assert classify_variant(per_image) == "too_broad"


def test_too_sparse_classification_works() -> None:
    per_image = pd.DataFrame(
        {
            "candidate_patch_fraction": [0.02, 0.03, 0.01, 0.02],
            "expected_positive_example": [True, True, False, False],
            "noted_complex_example": [False, False, True, False],
        }
    )

    assert classify_variant(per_image) == "too_sparse"


def test_plausible_for_review_classification_works() -> None:
    per_image = pd.DataFrame(
        {
            "candidate_patch_fraction": [0.45, 0.35, 0.10, 0.30],
            "expected_positive_example": [True, True, False, False],
            "noted_complex_example": [False, False, True, False],
        }
    )

    assert classify_variant(per_image) == "plausible_for_review"


def test_expected_positive_and_complex_flags_reported(tmp_path: Path) -> None:
    patch_path, image_path = write_inputs(tmp_path)
    _, _, summary, _ = run_confocal_striation_sensitivity(
        sensitivity_config(tmp_path),
        patch_table=patch_path,
        image_table=image_path,
        output_directory=tmp_path / "out",
    )

    records = summary["expected_positive_and_complex_behavior"]
    assert any(row["expected_positive_example"] for row in records)
    assert any(row["noted_complex_example"] for row in records)


def test_summary_json_serializable(tmp_path: Path) -> None:
    patch_path, image_path = write_inputs(tmp_path)
    _, _, summary, paths = run_confocal_striation_sensitivity(
        sensitivity_config(tmp_path),
        patch_table=patch_path,
        image_table=image_path,
        output_directory=tmp_path / "out",
    )

    json.dumps(summary)
    assert paths["summary_json"].exists()


def test_handles_missing_optional_feature_columns_gracefully(tmp_path: Path) -> None:
    patches = synthetic_patch_table().drop(columns=["contrast"])
    patch_path, image_path = write_inputs(tmp_path, patches)
    variants, _, _, _ = run_confocal_striation_sensitivity(
        sensitivity_config(tmp_path),
        patch_table=patch_path,
        image_table=image_path,
        output_directory=tmp_path / "out",
    )

    assert "contrast" in variants["missing_feature_columns"].iloc[0]


def test_existing_confocal_mask_outputs_are_not_modified(tmp_path: Path) -> None:
    patch_path, image_path = write_inputs(tmp_path)
    before_patch = patch_path.read_bytes()
    before_image = image_path.read_bytes()

    run_confocal_striation_sensitivity(
        sensitivity_config(tmp_path),
        patch_table=patch_path,
        image_table=image_path,
        output_directory=tmp_path / "out",
    )

    assert patch_path.read_bytes() == before_patch
    assert image_path.read_bytes() == before_image
