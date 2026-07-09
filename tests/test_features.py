from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sarcomere_analysis.features import (
    FEATURE_DONOR_COLUMNS,
    FEATURE_IMAGE_COLUMNS,
    FEATURE_PATCH_COLUMNS,
    assemble_feature_tables,
    write_feature_outputs,
)


def synthetic_patch_table(include_spacing: bool = True) -> pd.DataFrame:
    rows = []
    for image_id, donor_id, spacing_valid in [
        ("2.001-1", "2.001", [True, False, False]),
        ("2.001-2", "2.001", [False, False, False]),
        ("3.002-1", "3.002", [True, True, True]),
    ]:
        for index, valid_spacing in enumerate(spacing_valid):
            rows.append(
                {
                    "image_id": image_id,
                    "donor_id": donor_id,
                    "patch_id": f"{image_id}_p{index:05d}",
                    "y0": 0,
                    "x0": 0,
                    "y1": 64,
                    "x1": 64,
                    "center_y": 32,
                    "center_x": 32,
                    "tissue_fraction": 0.8,
                    "intensity_mean": 0.5,
                    "intensity_std": 0.1 + index * 0.01,
                    "rms_contrast": 0.1 + index * 0.01,
                    "gradient_energy": 0.02 + index * 0.01,
                    "valid_for_orientation": True,
                    "valid_for_periodicity": True,
                    "valid_for_spacing": True,
                    "invalid_reason": "ok",
                    "patch_oop": 0.2 + index * 0.1,
                    "patch_mean_orientation_rad": 0.0,
                    "patch_mean_orientation_deg": 0.0,
                    "patch_orientation_weight_sum": 1.0,
                    "patch_orientation_valid_pixels": 100,
                    "patch_spacing_um": 1.6 if valid_spacing else np.nan,
                    "patch_spacing_px": 12.0 if valid_spacing else np.nan,
                    "patch_periodicity_score": 0.3 if valid_spacing else np.nan,
                    "patch_spacing_confidence": 0.2 if valid_spacing else np.nan,
                    "patch_spacing_method": "autocorrelation",
                    "valid_for_spacing_final": valid_spacing,
                    "spacing_invalid_reason": "ok" if valid_spacing else "no_local_peak",
                }
            )
    table = pd.DataFrame(rows)
    if not include_spacing:
        table = table.drop(columns=["patch_spacing_um", "patch_spacing_px", "patch_periodicity_score", "patch_spacing_confidence", "valid_for_spacing_final", "spacing_invalid_reason"])
    return table


def synthetic_image_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_id": "2.001-1",
                "donor_id": "2.001",
                "tissue_fraction": 0.75,
                "total_patches": 3,
                "valid_orientation_patches": 3,
                "image_oop": 0.3,
                "image_mean_orientation_rad": 0.0,
                "image_mean_orientation_deg": 0.0,
                "image_oop_heterogeneity": 0.05,
                "n_orientation_valid_patches": 3,
                "image_spacing_mean_um": 1.6,
                "image_spacing_median_um": 1.6,
                "image_spacing_std_um": 0.0,
                "image_spacing_cv": 0.0,
                "n_spacing_valid_patches": 1,
                "spacing_valid_fraction": 1 / 3,
            },
            {
                "image_id": "2.001-2",
                "donor_id": "2.001",
                "tissue_fraction": 0.7,
                "total_patches": 3,
                "valid_orientation_patches": 3,
                "image_oop": 0.5,
                "image_mean_orientation_rad": 0.1,
                "image_mean_orientation_deg": 5.7,
                "image_oop_heterogeneity": 0.02,
                "n_orientation_valid_patches": 3,
                "n_spacing_valid_patches": 0,
                "spacing_valid_fraction": 0.0,
            },
            {
                "image_id": "3.002-1",
                "donor_id": "3.002",
                "tissue_fraction": 0.9,
                "total_patches": 3,
                "valid_orientation_patches": 3,
                "image_oop": 0.7,
                "image_mean_orientation_rad": 0.2,
                "image_mean_orientation_deg": 11.5,
                "image_oop_heterogeneity": 0.03,
                "n_orientation_valid_patches": 3,
                "n_spacing_valid_patches": 3,
                "spacing_valid_fraction": 1.0,
            },
        ]
    )


def synthetic_batch_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"image_id": "2.001-1", "donor_id": "2.001", "status": "ok"},
            {"image_id": "2.001-2", "donor_id": "2.001", "status": "ok"},
            {"image_id": "3.002-1", "donor_id": "3.002", "status": "ok"},
        ]
    )


def test_per_image_feature_assembly_preserves_one_row_per_image() -> None:
    _, per_image, _, _ = assemble_feature_tables(synthetic_patch_table(), synthetic_image_table(), synthetic_batch_summary())
    assert len(per_image) == 3
    assert per_image["image_id"].is_unique


def test_per_donor_aggregation_preserves_one_row_per_donor() -> None:
    _, _, per_donor, _ = assemble_feature_tables(synthetic_patch_table(), synthetic_image_table(), synthetic_batch_summary())
    assert len(per_donor) == 2
    assert set(per_donor["donor_id"]) == {"2.001", "3.002"}


def test_oop_values_remain_in_unit_interval_or_nan() -> None:
    _, per_image, per_donor, _ = assemble_feature_tables(synthetic_patch_table(), synthetic_image_table(), synthetic_batch_summary())
    assert per_image["image_oop"].dropna().between(0, 1).all()
    assert per_donor["median_image_oop"].dropna().between(0, 1).all()


def test_spacing_low_yield_flag_set_below_threshold() -> None:
    _, per_image, _, _ = assemble_feature_tables(
        synthetic_patch_table(),
        synthetic_image_table(),
        synthetic_batch_summary(),
        min_spacing_patches_per_image=5,
    )
    assert per_image.loc[per_image["image_id"] == "2.001-1", "spacing_low_yield_flag"].iloc[0]
    assert per_image.loc[per_image["image_id"] == "2.001-1", "spacing_endpoint_status"].iloc[0] == "insufficient_patch_yield"


def test_donor_spacing_status_insufficient_when_yield_low() -> None:
    _, _, per_donor, _ = assemble_feature_tables(
        synthetic_patch_table(),
        synthetic_image_table(),
        synthetic_batch_summary(),
        min_spacing_patches_per_donor=5,
    )
    assert (per_donor["spacing_endpoint_status"] == "insufficient_patch_yield").all()


def test_missing_required_primary_oop_columns_fails_clearly() -> None:
    image_table = synthetic_image_table().drop(columns=["image_oop"])
    with pytest.raises(ValueError, match="image_oop"):
        assemble_feature_tables(synthetic_patch_table(), image_table, synthetic_batch_summary())


def test_optional_spacing_columns_missing_warns_instead_of_crashing() -> None:
    patch_table = synthetic_patch_table(include_spacing=False)
    image_table = synthetic_image_table().drop(columns=["n_spacing_valid_patches", "spacing_valid_fraction"])
    _, per_image, _, summary = assemble_feature_tables(patch_table, image_table, synthetic_batch_summary())
    assert summary["warnings"]
    assert "n_spacing_valid_patches" in per_image.columns


def test_output_schema_contains_required_columns() -> None:
    per_patch, per_image, per_donor, _ = assemble_feature_tables(synthetic_patch_table(), synthetic_image_table(), synthetic_batch_summary())
    assert list(per_patch.columns[: len(FEATURE_PATCH_COLUMNS)]) == FEATURE_PATCH_COLUMNS
    assert list(per_image.columns[: len(FEATURE_IMAGE_COLUMNS)]) == FEATURE_IMAGE_COLUMNS
    assert list(per_donor.columns[: len(FEATURE_DONOR_COLUMNS)]) == FEATURE_DONOR_COLUMNS


def test_summary_json_is_serializable(tmp_path: Path) -> None:
    per_patch, per_image, per_donor, summary = assemble_feature_tables(synthetic_patch_table(), synthetic_image_table(), synthetic_batch_summary())
    paths = write_feature_outputs(per_patch, per_image, per_donor, summary, tmp_path)
    loaded = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert loaded["per_image_rows"] == 3
    assert paths["summary_txt"].exists()


def test_input_tables_are_not_modified_in_place() -> None:
    patches = synthetic_patch_table()
    images = synthetic_image_table()
    batch = synthetic_batch_summary()
    patches_before = patches.copy(deep=True)
    images_before = images.copy(deep=True)
    batch_before = batch.copy(deep=True)

    assemble_feature_tables(patches, images, batch)

    pd.testing.assert_frame_equal(patches, patches_before)
    pd.testing.assert_frame_equal(images, images_before)
    pd.testing.assert_frame_equal(batch, batch_before)
