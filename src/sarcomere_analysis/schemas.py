from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


MANIFEST_COLUMNS = [
    "image_id",
    "donor_id",
    "region_id",
    "filename",
    "image_path",
    "pixel_size_um",
    "expected_spacing_px_min",
    "expected_spacing_px_max",
]

PATCH_METRICS_COLUMNS = [
    "image_id",
    "donor_id",
    "patch_id",
    "y0",
    "x0",
    "y1",
    "x1",
    "center_y",
    "center_x",
    "tissue_fraction",
    "intensity_mean",
    "intensity_std",
    "gradient_energy",
    "valid_for_orientation",
    "valid_for_periodicity",
    "valid_for_spacing",
    "invalid_reason",
    "patch_oop",
    "patch_mean_orientation_rad",
    "patch_mean_orientation_deg",
    "patch_orientation_weight_sum",
    "patch_orientation_valid_pixels",
    "patch_spacing_um",
    "patch_spacing_px",
    "patch_periodicity_score",
    "patch_spacing_confidence",
    "patch_spacing_method",
    "valid_for_spacing_final",
    "spacing_invalid_reason",
]

IMAGE_METRICS_COLUMNS = [
    "image_id",
    "donor_id",
    "tissue_fraction",
    "total_patches",
    "valid_orientation_patches",
    "image_oop",
    "image_mean_orientation_rad",
    "image_mean_orientation_deg",
    "image_oop_heterogeneity",
    "n_orientation_valid_patches",
    "image_spacing_mean_um",
    "image_spacing_median_um",
    "image_spacing_std_um",
    "image_spacing_cv",
    "n_spacing_valid_patches",
    "spacing_valid_fraction",
]

BATCH_RUN_SUMMARY_COLUMNS = [
    "image_id",
    "donor_id",
    "status",
    "error_message",
    "runtime_seconds",
    "per_patch_metrics_path",
    "per_image_metrics_path",
    "provenance_path",
]


def stabilize_columns(
    df: pd.DataFrame,
    columns: list[str],
    required_core: Iterable[str] | None = None,
    fill_value: object = np.nan,
) -> pd.DataFrame:
    result = df.copy()
    missing_core = [column for column in required_core or [] if column not in result.columns]
    if missing_core:
        raise ValueError(f"Missing required core columns: {missing_core}")
    for column in columns:
        if column not in result.columns:
            result[column] = fill_value
    extras = [column for column in result.columns if column not in columns]
    return result[columns + extras]
