from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import output_dir


FEATURE_PATCH_COLUMNS = [
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
    "rms_contrast",
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
    "valid_for_spacing_final",
    "spacing_invalid_reason",
    "spacing_endpoint_status",
    "spacing_feature_family",
]

FEATURE_IMAGE_COLUMNS = [
    "image_id",
    "donor_id",
    "status",
    "source_patch_rows",
    "tissue_fraction",
    "total_patches",
    "valid_orientation_patches",
    "n_orientation_valid_patches",
    "orientation_valid_fraction",
    "valid_periodicity_patches",
    "periodicity_valid_fraction",
    "image_oop",
    "image_mean_orientation_rad",
    "image_mean_orientation_deg",
    "image_oop_heterogeneity",
    "mean_patch_oop",
    "median_patch_oop",
    "sd_patch_oop",
    "iqr_patch_oop",
    "patch_oop_valid_count",
    "patch_oop_valid_fraction",
    "mean_patch_contrast",
    "median_patch_contrast",
    "mean_gradient_energy",
    "median_gradient_energy",
    "invalid_reason_counts",
    "invalid_reason_fractions",
    "n_spacing_valid_patches",
    "spacing_valid_fraction",
    "spacing_low_yield_flag",
    "spacing_endpoint_status",
    "image_spacing_mean_um",
    "image_spacing_median_um",
    "image_spacing_std_um",
    "image_spacing_cv",
]

FEATURE_DONOR_COLUMNS = [
    "donor_id",
    "n_images",
    "n_ok_images",
    "median_image_oop",
    "mean_image_oop",
    "sd_image_oop",
    "median_oop_heterogeneity",
    "mean_oop_heterogeneity",
    "median_orientation_valid_fraction",
    "mean_orientation_valid_fraction",
    "median_tissue_fraction",
    "mean_tissue_fraction",
    "total_patch_rows",
    "total_valid_orientation_patches",
    "total_valid_spacing_patches",
    "n_images_insufficient_spacing",
    "spacing_endpoint_status",
    "spacing_global_status",
]

REQUIRED_PATCH_COLUMNS = [
    "image_id",
    "donor_id",
    "patch_id",
    "valid_for_orientation",
    "valid_for_periodicity",
    "valid_for_spacing",
    "invalid_reason",
    "patch_oop",
]

REQUIRED_IMAGE_COLUMNS = [
    "image_id",
    "donor_id",
    "total_patches",
    "image_oop",
    "image_oop_heterogeneity",
    "n_orientation_valid_patches",
]

OPTIONAL_PATCH_COLUMNS = [
    "rms_contrast",
    "gradient_energy",
    "valid_for_spacing_final",
    "patch_spacing_um",
    "patch_spacing_px",
    "patch_periodicity_score",
    "patch_spacing_confidence",
    "spacing_invalid_reason",
]

OPTIONAL_IMAGE_COLUMNS = [
    "tissue_fraction",
    "valid_orientation_patches",
    "image_mean_orientation_rad",
    "image_mean_orientation_deg",
    "n_spacing_valid_patches",
    "spacing_valid_fraction",
    "image_spacing_mean_um",
    "image_spacing_median_um",
    "image_spacing_std_um",
    "image_spacing_cv",
]


def load_feature_inputs(
    cfg: dict[str, Any],
    patch_table: str | Path | None = None,
    image_table: str | Path | None = None,
    batch_summary: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = output_dir(cfg)
    patch_path = Path(patch_table) if patch_table else root / "tables" / "per_patch_metrics.csv"
    image_path = Path(image_table) if image_table else root / "tables" / "per_image_metrics.csv"
    batch_path = Path(batch_summary) if batch_summary else root / "tables" / "batch_run_summary.csv"
    patches = pd.read_csv(patch_path, dtype={"image_id": str, "donor_id": str, "patch_id": str})
    images = pd.read_csv(image_path, dtype={"image_id": str, "donor_id": str})
    batch = (
        pd.read_csv(batch_path, dtype={"image_id": str, "donor_id": str})
        if batch_path.exists()
        else pd.DataFrame(columns=["image_id", "donor_id", "status"])
    )
    return patches, images, batch


def assemble_feature_tables(
    patch_metrics: pd.DataFrame,
    image_metrics: pd.DataFrame,
    batch_summary: pd.DataFrame | None = None,
    min_spacing_patches_per_image: int = 5,
    min_spacing_patches_per_donor: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    patches = patch_metrics.copy(deep=True)
    images = image_metrics.copy(deep=True)
    batch = batch_summary.copy(deep=True) if batch_summary is not None else pd.DataFrame()
    warnings: list[str] = []

    require_columns(patches, REQUIRED_PATCH_COLUMNS, "patch metrics")
    require_columns(images, REQUIRED_IMAGE_COLUMNS, "image metrics")
    add_optional_columns(patches, OPTIONAL_PATCH_COLUMNS, warnings, "patch metrics")
    add_optional_columns(images, OPTIONAL_IMAGE_COLUMNS, warnings, "image metrics")

    feature_patches = assemble_patch_features(patches)
    feature_images = assemble_image_features(
        feature_patches,
        images,
        batch,
        min_spacing_patches_per_image=min_spacing_patches_per_image,
    )
    global_spacing_status = global_spacing_status_from_images(feature_images)
    feature_donors = assemble_donor_features(
        feature_images,
        min_spacing_patches_per_donor=min_spacing_patches_per_donor,
        global_spacing_status=global_spacing_status,
    )
    summary = feature_assembly_summary(
        feature_patches,
        feature_images,
        feature_donors,
        warnings,
        min_spacing_patches_per_image,
        min_spacing_patches_per_donor,
        global_spacing_status,
    )
    return feature_patches, feature_images, feature_donors, summary


def assemble_patch_features(patches: pd.DataFrame) -> pd.DataFrame:
    result = patches.copy(deep=True)
    valid_spacing = bool_column(result, "valid_for_spacing_final")
    result["spacing_endpoint_status"] = np.where(valid_spacing, "accepted_exploratory", "not_accepted_or_not_available")
    result["spacing_feature_family"] = "exploratory_low_yield"
    return stabilize_feature_columns(result, FEATURE_PATCH_COLUMNS)


def assemble_image_features(
    feature_patches: pd.DataFrame,
    image_metrics: pd.DataFrame,
    batch_summary: pd.DataFrame,
    min_spacing_patches_per_image: int,
) -> pd.DataFrame:
    images = image_metrics.copy(deep=True)
    if not batch_summary.empty and {"image_id", "status"}.issubset(batch_summary.columns):
        status = batch_summary[["image_id", "status"]].drop_duplicates("image_id")
        images = images.merge(status, on="image_id", how="left")
    elif "status" not in images.columns:
        images["status"] = np.nan

    grouped = feature_patches.groupby("image_id", dropna=False)
    patch_summary = grouped.apply(summarize_patches_for_image, include_groups=False).reset_index()
    result = images.merge(patch_summary, on="image_id", how="left", suffixes=("", "_from_patches"))
    result["source_patch_rows"] = numeric_column(result, "source_patch_rows").fillna(0).astype(int)

    result["orientation_valid_fraction"] = safe_divide(
        numeric_column(result, "n_orientation_valid_patches"),
        numeric_column(result, "total_patches"),
    )
    result["periodicity_valid_fraction"] = safe_divide(
        numeric_column(result, "valid_periodicity_patches"),
        numeric_column(result, "source_patch_rows"),
    )
    result["patch_oop_valid_fraction"] = safe_divide(
        numeric_column(result, "patch_oop_valid_count"),
        numeric_column(result, "source_patch_rows"),
    )
    n_spacing = numeric_column(result, "n_spacing_valid_patches").fillna(0).astype(int)
    result["spacing_low_yield_flag"] = n_spacing < int(min_spacing_patches_per_image)
    result["spacing_endpoint_status"] = np.where(
        result["spacing_low_yield_flag"],
        "insufficient_patch_yield",
        "exploratory_patch_yield_met",
    )
    return stabilize_feature_columns(result, FEATURE_IMAGE_COLUMNS)


def summarize_patches_for_image(group: pd.DataFrame) -> pd.Series:
    invalid_counts = value_counts_dict(group.get("invalid_reason", pd.Series(dtype=object)))
    invalid_fractions = fraction_counts_dict(group.get("invalid_reason", pd.Series(dtype=object)))
    patch_oop = numeric_column(group, "patch_oop")
    contrast = numeric_column(group, "rms_contrast")
    if contrast.isna().all():
        contrast = numeric_column(group, "intensity_std")
    gradient = numeric_column(group, "gradient_energy")
    return pd.Series(
        {
            "source_patch_rows": int(len(group)),
            "valid_periodicity_patches": int(bool_column(group, "valid_for_periodicity").sum()),
            "mean_patch_oop": finite_stat(patch_oop, np.mean),
            "median_patch_oop": finite_stat(patch_oop, np.median),
            "sd_patch_oop": finite_stat(patch_oop, np.std),
            "iqr_patch_oop": finite_iqr(patch_oop),
            "patch_oop_valid_count": int(patch_oop.notna().sum()),
            "mean_patch_contrast": finite_stat(contrast, np.mean),
            "median_patch_contrast": finite_stat(contrast, np.median),
            "mean_gradient_energy": finite_stat(gradient, np.mean),
            "median_gradient_energy": finite_stat(gradient, np.median),
            "invalid_reason_counts": json.dumps(invalid_counts, sort_keys=True),
            "invalid_reason_fractions": json.dumps(invalid_fractions, sort_keys=True),
        }
    )


def assemble_donor_features(
    feature_images: pd.DataFrame,
    min_spacing_patches_per_donor: int,
    global_spacing_status: str,
) -> pd.DataFrame:
    rows = []
    for donor_id, group in feature_images.groupby("donor_id", dropna=False):
        spacing_total = int(numeric_column(group, "n_spacing_valid_patches").fillna(0).sum())
        rows.append(
            {
                "donor_id": str(donor_id),
                "n_images": int(len(group)),
                "n_ok_images": int((group.get("status", pd.Series(index=group.index, dtype=object)).fillna("ok") == "ok").sum()),
                "median_image_oop": finite_stat(numeric_column(group, "image_oop"), np.median),
                "mean_image_oop": finite_stat(numeric_column(group, "image_oop"), np.mean),
                "sd_image_oop": finite_stat(numeric_column(group, "image_oop"), np.std),
                "median_oop_heterogeneity": finite_stat(numeric_column(group, "image_oop_heterogeneity"), np.median),
                "mean_oop_heterogeneity": finite_stat(numeric_column(group, "image_oop_heterogeneity"), np.mean),
                "median_orientation_valid_fraction": finite_stat(numeric_column(group, "orientation_valid_fraction"), np.median),
                "mean_orientation_valid_fraction": finite_stat(numeric_column(group, "orientation_valid_fraction"), np.mean),
                "median_tissue_fraction": finite_stat(numeric_column(group, "tissue_fraction"), np.median),
                "mean_tissue_fraction": finite_stat(numeric_column(group, "tissue_fraction"), np.mean),
                "total_patch_rows": int(numeric_column(group, "source_patch_rows").fillna(0).sum()),
                "total_valid_orientation_patches": int(numeric_column(group, "n_orientation_valid_patches").fillna(0).sum()),
                "total_valid_spacing_patches": spacing_total,
                "n_images_insufficient_spacing": int((group["spacing_endpoint_status"] == "insufficient_patch_yield").sum()),
                "spacing_endpoint_status": (
                    "insufficient_patch_yield"
                    if spacing_total < int(min_spacing_patches_per_donor)
                    else "exploratory_patch_yield_met"
                ),
                "spacing_global_status": global_spacing_status,
            }
        )
    return stabilize_feature_columns(pd.DataFrame(rows), FEATURE_DONOR_COLUMNS)


def write_feature_outputs(
    per_patch: pd.DataFrame,
    per_image: pd.DataFrame,
    per_donor: pd.DataFrame,
    summary: dict[str, Any],
    output_directory: str | Path,
) -> dict[str, Path]:
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_patch": out_dir / "features_per_patch.csv",
        "per_image": out_dir / "features_per_image.csv",
        "per_donor": out_dir / "features_per_donor.csv",
        "summary_json": out_dir / "feature_assembly_summary.json",
        "summary_txt": out_dir / "feature_assembly_summary.txt",
    }
    per_patch.to_csv(paths["per_patch"], index=False)
    per_image.to_csv(paths["per_image"], index=False)
    per_donor.to_csv(paths["per_donor"], index=False)
    with paths["summary_json"].open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2)
    paths["summary_txt"].write_text(summary_text(summary), encoding="utf-8")
    return paths


def feature_assembly_summary(
    per_patch: pd.DataFrame,
    per_image: pd.DataFrame,
    per_donor: pd.DataFrame,
    warnings: list[str],
    min_spacing_patches_per_image: int,
    min_spacing_patches_per_donor: int,
    spacing_global_status: str,
) -> dict[str, Any]:
    insufficient_images = int((per_image["spacing_endpoint_status"] == "insufficient_patch_yield").sum()) if not per_image.empty else 0
    insufficient_donors = int((per_donor["spacing_endpoint_status"] == "insufficient_patch_yield").sum()) if not per_donor.empty else 0
    return json_safe(
        {
            "per_patch_rows": int(len(per_patch)),
            "per_image_rows": int(len(per_image)),
            "per_donor_rows": int(len(per_donor)),
            "donor_count": int(per_donor["donor_id"].nunique()) if "donor_id" in per_donor.columns else 0,
            "primary_feature_family": "orientation_oop",
            "spacing_global_status": spacing_global_status,
            "min_spacing_patches_per_image": int(min_spacing_patches_per_image),
            "min_spacing_patches_per_donor": int(min_spacing_patches_per_donor),
            "images_with_insufficient_spacing_yield": insufficient_images,
            "donors_with_insufficient_spacing_yield": insufficient_donors,
            "total_valid_spacing_patches": int(numeric_column(per_image, "n_spacing_valid_patches").fillna(0).sum()) if not per_image.empty else 0,
            "warnings": warnings,
            "caution": (
                "Feature assembly performs no statistics, clinical modeling, FIJI validation, or biological inference. "
                "Spacing fields are preserved as exploratory low-yield descriptors and should not gate OOP/orientation features."
            ),
        }
    )


def global_spacing_status_from_images(per_image: pd.DataFrame) -> str:
    total_valid = int(numeric_column(per_image, "n_spacing_valid_patches").fillna(0).sum()) if not per_image.empty else 0
    total_patches = int(numeric_column(per_image, "total_patches").fillna(0).sum()) if not per_image.empty else 0
    if total_valid == 0:
        return "exploratory_no_valid_spacing"
    if total_patches > 0 and total_valid / total_patches < 0.01:
        return "exploratory_low_yield"
    return "exploratory"


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required {label} columns: {missing}")


def add_optional_columns(df: pd.DataFrame, columns: list[str], warnings: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        warnings.append(f"Missing optional {label} columns filled with NaN: {missing}")
    for column in missing:
        df[column] = np.nan


def stabilize_feature_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = np.nan
    extras = [column for column in result.columns if column not in columns]
    return result[columns + extras]


def bool_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    values = df[column]
    if values.dtype == object:
        return values.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
    return values.fillna(False).astype(bool)


def numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return num.divide(den.where(den != 0)).replace([np.inf, -np.inf], np.nan)


def finite_stat(values: pd.Series, func: Any) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if finite.size == 0:
        return float("nan")
    return float(func(finite))


def finite_iqr(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if finite.size == 0:
        return float("nan")
    q75, q25 = np.percentile(finite, [75, 25])
    return float(q75 - q25)


def value_counts_dict(values: pd.Series) -> dict[str, int]:
    counts = values.fillna("missing").astype(str).value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def fraction_counts_dict(values: pd.Series) -> dict[str, float]:
    counts = value_counts_dict(values)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {key: float(value / total) for key, value in counts.items()}


def summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Feature Assembly Summary",
        f"per_patch_rows: {summary.get('per_patch_rows')}",
        f"per_image_rows: {summary.get('per_image_rows')}",
        f"per_donor_rows: {summary.get('per_donor_rows')}",
        f"donor_count: {summary.get('donor_count')}",
        f"primary_feature_family: {summary.get('primary_feature_family')}",
        f"spacing_global_status: {summary.get('spacing_global_status')}",
        f"images_with_insufficient_spacing_yield: {summary.get('images_with_insufficient_spacing_yield')}",
        f"donors_with_insufficient_spacing_yield: {summary.get('donors_with_insufficient_spacing_yield')}",
        f"warnings: {summary.get('warnings', [])}",
        str(summary.get("caution", "")),
    ]
    return "\n".join(lines) + "\n"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
