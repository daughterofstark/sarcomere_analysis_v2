from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import output_dir


ANALYSIS_IMAGE_COLUMNS = [
    "image_id",
    "donor_id",
    "is_healthy",
    "feature_source",
    "metadata_source",
    "status",
    "image_oop",
    "image_mean_orientation_rad",
    "image_mean_orientation_deg",
    "image_oop_heterogeneity",
    "n_orientation_valid_patches",
    "valid_orientation_patches",
    "orientation_valid_fraction",
    "n_spacing_valid_patches",
    "spacing_valid_fraction",
    "spacing_low_yield_flag",
    "spacing_endpoint_status",
    "tissue_fraction",
    "total_patches",
    "source_patch_rows",
    "valid_periodicity_patches",
    "periodicity_valid_fraction",
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
    "image_path",
]

ANALYSIS_DONOR_COLUMNS = [
    "donor_id",
    "is_healthy",
    "feature_source",
    "metadata_source",
    "n_images",
    "n_ok_images",
    "median_image_oop",
    "mean_image_oop",
    "sd_image_oop",
    "median_oop_heterogeneity",
    "mean_oop_heterogeneity",
    "median_orientation_valid_fraction",
    "mean_orientation_valid_fraction",
    "total_valid_orientation_patches",
    "total_patch_rows",
    "total_valid_spacing_patches",
    "n_images_insufficient_spacing",
    "spacing_endpoint_status",
    "spacing_global_status",
]

REQUIRED_IMAGE_FEATURE_COLUMNS = [
    "image_id",
    "donor_id",
    "image_oop",
    "image_oop_heterogeneity",
    "n_orientation_valid_patches",
    "orientation_valid_fraction",
    "spacing_endpoint_status",
]

REQUIRED_DONOR_FEATURE_COLUMNS = [
    "donor_id",
    "n_images",
    "median_image_oop",
    "mean_image_oop",
    "median_orientation_valid_fraction",
    "total_valid_orientation_patches",
    "spacing_endpoint_status",
]

REQUIRED_ENRICHED_MANIFEST_COLUMNS = ["image_id", "donor_id", "is_healthy"]
REQUIRED_DONOR_METADATA_COLUMNS = ["donor_id", "is_healthy", "n_images"]


def load_analysis_inputs(
    cfg: dict[str, Any],
    features_image: str | Path | None = None,
    features_donor: str | Path | None = None,
    enriched_manifest: str | Path | None = None,
    donor_metadata: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = output_dir(cfg) / "tables"
    image_features_path = Path(features_image) if features_image else root / "features_per_image.csv"
    donor_features_path = Path(features_donor) if features_donor else root / "features_per_donor.csv"
    enriched_path = Path(enriched_manifest) if enriched_manifest else root / "enriched_manifest.csv"
    donor_metadata_path = Path(donor_metadata) if donor_metadata else root / "donor_metadata.csv"
    return (
        pd.read_csv(image_features_path, dtype={"image_id": str, "donor_id": str}),
        pd.read_csv(donor_features_path, dtype={"donor_id": str}),
        pd.read_csv(enriched_path, dtype={"image_id": str, "donor_id": str, "region_id": str}),
        pd.read_csv(donor_metadata_path, dtype={"donor_id": str}),
    )


def build_analysis_tables(
    image_features: pd.DataFrame,
    donor_features: pd.DataFrame,
    enriched_manifest: pd.DataFrame,
    donor_metadata: pd.DataFrame,
    strict: bool = False,
    expected_healthy_donor_count: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    images = image_features.copy(deep=True)
    donors = donor_features.copy(deep=True)
    manifest = enriched_manifest.copy(deep=True)
    donor_meta = donor_metadata.copy(deep=True)

    require_columns(images, REQUIRED_IMAGE_FEATURE_COLUMNS, "image feature")
    require_columns(donors, REQUIRED_DONOR_FEATURE_COLUMNS, "donor feature")
    require_columns(manifest, REQUIRED_ENRICHED_MANIFEST_COLUMNS, "enriched manifest")
    require_columns(donor_meta, REQUIRED_DONOR_METADATA_COLUMNS, "donor metadata")
    standardize_ids(images, image=True)
    standardize_ids(donors, image=False)
    standardize_ids(manifest, image=True)
    standardize_ids(donor_meta, image=False)
    validate_unique(images, "image_id", "image features")
    validate_unique(manifest, "image_id", "enriched manifest")
    validate_unique(donors, "donor_id", "donor features")
    validate_unique(donor_meta, "donor_id", "donor metadata")

    analysis_image, missing_image_metadata = join_image_tables(images, manifest, strict)
    analysis_donor, missing_donor_metadata = join_donor_tables(donors, donor_meta, strict)
    summary = analysis_table_summary(
        analysis_image,
        analysis_donor,
        missing_image_metadata,
        missing_donor_metadata,
        expected_healthy_donor_count,
    )
    return analysis_image, analysis_donor, summary


def join_image_tables(
    image_features: pd.DataFrame,
    enriched_manifest: pd.DataFrame,
    strict: bool,
) -> tuple[pd.DataFrame, list[str]]:
    metadata = rename_conflicting_metadata_columns(
        enriched_manifest,
        protected=set(image_features.columns) - {"image_id", "donor_id"},
    )
    join_keys = ["image_id", "donor_id"] if "donor_id" in image_features.columns and "donor_id" in metadata.columns else ["image_id"]
    joined = image_features.merge(metadata, on=join_keys, how="left", indicator="_metadata_join_status")
    if len(joined) != len(image_features):
        raise ValueError(f"Image analysis join changed row count from {len(image_features)} to {len(joined)}")
    if joined["image_id"].duplicated().any():
        raise ValueError("Image analysis table contains duplicate image_id values after join.")
    missing = sorted(joined.loc[joined["_metadata_join_status"] == "left_only", "image_id"].astype(str).unique())
    if strict and missing:
        raise ValueError(f"Missing image metadata rows for image_id values: {missing}")
    joined["feature_source"] = "features_per_image"
    joined["metadata_source"] = np.where(joined["_metadata_join_status"] == "both", "enriched_manifest", "")
    joined = joined.drop(columns=["_metadata_join_status"])
    return stabilize_analysis_columns(joined, ANALYSIS_IMAGE_COLUMNS), missing


def join_donor_tables(
    donor_features: pd.DataFrame,
    donor_metadata: pd.DataFrame,
    strict: bool,
) -> tuple[pd.DataFrame, list[str]]:
    metadata = rename_conflicting_metadata_columns(
        donor_metadata,
        protected=set(donor_features.columns) - {"donor_id"},
    )
    joined = donor_features.merge(metadata, on="donor_id", how="left", indicator="_metadata_join_status")
    if len(joined) != len(donor_features):
        raise ValueError(f"Donor analysis join changed row count from {len(donor_features)} to {len(joined)}")
    if joined["donor_id"].duplicated().any():
        raise ValueError("Donor analysis table contains duplicate donor_id values after join.")
    missing = sorted(joined.loc[joined["_metadata_join_status"] == "left_only", "donor_id"].astype(str).unique())
    if strict and missing:
        raise ValueError(f"Missing donor metadata rows for donor_id values: {missing}")
    joined["feature_source"] = "features_per_donor"
    joined["metadata_source"] = np.where(joined["_metadata_join_status"] == "both", "donor_metadata", "")
    joined = joined.drop(columns=["_metadata_join_status"])
    return stabilize_analysis_columns(joined, ANALYSIS_DONOR_COLUMNS), missing


def rename_conflicting_metadata_columns(metadata: pd.DataFrame, protected: set[str]) -> pd.DataFrame:
    result = metadata.copy(deep=True)
    rename_map = {
        column: f"{column}_metadata"
        for column in result.columns
        if column not in {"image_id", "donor_id", "is_healthy"} and column in protected
    }
    return result.rename(columns=rename_map)


def write_analysis_outputs(
    per_image: pd.DataFrame,
    per_donor: pd.DataFrame,
    summary: dict[str, Any],
    output_directory: str | Path,
) -> dict[str, Path]:
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_image": out_dir / "analysis_per_image.csv",
        "per_donor": out_dir / "analysis_per_donor.csv",
        "summary_json": out_dir / "analysis_table_summary.json",
        "summary_txt": out_dir / "analysis_table_summary.txt",
    }
    per_image.to_csv(paths["per_image"], index=False)
    per_donor.to_csv(paths["per_donor"], index=False)
    with paths["summary_json"].open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2)
    paths["summary_txt"].write_text(analysis_summary_text(summary), encoding="utf-8")
    return paths


def analysis_table_summary(
    per_image: pd.DataFrame,
    per_donor: pd.DataFrame,
    missing_image_metadata: list[str],
    missing_donor_metadata: list[str],
    expected_healthy_donor_count: int | None,
) -> dict[str, Any]:
    healthy_donors = sorted(per_donor.loc[bool_column(per_donor, "is_healthy"), "donor_id"].astype(str).unique())
    spacing_global_status = ""
    if "spacing_global_status" in per_donor.columns:
        values = per_donor["spacing_global_status"].dropna().astype(str).unique()
        if len(values):
            spacing_global_status = values[0] if len(values) == 1 else ";".join(sorted(values))
    expected_status_note = ""
    if expected_healthy_donor_count is not None and len(healthy_donors) != int(expected_healthy_donor_count):
        expected_status_note = (
            f"Healthy donor count {len(healthy_donors)} differs from configured count "
            f"{int(expected_healthy_donor_count)}."
        )
    return json_safe(
        {
            "image_rows": int(len(per_image)),
            "donor_rows": int(len(per_donor)),
            "unique_donors": int(per_donor["donor_id"].nunique()) if "donor_id" in per_donor.columns else 0,
            "healthy_donors": healthy_donors,
            "healthy_donor_count": int(len(healthy_donors)),
            "healthy_image_rows": int(bool_column(per_image, "is_healthy").sum()) if "is_healthy" in per_image.columns else 0,
            "missing_image_metadata_count": int(len(missing_image_metadata)),
            "missing_image_metadata_ids": missing_image_metadata,
            "missing_donor_metadata_count": int(len(missing_donor_metadata)),
            "missing_donor_metadata_ids": missing_donor_metadata,
            "spacing_global_status": spacing_global_status,
            "spacing_is_exploratory_low_yield": spacing_global_status == "exploratory_low_yield",
            "expected_healthy_donor_count": expected_healthy_donor_count,
            "expected_healthy_donor_count_note": expected_status_note,
            "caution": (
                "Analysis tables are joined inputs only. No p-values, correlations, group comparisons, "
                "model fits, plots, validation, or biological interpretation were computed."
            ),
        }
    )


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required {label} columns: {missing}")


def validate_unique(df: pd.DataFrame, column: str, label: str) -> None:
    duplicates = sorted(df.loc[df[column].duplicated(), column].astype(str).unique())
    if duplicates:
        raise ValueError(f"{label} contains duplicate {column} values: {duplicates}")


def standardize_ids(df: pd.DataFrame, image: bool) -> None:
    if "donor_id" in df.columns:
        df["donor_id"] = df["donor_id"].map(standardize_id)
        if pd.api.types.is_numeric_dtype(df["donor_id"]):
            raise ValueError("donor_id must be treated as a string identifier, not numeric.")
    if image and "image_id" in df.columns:
        df["image_id"] = df["image_id"].map(standardize_id)


def standardize_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text.replace(".", "", 1).isdigit():
        text = text[:-2]
    return text


def stabilize_analysis_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy(deep=True)
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


def analysis_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Analysis Table Summary",
        f"image_rows: {summary.get('image_rows')}",
        f"donor_rows: {summary.get('donor_rows')}",
        f"unique_donors: {summary.get('unique_donors')}",
        f"healthy_donor_count: {summary.get('healthy_donor_count')}",
        f"healthy_image_rows: {summary.get('healthy_image_rows')}",
        f"missing_image_metadata_count: {summary.get('missing_image_metadata_count')}",
        f"missing_donor_metadata_count: {summary.get('missing_donor_metadata_count')}",
        f"spacing_global_status: {summary.get('spacing_global_status')}",
        f"spacing_is_exploratory_low_yield: {summary.get('spacing_is_exploratory_low_yield')}",
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
