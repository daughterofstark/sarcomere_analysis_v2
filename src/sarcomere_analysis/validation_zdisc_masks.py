from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import output_dir
from .zdisc_annotation import json_safe


REQUIRED_ANNOTATION_COLUMNS = [
    "annotation_id",
    "image_id",
    "donor_id",
    "patch_id",
    "annotation_status",
    "zdisc_pixel_fraction",
    "has_zdisc_labels",
    "manual_mask_orientation_deg",
    "orientation_estimable",
]

REQUIRED_PATCH_COLUMNS = [
    "image_id",
    "donor_id",
    "patch_id",
    "patch_oop",
    "patch_mean_orientation_deg",
]

AUTOMATED_PATCH_COLUMNS = [
    "image_id",
    "patch_id",
    "donor_id",
    "patch_oop",
    "patch_mean_orientation_deg",
    "valid_for_orientation",
    "invalid_reason",
]


def default_validation_paths(cfg: dict[str, Any], output_directory: str | Path | None = None) -> dict[str, Path]:
    out_dir = Path(output_directory) if output_directory else output_dir(cfg) / "validation"
    return {
        "matched_csv": out_dir / "zdisc_mask_validation_matched.csv",
        "summary_json": out_dir / "zdisc_mask_validation_summary.json",
        "summary_txt": out_dir / "zdisc_mask_validation_summary.txt",
    }


def load_validation_inputs(
    cfg: dict[str, Any],
    annotation_features: str | Path | None = None,
    patch_features: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = output_dir(cfg)
    annotation_path = Path(annotation_features) if annotation_features else root / "zdisc_annotation" / "zdisc_annotation_features.csv"
    patch_path = Path(patch_features) if patch_features else root / "tables" / "features_per_patch.csv"
    annotations = pd.read_csv(annotation_path, dtype={"annotation_id": str, "image_id": str, "donor_id": str, "patch_id": str})
    patches = pd.read_csv(patch_path, dtype={"image_id": str, "donor_id": str, "patch_id": str})
    return annotations, patches


def validate_zdisc_masks(
    cfg: dict[str, Any],
    annotation_features: str | Path | None = None,
    patch_features: str | Path | None = None,
    output_directory: str | Path | None = None,
    min_n_for_correlation: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    annotations, patches = load_validation_inputs(cfg, annotation_features, patch_features)
    matched, summary = build_zdisc_mask_validation(annotations, patches, min_n_for_correlation=min_n_for_correlation)
    paths = default_validation_paths(cfg, output_directory)
    write_validation_outputs(matched, summary, paths)
    return matched, summary, paths


def build_zdisc_mask_validation(
    annotation_features: pd.DataFrame,
    patch_features: pd.DataFrame,
    min_n_for_correlation: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    annotations = annotation_features.copy(deep=True)
    patches = patch_features.copy(deep=True)
    require_columns(annotations, REQUIRED_ANNOTATION_COLUMNS, "annotation feature table")
    require_columns(patches, REQUIRED_PATCH_COLUMNS, "patch feature table")
    standardize_ids(annotations, ["annotation_id", "image_id", "donor_id", "patch_id"])
    standardize_ids(patches, ["image_id", "donor_id", "patch_id"])

    patch_subset_columns = [column for column in AUTOMATED_PATCH_COLUMNS if column in patches.columns]
    patch_subset = patches[patch_subset_columns].copy()
    duplicate_keys = patch_subset.duplicated(["image_id", "patch_id"], keep=False)
    if duplicate_keys.any():
        duplicated = patch_subset.loc[duplicate_keys, ["image_id", "patch_id"]].drop_duplicates().to_dict("records")
        raise ValueError(f"Patch feature table contains duplicate image_id/patch_id keys: {duplicated[:10]}")
    patch_subset = patch_subset.rename(
        columns={
            "donor_id": "automated_donor_id",
            "patch_oop": "automated_patch_oop",
            "patch_mean_orientation_deg": "automated_patch_mean_orientation_deg",
            "valid_for_orientation": "automated_valid_for_orientation",
            "invalid_reason": "automated_invalid_reason",
        }
    )
    matched = annotations.merge(patch_subset, on=["image_id", "patch_id"], how="left", indicator="_patch_join_status")
    joined = matched["_patch_join_status"] == "both"
    donor_mismatch = joined & (matched["donor_id"].astype(str) != matched["automated_donor_id"].astype(str))
    matched["validation_match_status"] = np.select(
        [~joined, donor_mismatch],
        ["unmatched_patch", "donor_id_mismatch"],
        default="matched",
    )
    matched["manual_mask_orientation_deg"] = pd.to_numeric(matched["manual_mask_orientation_deg"], errors="coerce")
    matched["automated_patch_mean_orientation_deg"] = pd.to_numeric(matched["automated_patch_mean_orientation_deg"], errors="coerce")
    matched["automated_patch_oop"] = pd.to_numeric(matched["automated_patch_oop"], errors="coerce")
    matched["zdisc_pixel_fraction"] = pd.to_numeric(matched["zdisc_pixel_fraction"], errors="coerce")
    matched["orientation_estimable_bool"] = bool_column(matched, "orientation_estimable")
    orientation_pair_mask = (
        (matched["validation_match_status"] == "matched")
        & matched["orientation_estimable_bool"]
        & matched["manual_mask_orientation_deg"].notna()
        & matched["automated_patch_mean_orientation_deg"].notna()
    )
    errors = axial_angular_error_series(
        matched.loc[orientation_pair_mask, "manual_mask_orientation_deg"],
        matched.loc[orientation_pair_mask, "automated_patch_mean_orientation_deg"],
    )
    matched["axial_orientation_error_deg"] = np.nan
    matched.loc[orientation_pair_mask, "axial_orientation_error_deg"] = errors.to_numpy()
    summary = build_validation_summary(matched, min_n_for_correlation=min_n_for_correlation)
    return matched, summary


def axial_angular_error_deg(angle_a: float, angle_b: float) -> float:
    a = float(angle_a) % 180.0
    b = float(angle_b) % 180.0
    diff = abs(a - b)
    return float(min(diff, 180.0 - diff))


def axial_angular_error_series(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.Series([axial_angular_error_deg(x, y) for x, y in zip(a, b)], index=a.index, dtype=float)


def build_validation_summary(matched: pd.DataFrame, min_n_for_correlation: int = 10) -> dict[str, Any]:
    orientation_errors = pd.to_numeric(matched["axial_orientation_error_deg"], errors="coerce").dropna()
    oop_medians = group_oop_medians(matched)
    spearman = spearman_summary(matched, min_n_for_correlation=min_n_for_correlation)
    status_counts = matched["annotation_status"].fillna("missing").astype(str).value_counts().to_dict()
    return json_safe(
        {
            "mode": "pilot_zdisc_mask_validation",
            "total_annotation_masks": int(len(matched)),
            "matched_rows_to_automated_patch_features": int((matched["validation_match_status"] == "matched").sum()),
            "unmatched_rows": int((matched["validation_match_status"] == "unmatched_patch").sum()),
            "donor_id_mismatches": int((matched["validation_match_status"] == "donor_id_mismatch").sum()),
            "masks_with_zdisc_labels": int(bool_column(matched, "has_zdisc_labels").sum()) if len(matched) else 0,
            "masks_orientation_estimable": int(bool_column(matched, "orientation_estimable").sum()) if len(matched) else 0,
            "empty_masks": int((matched["annotation_status"] == "empty").sum()) if len(matched) else 0,
            "ignore_only_masks": int((matched["annotation_status"] == "ignore_only").sum()) if len(matched) else 0,
            "annotation_status_counts": {str(key): int(value) for key, value in status_counts.items()},
            "n_orientation_pairs": int(len(orientation_errors)),
            "median_axial_error_deg": float(np.nanmedian(orientation_errors)) if len(orientation_errors) else np.nan,
            "mean_axial_error_deg": float(np.nanmean(orientation_errors)) if len(orientation_errors) else np.nan,
            "iqr_axial_error_deg": iqr(orientation_errors),
            "oop_medians_by_annotation_status": oop_medians,
            "spearman_zdisc_fraction_vs_patch_oop": spearman,
            "interpretation_flags": [
                "pilot_validation_only",
                "manual_masks_drawn_by_user_not_independent_blinded_expert",
                "small_orientation_estimable_n",
                "not_final_publication_validation",
                "does_not_validate_spacing",
                "spacing_remains_exploratory_low_yield",
                "no_p_values_or_clinical_statistics_computed",
            ],
        }
    )


def group_oop_medians(matched: pd.DataFrame) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for status in ["zdisc_labeled", "empty", "ignore_only", "mixed"]:
        values = pd.to_numeric(matched.loc[matched["annotation_status"] == status, "automated_patch_oop"], errors="coerce").dropna()
        result[status] = None if values.empty else float(np.nanmedian(values))
    return result


def spearman_summary(matched: pd.DataFrame, min_n_for_correlation: int = 10) -> dict[str, Any]:
    subset = matched.loc[matched["validation_match_status"] == "matched", ["zdisc_pixel_fraction", "automated_patch_oop"]].copy()
    subset["zdisc_pixel_fraction"] = pd.to_numeric(subset["zdisc_pixel_fraction"], errors="coerce")
    subset["automated_patch_oop"] = pd.to_numeric(subset["automated_patch_oop"], errors="coerce")
    subset = subset.dropna()
    n = int(len(subset))
    if n < int(min_n_for_correlation):
        return {"computed": False, "reason": "too_few_rows", "n": n, "rho": None, "p_value": None}
    if subset["zdisc_pixel_fraction"].nunique() < 2 or subset["automated_patch_oop"].nunique() < 2:
        return {"computed": False, "reason": "constant_input", "n": n, "rho": None, "p_value": None}
    rho, p_value = spearmanr(subset["zdisc_pixel_fraction"], subset["automated_patch_oop"])
    return {
        "computed": bool(np.isfinite(rho)),
        "reason": "exploratory_pilot_only" if np.isfinite(rho) else "not_finite",
        "n": n,
        "rho": float(rho) if np.isfinite(rho) else None,
        "p_value": float(p_value) if np.isfinite(p_value) else None,
        "caution": "Exploratory pilot association only; no validation claim or threshold selection.",
    }


def iqr(values: pd.Series) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25))


def write_validation_outputs(matched: pd.DataFrame, summary: dict[str, Any], paths: dict[str, Path]) -> None:
    paths["matched_csv"].parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(paths["matched_csv"], index=False)
    paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    lines = [f"{key}: {value}" for key, value in summary.items()]
    paths["summary_txt"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required {label} columns: {missing}")


def standardize_ids(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str).str.strip()


def bool_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    values = df[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_string_dtype(values) or values.dtype == object:
        return values.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
    return values.fillna(False).astype(bool)
