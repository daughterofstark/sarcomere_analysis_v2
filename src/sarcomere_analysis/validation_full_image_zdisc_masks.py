from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import output_dir
from .validation_zdisc_masks import axial_angular_error_series, bool_column, iqr, require_columns, standardize_ids
from .zdisc_annotation import json_safe


REQUIRED_FULL_IMAGE_ANNOTATION_COLUMNS = [
    "annotation_id",
    "image_id",
    "donor_id",
    "annotation_status",
    "zdisc_pixel_fraction",
    "has_zdisc_labels",
    "manual_mask_orientation_deg",
    "orientation_estimable",
]

REQUIRED_IMAGE_FEATURE_COLUMNS = [
    "image_id",
    "donor_id",
    "image_oop",
    "image_mean_orientation_deg",
]

AUTOMATED_IMAGE_COLUMNS = [
    "image_id",
    "donor_id",
    "image_oop",
    "image_mean_orientation_deg",
    "image_oop_heterogeneity",
    "n_orientation_valid_patches",
    "orientation_valid_fraction",
    "status",
    "spacing_endpoint_status",
    "spacing_valid_fraction",
]


def default_full_image_validation_paths(cfg: dict[str, Any], output_directory: str | Path | None = None) -> dict[str, Path]:
    out_dir = Path(output_directory) if output_directory else output_dir(cfg) / "validation"
    return {
        "matched_csv": out_dir / "full_image_zdisc_mask_validation_matched.csv",
        "summary_json": out_dir / "full_image_zdisc_mask_validation_summary.json",
        "summary_txt": out_dir / "full_image_zdisc_mask_validation_summary.txt",
    }


def load_full_image_validation_inputs(
    cfg: dict[str, Any],
    annotation_features: str | Path | None = None,
    image_features: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = output_dir(cfg)
    annotation_path = (
        Path(annotation_features)
        if annotation_features
        else root / "full_image_zdisc_annotation" / "full_image_zdisc_annotation_features.csv"
    )
    image_path = Path(image_features) if image_features else root / "tables" / "features_per_image.csv"
    annotations = pd.read_csv(annotation_path, dtype={"annotation_id": str, "image_id": str, "donor_id": str})
    images = pd.read_csv(image_path, dtype={"image_id": str, "donor_id": str})
    return annotations, images


def validate_full_image_zdisc_masks(
    cfg: dict[str, Any],
    annotation_features: str | Path | None = None,
    image_features: str | Path | None = None,
    output_directory: str | Path | None = None,
    min_n_for_correlation: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    annotations, images = load_full_image_validation_inputs(cfg, annotation_features, image_features)
    matched, summary = build_full_image_zdisc_mask_validation(
        annotations,
        images,
        min_n_for_correlation=min_n_for_correlation,
    )
    paths = default_full_image_validation_paths(cfg, output_directory)
    write_full_image_validation_outputs(matched, summary, paths)
    return matched, summary, paths


def build_full_image_zdisc_mask_validation(
    annotation_features: pd.DataFrame,
    image_features: pd.DataFrame,
    min_n_for_correlation: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    annotations = annotation_features.copy(deep=True)
    images = image_features.copy(deep=True)
    require_columns(annotations, REQUIRED_FULL_IMAGE_ANNOTATION_COLUMNS, "full-image annotation feature table")
    require_columns(images, REQUIRED_IMAGE_FEATURE_COLUMNS, "image feature table")
    standardize_ids(annotations, ["annotation_id", "image_id", "donor_id"])
    standardize_ids(images, ["image_id", "donor_id"])

    image_subset_columns = [column for column in AUTOMATED_IMAGE_COLUMNS if column in images.columns]
    image_subset = images[image_subset_columns].copy()
    duplicate_keys = image_subset.duplicated(["image_id"], keep=False)
    if duplicate_keys.any():
        duplicated = image_subset.loc[duplicate_keys, ["image_id"]].drop_duplicates().to_dict("records")
        raise ValueError(f"Image feature table contains duplicate image_id keys: {duplicated[:10]}")
    image_subset = image_subset.rename(
        columns={
            "donor_id": "automated_donor_id",
            "image_oop": "automated_image_oop",
            "image_mean_orientation_deg": "automated_image_mean_orientation_deg",
            "image_oop_heterogeneity": "automated_image_oop_heterogeneity",
            "n_orientation_valid_patches": "automated_n_orientation_valid_patches",
            "orientation_valid_fraction": "automated_orientation_valid_fraction",
            "status": "automated_status",
            "spacing_endpoint_status": "automated_spacing_endpoint_status",
            "spacing_valid_fraction": "automated_spacing_valid_fraction",
        }
    )
    matched = annotations.merge(image_subset, on="image_id", how="left", indicator="_image_join_status")
    joined = matched["_image_join_status"] == "both"
    donor_mismatch = joined & (matched["donor_id"].astype(str) != matched["automated_donor_id"].astype(str))
    matched["validation_match_status"] = np.select(
        [~joined, donor_mismatch],
        ["unmatched_image", "donor_id_mismatch"],
        default="matched",
    )
    matched["manual_mask_orientation_deg"] = pd.to_numeric(matched["manual_mask_orientation_deg"], errors="coerce")
    matched["automated_image_mean_orientation_deg"] = pd.to_numeric(matched["automated_image_mean_orientation_deg"], errors="coerce")
    matched["automated_image_oop"] = pd.to_numeric(matched["automated_image_oop"], errors="coerce")
    matched["zdisc_pixel_fraction"] = pd.to_numeric(matched["zdisc_pixel_fraction"], errors="coerce")
    matched["orientation_estimable_bool"] = bool_column(matched, "orientation_estimable")
    orientation_pair_mask = (
        (matched["validation_match_status"] == "matched")
        & matched["orientation_estimable_bool"]
        & matched["manual_mask_orientation_deg"].notna()
        & matched["automated_image_mean_orientation_deg"].notna()
    )
    errors = axial_angular_error_series(
        matched.loc[orientation_pair_mask, "manual_mask_orientation_deg"],
        matched.loc[orientation_pair_mask, "automated_image_mean_orientation_deg"],
    )
    matched["axial_orientation_error_deg"] = np.nan
    matched.loc[orientation_pair_mask, "axial_orientation_error_deg"] = errors.to_numpy()
    summary = build_full_image_validation_summary(matched, min_n_for_correlation=min_n_for_correlation)
    return matched, summary


def build_full_image_validation_summary(matched: pd.DataFrame, min_n_for_correlation: int = 10) -> dict[str, Any]:
    orientation_errors = pd.to_numeric(matched["axial_orientation_error_deg"], errors="coerce").dropna()
    status_counts = matched["annotation_status"].fillna("missing").astype(str).value_counts().to_dict()
    return json_safe(
        {
            "mode": "pilot_full_image_zdisc_mask_validation",
            "total_full_image_annotations": int(len(matched)),
            "matched_rows_to_automated_image_features": int((matched["validation_match_status"] == "matched").sum()),
            "unmatched_rows": int((matched["validation_match_status"] == "unmatched_image").sum()),
            "donor_id_mismatches": int((matched["validation_match_status"] == "donor_id_mismatch").sum()),
            "images_with_zdisc_labels": int(bool_column(matched, "has_zdisc_labels").sum()) if len(matched) else 0,
            "orientation_estimable_masks": int(bool_column(matched, "orientation_estimable").sum()) if len(matched) else 0,
            "empty_masks": int((matched["annotation_status"] == "empty").sum()) if len(matched) else 0,
            "annotation_status_counts": {str(key): int(value) for key, value in status_counts.items()},
            "n_orientation_pairs": int(len(orientation_errors)),
            "median_axial_error_deg": float(np.nanmedian(orientation_errors)) if len(orientation_errors) else np.nan,
            "mean_axial_error_deg": float(np.nanmean(orientation_errors)) if len(orientation_errors) else np.nan,
            "iqr_axial_error_deg": iqr(orientation_errors),
            "oop_medians_by_annotation_status": group_image_oop_medians(matched),
            "spearman_zdisc_fraction_vs_image_oop": spearman_summary(matched, min_n_for_correlation=min_n_for_correlation),
            "interpretation_flags": [
                "pilot_validation_only",
                "manual_masks_drawn_by_user_not_independent_blinded_expert",
                "small_orientation_estimable_n",
                "full_image_masks_are_sparse_annotations_not_exhaustive_segmentation",
                "not_final_publication_validation",
                "does_not_validate_spacing",
                "spacing_remains_exploratory_low_yield",
                "no_p_values_or_clinical_statistics_computed",
            ],
        }
    )


def group_image_oop_medians(matched: pd.DataFrame) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for status in ["zdisc_labeled", "empty", "ignore_only", "mixed"]:
        values = pd.to_numeric(matched.loc[matched["annotation_status"] == status, "automated_image_oop"], errors="coerce").dropna()
        result[status] = None if values.empty else float(np.nanmedian(values))
    return result


def spearman_summary(matched: pd.DataFrame, min_n_for_correlation: int = 10) -> dict[str, Any]:
    subset = matched.loc[matched["validation_match_status"] == "matched", ["zdisc_pixel_fraction", "automated_image_oop"]].copy()
    subset["zdisc_pixel_fraction"] = pd.to_numeric(subset["zdisc_pixel_fraction"], errors="coerce")
    subset["automated_image_oop"] = pd.to_numeric(subset["automated_image_oop"], errors="coerce")
    subset = subset.dropna()
    n = int(len(subset))
    if n < int(min_n_for_correlation):
        return {"computed": False, "reason": "too_few_rows", "n": n, "rho": None, "p_value": None}
    if subset["zdisc_pixel_fraction"].nunique() < 2 or subset["automated_image_oop"].nunique() < 2:
        return {"computed": False, "reason": "constant_input", "n": n, "rho": None, "p_value": None}
    rho, p_value = spearmanr(subset["zdisc_pixel_fraction"], subset["automated_image_oop"])
    return {
        "computed": bool(np.isfinite(rho)),
        "reason": "exploratory_pilot_only" if np.isfinite(rho) else "not_finite",
        "n": n,
        "rho": float(rho) if np.isfinite(rho) else None,
        "p_value": float(p_value) if np.isfinite(p_value) else None,
        "caution": "Exploratory pilot association only; no validation claim or threshold selection.",
    }


def write_full_image_validation_outputs(matched: pd.DataFrame, summary: dict[str, Any], paths: dict[str, Path]) -> None:
    paths["matched_csv"].parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(paths["matched_csv"], index=False)
    paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    lines = [f"{key}: {value}" for key, value in summary.items()]
    paths["summary_txt"].write_text("\n".join(lines) + "\n", encoding="utf-8")
