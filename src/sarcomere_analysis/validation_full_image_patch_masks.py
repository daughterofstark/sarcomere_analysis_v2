from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import output_dir
from .full_image_annotation import default_full_image_annotation_dir
from .validation_zdisc_masks import axial_angular_error_series, bool_column, iqr, require_columns, standardize_ids
from .zdisc_annotation import json_safe, shape_string
from .zdisc_annotation_features import annotation_status, connected_zdisc_components, estimate_mask_orientation
from .zdisc_draw_ui import load_crop_image, load_draw_index, load_mask


REQUIRED_FULL_IMAGE_INDEX_COLUMNS = [
    "annotation_id",
    "image_id",
    "donor_id",
    "annotation_image_path",
    "mask_path",
]

REQUIRED_PATCH_FEATURE_COLUMNS = [
    "image_id",
    "donor_id",
    "patch_id",
    "y0",
    "x0",
    "y1",
    "x1",
    "patch_oop",
    "patch_mean_orientation_deg",
]

AUTOMATED_PATCH_COLUMNS = [
    "image_id",
    "donor_id",
    "patch_id",
    "y0",
    "x0",
    "y1",
    "x1",
    "patch_oop",
    "patch_mean_orientation_deg",
    "valid_for_orientation",
    "valid_for_periodicity",
    "valid_for_spacing",
    "invalid_reason",
]

FULL_IMAGE_PATCH_VALIDATION_COLUMNS = [
    "annotation_id",
    "image_id",
    "donor_id",
    "patch_id",
    "y0",
    "x0",
    "y1",
    "x1",
    "manual_zdisc_pixel_count",
    "manual_ignore_pixel_count",
    "manual_zdisc_pixel_fraction",
    "manual_ignore_pixel_fraction",
    "manual_has_zdisc_labels",
    "manual_has_ignore_labels",
    "manual_patch_annotation_status",
    "manual_patch_orientation_deg",
    "manual_patch_orientation_confidence",
    "manual_patch_orientation_estimable",
    "reason_not_estimable",
    "automated_donor_id",
    "automated_patch_oop",
    "automated_patch_mean_orientation_deg",
    "automated_valid_for_orientation",
    "automated_valid_for_periodicity",
    "automated_valid_for_spacing",
    "automated_invalid_reason",
    "validation_match_status",
    "axial_orientation_error_deg",
]


def default_full_image_patch_validation_paths(
    cfg: dict[str, Any],
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    out_dir = Path(output_directory) if output_directory else output_dir(cfg) / "validation"
    return {
        "matched_csv": out_dir / "full_image_patch_mask_validation_matched.csv",
        "summary_json": out_dir / "full_image_patch_mask_validation_summary.json",
        "summary_txt": out_dir / "full_image_patch_mask_validation_summary.txt",
    }


def load_full_image_patch_validation_inputs(
    cfg: dict[str, Any],
    annotation_index: str | Path | None = None,
    patch_features: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = output_dir(cfg)
    index_path = Path(annotation_index) if annotation_index else default_full_image_annotation_dir(cfg) / "full_image_annotation_index.csv"
    patch_path = Path(patch_features) if patch_features else root / "tables" / "features_per_patch.csv"
    index = load_draw_index(index_path)
    patches = pd.read_csv(patch_path, dtype={"image_id": str, "donor_id": str, "patch_id": str})
    return index, patches


def validate_full_image_patch_masks(
    cfg: dict[str, Any],
    annotation_index: str | Path | None = None,
    mask_dir: str | Path | None = None,
    patch_features: str | Path | None = None,
    output_directory: str | Path | None = None,
    min_zdisc_pixels: int = 10,
    min_n_for_correlation: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    index, patches = load_full_image_patch_validation_inputs(cfg, annotation_index, patch_features)
    matched, summary = build_full_image_patch_mask_validation(
        index,
        patches,
        mask_dir=mask_dir,
        min_zdisc_pixels=min_zdisc_pixels,
        min_n_for_correlation=min_n_for_correlation,
    )
    paths = default_full_image_patch_validation_paths(cfg, output_directory)
    write_full_image_patch_validation_outputs(matched, summary, paths)
    return matched, summary, paths


def build_full_image_patch_mask_validation(
    annotation_index: pd.DataFrame,
    patch_features: pd.DataFrame,
    mask_dir: str | Path | None = None,
    min_zdisc_pixels: int = 10,
    min_n_for_correlation: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    index = annotation_index.copy(deep=True)
    patches = patch_features.copy(deep=True)
    require_columns(index, REQUIRED_FULL_IMAGE_INDEX_COLUMNS, "full-image annotation index")
    require_columns(patches, REQUIRED_PATCH_FEATURE_COLUMNS, "patch feature table")
    standardize_ids(index, ["annotation_id", "image_id", "donor_id"])
    standardize_ids(patches, ["image_id", "donor_id", "patch_id"])
    coerce_patch_coordinates(patches)

    duplicate_keys = patches.duplicated(["image_id", "patch_id"], keep=False)
    if duplicate_keys.any():
        duplicated = patches.loc[duplicate_keys, ["image_id", "patch_id"]].drop_duplicates().to_dict("records")
        raise ValueError(f"Patch feature table contains duplicate image_id/patch_id keys: {duplicated[:10]}")

    manual = extract_manual_patch_features_from_full_image_masks(
        index,
        patches,
        mask_dir=mask_dir,
        min_zdisc_pixels=min_zdisc_pixels,
    )
    matched = join_manual_patch_features_to_automated(manual, patches)
    summary = build_full_image_patch_validation_summary(
        matched,
        annotated_image_count=int(index["image_id"].nunique()),
        images_without_patch_features=images_without_patch_features(index, patches),
        min_n_for_correlation=min_n_for_correlation,
    )
    return stabilize_full_image_patch_validation_table(matched), summary


def coerce_patch_coordinates(patches: pd.DataFrame) -> None:
    for column in ["y0", "x0", "y1", "x1"]:
        patches[column] = pd.to_numeric(patches[column], errors="raise").astype(int)


def images_without_patch_features(index: pd.DataFrame, patches: pd.DataFrame) -> list[str]:
    annotated = set(index["image_id"].astype(str))
    available = set(patches["image_id"].astype(str))
    return sorted(annotated.difference(available))


def extract_manual_patch_features_from_full_image_masks(
    index: pd.DataFrame,
    patches: pd.DataFrame,
    mask_dir: str | Path | None = None,
    min_zdisc_pixels: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    patch_groups = {str(image_id): group.copy() for image_id, group in patches.groupby("image_id", sort=False)}
    for _, index_row in index.iterrows():
        image_id = str(index_row["image_id"])
        if image_id not in patch_groups:
            continue
        image = load_crop_image(index_row["annotation_image_path"])
        mask_path = resolve_mask_path(index_row["mask_path"], mask_dir)
        mask = load_mask(mask_path, expected_shape=image.shape)
        for _, patch_row in patch_groups[image_id].iterrows():
            rows.append(extract_one_manual_patch_feature(index_row, patch_row, mask, min_zdisc_pixels=min_zdisc_pixels))
    return pd.DataFrame(rows)


def resolve_mask_path(mask_path: str | Path, mask_dir: str | Path | None = None) -> Path:
    original = Path(mask_path)
    if mask_dir is None:
        return original
    return Path(mask_dir) / original.name


def extract_one_manual_patch_feature(
    index_row: pd.Series,
    patch_row: pd.Series,
    full_mask: np.ndarray,
    min_zdisc_pixels: int = 10,
) -> dict[str, Any]:
    y0, x0, y1, x1 = bounded_patch_coordinates(patch_row, full_mask.shape)
    patch_mask = full_mask[y0:y1, x0:x1]
    zdisc = patch_mask == 1
    ignore = patch_mask == 2
    zdisc_count = int(np.sum(zdisc))
    ignore_count = int(np.sum(ignore))
    total_pixels = int(patch_mask.size)
    component_labels, component_count = connected_zdisc_components(zdisc)
    orientation = estimate_mask_orientation(
        zdisc,
        component_count=component_count,
        min_zdisc_pixels=min_zdisc_pixels,
        min_components=1,
    )
    return {
        "annotation_id": str(index_row["annotation_id"]),
        "image_id": str(index_row["image_id"]),
        "donor_id": str(index_row["donor_id"]),
        "patch_id": str(patch_row["patch_id"]),
        "y0": int(y0),
        "x0": int(x0),
        "y1": int(y1),
        "x1": int(x1),
        "manual_mask_shape": shape_string(patch_mask.shape),
        "manual_zdisc_pixel_count": zdisc_count,
        "manual_ignore_pixel_count": ignore_count,
        "manual_zdisc_pixel_fraction": zdisc_count / total_pixels if total_pixels else np.nan,
        "manual_ignore_pixel_fraction": ignore_count / total_pixels if total_pixels else np.nan,
        "manual_has_zdisc_labels": bool(zdisc_count > 0),
        "manual_has_ignore_labels": bool(ignore_count > 0),
        "manual_patch_annotation_status": annotation_status(zdisc_count, ignore_count),
        "manual_patch_orientation_deg": orientation["manual_mask_orientation_deg"],
        "manual_patch_orientation_confidence": orientation["manual_mask_orientation_confidence"],
        "manual_patch_orientation_estimable": orientation["orientation_estimable"],
        "reason_not_estimable": orientation["reason_not_estimable"],
    }


def bounded_patch_coordinates(patch_row: pd.Series, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = int(shape[0]), int(shape[1])
    y0 = max(0, min(int(patch_row["y0"]), height))
    x0 = max(0, min(int(patch_row["x0"]), width))
    y1 = max(0, min(int(patch_row["y1"]), height))
    x1 = max(0, min(int(patch_row["x1"]), width))
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 < x0:
        x0, x1 = x1, x0
    return y0, x0, y1, x1


def join_manual_patch_features_to_automated(manual: pd.DataFrame, patches: pd.DataFrame) -> pd.DataFrame:
    if manual.empty:
        manual = pd.DataFrame(columns=[column for column in FULL_IMAGE_PATCH_VALIDATION_COLUMNS if not column.startswith("automated_")])
    patch_subset_columns = [column for column in AUTOMATED_PATCH_COLUMNS if column in patches.columns]
    patch_subset = patches[patch_subset_columns].copy()
    patch_subset = patch_subset.rename(
        columns={
            "donor_id": "automated_donor_id",
            "patch_oop": "automated_patch_oop",
            "patch_mean_orientation_deg": "automated_patch_mean_orientation_deg",
            "valid_for_orientation": "automated_valid_for_orientation",
            "valid_for_periodicity": "automated_valid_for_periodicity",
            "valid_for_spacing": "automated_valid_for_spacing",
            "invalid_reason": "automated_invalid_reason",
        }
    )
    matched = manual.merge(patch_subset, on=["image_id", "patch_id", "y0", "x0", "y1", "x1"], how="left", indicator="_patch_join_status")
    joined = matched["_patch_join_status"] == "both"
    donor_mismatch = joined & (matched["donor_id"].astype(str) != matched["automated_donor_id"].astype(str))
    matched["validation_match_status"] = np.select(
        [~joined, donor_mismatch],
        ["unmatched_patch", "donor_id_mismatch"],
        default="matched",
    )
    matched["manual_patch_orientation_deg"] = pd.to_numeric(matched["manual_patch_orientation_deg"], errors="coerce")
    matched["automated_patch_mean_orientation_deg"] = pd.to_numeric(matched["automated_patch_mean_orientation_deg"], errors="coerce")
    matched["automated_patch_oop"] = pd.to_numeric(matched["automated_patch_oop"], errors="coerce")
    matched["manual_zdisc_pixel_fraction"] = pd.to_numeric(matched["manual_zdisc_pixel_fraction"], errors="coerce")
    matched["manual_patch_orientation_estimable_bool"] = bool_column(matched, "manual_patch_orientation_estimable")
    orientation_pair_mask = (
        (matched["validation_match_status"] == "matched")
        & matched["manual_patch_orientation_estimable_bool"]
        & matched["manual_patch_orientation_deg"].notna()
        & matched["automated_patch_mean_orientation_deg"].notna()
    )
    errors = axial_angular_error_series(
        matched.loc[orientation_pair_mask, "manual_patch_orientation_deg"],
        matched.loc[orientation_pair_mask, "automated_patch_mean_orientation_deg"],
    )
    matched["axial_orientation_error_deg"] = np.nan
    matched.loc[orientation_pair_mask, "axial_orientation_error_deg"] = errors.to_numpy()
    return matched


def build_full_image_patch_validation_summary(
    matched: pd.DataFrame,
    annotated_image_count: int,
    images_without_patch_features: list[str],
    min_n_for_correlation: int = 10,
) -> dict[str, Any]:
    orientation_errors = pd.to_numeric(matched.get("axial_orientation_error_deg", pd.Series(dtype=float)), errors="coerce").dropna()
    status_counts = matched.get("manual_patch_annotation_status", pd.Series(dtype=str)).fillna("missing").astype(str).value_counts().to_dict()
    total_patch_rows = int(len(matched))
    return json_safe(
        {
            "mode": "pilot_full_image_patch_mask_validation",
            "full_images_with_masks": int(annotated_image_count),
            "full_images_without_patch_features": images_without_patch_features,
            "total_automated_patches_in_annotated_images": total_patch_rows,
            "matched_patch_rows": int((matched.get("validation_match_status", "") == "matched").sum()) if total_patch_rows else 0,
            "unmatched_patch_rows": int((matched.get("validation_match_status", "") == "unmatched_patch").sum()) if total_patch_rows else 0,
            "donor_id_mismatches": int((matched.get("validation_match_status", "") == "donor_id_mismatch").sum()) if total_patch_rows else 0,
            "patches_with_manual_zdisc_labels": int(bool_column(matched, "manual_has_zdisc_labels").sum()) if total_patch_rows else 0,
            "patches_empty": int((matched.get("manual_patch_annotation_status", "") == "empty").sum()) if total_patch_rows else 0,
            "patches_ignore_only": int((matched.get("manual_patch_annotation_status", "") == "ignore_only").sum()) if total_patch_rows else 0,
            "patches_mixed": int((matched.get("manual_patch_annotation_status", "") == "mixed").sum()) if total_patch_rows else 0,
            "patches_manual_orientation_estimable": int(bool_column(matched, "manual_patch_orientation_estimable").sum()) if total_patch_rows else 0,
            "manual_patch_annotation_status_counts": {str(key): int(value) for key, value in status_counts.items()},
            "n_orientation_pairs": int(len(orientation_errors)),
            "median_axial_error_deg": float(np.nanmedian(orientation_errors)) if len(orientation_errors) else np.nan,
            "mean_axial_error_deg": float(np.nanmean(orientation_errors)) if len(orientation_errors) else np.nan,
            "iqr_axial_error_deg": iqr(orientation_errors),
            "oop_medians_by_manual_patch_status": group_patch_oop_medians(matched),
            "spearman_zdisc_fraction_vs_patch_oop": patch_spearman_summary(matched, min_n_for_correlation=min_n_for_correlation),
            "interpretation_flags": [
                "pilot_local_validation_only",
                "manual_masks_drawn_by_user_not_independent_blinded_expert",
                "full_image_masks_are_sparse_annotations_not_exhaustive_segmentation",
                "patch_level_comparison_more_appropriate_than_image_level_for_sparse_manual_labels",
                "no_clinical_or_biological_claims_made",
                "does_not_validate_spacing",
                "spacing_remains_exploratory_low_yield",
                "no_p_values_or_clinical_statistics_computed",
            ],
        }
    )


def group_patch_oop_medians(matched: pd.DataFrame) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for status in ["zdisc_labeled", "empty", "ignore_only", "mixed"]:
        values = pd.to_numeric(
            matched.loc[matched.get("manual_patch_annotation_status") == status, "automated_patch_oop"],
            errors="coerce",
        ).dropna()
        result[status] = None if values.empty else float(np.nanmedian(values))
    return result


def patch_spearman_summary(matched: pd.DataFrame, min_n_for_correlation: int = 10) -> dict[str, Any]:
    if matched.empty:
        return {"computed": False, "reason": "too_few_rows", "n": 0, "rho": None, "p_value": None}
    subset = matched.loc[matched["validation_match_status"] == "matched", ["manual_zdisc_pixel_fraction", "automated_patch_oop"]].copy()
    subset["manual_zdisc_pixel_fraction"] = pd.to_numeric(subset["manual_zdisc_pixel_fraction"], errors="coerce")
    subset["automated_patch_oop"] = pd.to_numeric(subset["automated_patch_oop"], errors="coerce")
    subset = subset.dropna()
    n = int(len(subset))
    if n < int(min_n_for_correlation):
        return {"computed": False, "reason": "too_few_rows", "n": n, "rho": None, "p_value": None}
    if subset["manual_zdisc_pixel_fraction"].nunique() < 2 or subset["automated_patch_oop"].nunique() < 2:
        return {"computed": False, "reason": "constant_input", "n": n, "rho": None, "p_value": None}
    rho, p_value = spearmanr(subset["manual_zdisc_pixel_fraction"], subset["automated_patch_oop"])
    return {
        "computed": bool(np.isfinite(rho)),
        "reason": "exploratory_pilot_only" if np.isfinite(rho) else "not_finite",
        "n": n,
        "rho": float(rho) if np.isfinite(rho) else None,
        "p_value": float(p_value) if np.isfinite(p_value) else None,
        "caution": "Exploratory pilot association only; no validation claim or threshold selection.",
    }


def stabilize_full_image_patch_validation_table(matched: pd.DataFrame) -> pd.DataFrame:
    result = matched.copy(deep=True)
    for column in FULL_IMAGE_PATCH_VALIDATION_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
    for column in ["annotation_id", "image_id", "donor_id", "patch_id", "validation_match_status"]:
        result[column] = result[column].fillna("").astype(str)
    return result[FULL_IMAGE_PATCH_VALIDATION_COLUMNS]


def write_full_image_patch_validation_outputs(matched: pd.DataFrame, summary: dict[str, Any], paths: dict[str, Path]) -> None:
    paths["matched_csv"].parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(paths["matched_csv"], index=False)
    paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    lines = [f"{key}: {value}" for key, value in summary.items()]
    paths["summary_txt"].write_text("\n".join(lines) + "\n", encoding="utf-8")
