from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import output_dir
from .zdisc_annotation import json_safe


CANONICAL_COLUMNS = [
    "annotation_id",
    "patch_filename",
    "striations_visible",
    "organisation_score",
    "dominant_orientation_deg",
    "confidence_score",
    "spacing_measurable",
    "manual_sarcomere_length_um_optional",
    "notes",
]

NORMALIZED_COLUMNS = [
    "annotation_id",
    "patch_filename",
    "striations_visible",
    "organisation_score",
    "confidence_score",
    "spacing_measurable",
    "manual_sarcomere_length_um_optional",
    "notes",
    "expert_dominant_orientation_deg_raw",
    "expert_orientation_usable_primary",
    "audit_flags",
]

MATCHED_COLUMNS = [
    "annotation_id",
    "patch_filename",
    "image_id",
    "donor_id",
    "patch_id",
    "oop_bin",
    "automated_patch_oop",
    "automated_patch_orientation_deg",
    "striations_visible",
    "organisation_score",
    "confidence_score",
    "spacing_measurable",
    "manual_sarcomere_length_um_optional",
    "expert_dominant_orientation_deg_raw",
    "expert_orientation_usable_primary",
    "validation_match_status",
]

ALLOWED_TRI_STATE = {"yes", "unclear", "no"}


def default_expert_annotation_validation_paths(
    cfg: dict[str, Any],
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    out_dir = Path(output_directory) if output_directory else output_dir(cfg) / "validation" / "expert_annotation_validation"
    return {
        "normalized_csv": out_dir / "expert_annotations_normalized.csv",
        "matched_csv": out_dir / "expert_annotation_validation_matched.csv",
        "summary_json": out_dir / "expert_annotation_validation_summary.json",
        "summary_txt": out_dir / "expert_annotation_validation_summary.txt",
    }


def default_annotations_path(cfg: dict[str, Any]) -> Path:
    return output_dir(cfg) / "expert_annotation_pack" / "expert_annotation_template_NG.csv"


def default_internal_key_path(cfg: dict[str, Any]) -> Path:
    return output_dir(cfg) / "expert_annotation_pack" / "internal_blinding_key.csv"


def validate_expert_annotations(
    cfg: dict[str, Any],
    annotations: str | Path | None = None,
    internal_key: str | Path | None = None,
    output_directory: str | Path | None = None,
    min_n_correlation: int = 10,
    min_confidence: int = 3,
    allow_orientation_exploratory: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    annotation_path = Path(annotations) if annotations else default_annotations_path(cfg)
    key_path = Path(internal_key) if internal_key else default_internal_key_path(cfg)
    raw_annotations = pd.read_csv(annotation_path, dtype={"annotation_id": str, "patch_filename": str})
    key = pd.read_csv(key_path, dtype={"annotation_id": str, "patch_filename": str, "image_id": str, "donor_id": str, "patch_id": str})
    normalized, normalization_audit = normalize_expert_annotations(raw_annotations)
    matched = match_expert_annotations(normalized, key)
    summary = build_expert_annotation_summary(
        normalized,
        matched,
        normalization_audit=normalization_audit,
        min_n_correlation=min_n_correlation,
        min_confidence=min_confidence,
        allow_orientation_exploratory=allow_orientation_exploratory,
    )
    paths = default_expert_annotation_validation_paths(cfg, output_directory)
    write_expert_annotation_validation_outputs(normalized, matched, summary, paths)
    return normalized, matched, summary, paths


def normalize_expert_annotations(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = raw.copy(deep=True)
    data = drop_empty_unnamed_columns(data)
    rename_map = {column: normalize_column_name(column) for column in data.columns}
    data = data.rename(columns=rename_map)
    duplicate_columns = data.columns[data.columns.duplicated()].tolist()
    if duplicate_columns:
        data = data.loc[:, ~data.columns.duplicated()].copy()
    for column in CANONICAL_COLUMNS:
        if column not in data.columns:
            data[column] = np.nan

    result = pd.DataFrame()
    result["annotation_id"] = data["annotation_id"].fillna("").astype(str).str.strip()
    result["patch_filename"] = data["patch_filename"].fillna("").astype(str).str.strip()
    result["striations_visible"], invalid_visibility = normalize_categorical(data["striations_visible"], ALLOWED_TRI_STATE)
    result["organisation_score"], invalid_organisation = normalize_score(data["organisation_score"], "organisation_score")
    result["confidence_score"], invalid_confidence = normalize_score(data["confidence_score"], "confidence_score")
    result["spacing_measurable"], invalid_spacing = normalize_categorical(data["spacing_measurable"], ALLOWED_TRI_STATE, allow_blank=True)
    result["manual_sarcomere_length_um_optional"] = pd.to_numeric(data["manual_sarcomere_length_um_optional"], errors="coerce")
    result["notes"] = data["notes"].fillna("").astype(str)
    result["expert_dominant_orientation_deg_raw"] = data["dominant_orientation_deg"]
    result["expert_orientation_usable_primary"] = False

    flags = []
    for index in result.index:
        row_flags = []
        if bool(invalid_visibility.loc[index]):
            row_flags.append("invalid_striations_visible")
        if bool(invalid_organisation.loc[index]):
            row_flags.append("invalid_organisation_score")
        if bool(invalid_confidence.loc[index]):
            row_flags.append("invalid_confidence_score")
        if bool(invalid_spacing.loc[index]):
            row_flags.append("invalid_spacing_measurable")
        flags.append(";".join(row_flags) if row_flags else "ok")
    result["audit_flags"] = flags
    result = result[NORMALIZED_COLUMNS]
    audit = {
        "raw_columns": list(raw.columns),
        "normalized_columns": list(data.columns),
        "dropped_duplicate_normalized_columns": [str(column) for column in duplicate_columns],
        "invalid_values_by_field": {
            "striations_visible": int(invalid_visibility.sum()),
            "organisation_score": int(invalid_organisation.sum()),
            "confidence_score": int(invalid_confidence.sum()),
            "spacing_measurable": int(invalid_spacing.sum()),
        },
    }
    return result, audit


def normalize_column_name(column: str) -> str:
    name = str(column).strip().lower()
    while name.endswith("*"):
        name = name[:-1].strip()
    name = name.replace(" ", "_")
    return name


def drop_empty_unnamed_columns(data: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for column in data.columns:
        is_unnamed = str(column).strip().lower().startswith("unnamed:")
        if is_unnamed and data[column].isna().all():
            continue
        keep.append(column)
    return data[keep].copy()


def normalize_categorical(values: pd.Series, allowed: set[str], allow_blank: bool = False) -> tuple[pd.Series, pd.Series]:
    raw = values.copy()
    normalized = raw.fillna("").astype(str).str.strip().str.lower()
    normalized = normalized.replace({"": np.nan, "nan": np.nan, "n/a": np.nan, "na": np.nan})
    invalid = normalized.notna() & ~normalized.isin(allowed)
    normalized = normalized.mask(invalid, np.nan)
    if not allow_blank:
        invalid = invalid.fillna(False)
    return normalized, invalid.fillna(False)


def normalize_score(values: pd.Series, field_name: str) -> tuple[pd.Series, pd.Series]:
    _ = field_name
    numeric = pd.to_numeric(values, errors="coerce")
    raw_present = values.notna() & (values.astype(str).str.strip() != "")
    invalid = raw_present & (numeric.isna() | (numeric < 1) | (numeric > 5))
    numeric = numeric.mask(invalid, np.nan)
    return numeric, invalid.fillna(False)


def match_expert_annotations(normalized: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    key_table = key.copy(deep=True)
    for column in ["annotation_id", "patch_filename", "image_id", "donor_id", "patch_id"]:
        if column in key_table.columns:
            key_table[column] = key_table[column].fillna("").astype(str).str.strip()
    key_columns = [
        column
        for column in [
            "annotation_id",
            "patch_filename",
            "image_id",
            "donor_id",
            "patch_id",
            "oop_bin",
            "automated_patch_oop",
            "automated_patch_orientation_deg",
            "health_status",
        ]
        if column in key_table.columns
    ]
    matched = normalized.merge(
        key_table[key_columns],
        on=["annotation_id", "patch_filename"],
        how="left",
        indicator="_key_join_status",
    )
    matched["validation_match_status"] = np.where(matched["_key_join_status"] == "both", "matched", "unmatched_annotation")
    matched["automated_patch_oop"] = pd.to_numeric(matched.get("automated_patch_oop", np.nan), errors="coerce")
    matched["automated_patch_orientation_deg"] = pd.to_numeric(matched.get("automated_patch_orientation_deg", np.nan), errors="coerce")
    for column in MATCHED_COLUMNS:
        if column not in matched.columns:
            matched[column] = np.nan
    return matched[MATCHED_COLUMNS]


def build_expert_annotation_summary(
    normalized: pd.DataFrame,
    matched: pd.DataFrame,
    normalization_audit: dict[str, Any],
    min_n_correlation: int = 10,
    min_confidence: int = 3,
    allow_orientation_exploratory: bool = False,
) -> dict[str, Any]:
    duplicate_ids = sorted(normalized.loc[normalized["annotation_id"].duplicated(keep=False), "annotation_id"].dropna().unique().tolist())
    unmatched = sorted(matched.loc[matched["validation_match_status"] == "unmatched_annotation", "annotation_id"].dropna().astype(str).tolist())
    high_conf = matched.loc[pd.to_numeric(matched["confidence_score"], errors="coerce") >= int(min_confidence)].copy()
    summary = {
        "mode": "expert_annotation_oop_validation",
        "audit": {
            "total_rows": int(len(normalized)),
            "annotations_matched_to_internal_key": int((matched["validation_match_status"] == "matched").sum()),
            "unmatched_annotation_ids": unmatched,
            "duplicate_annotation_ids": duplicate_ids,
            "completed_striations_visible_count": int(normalized["striations_visible"].notna().sum()),
            "completed_organisation_score_count": int(normalized["organisation_score"].notna().sum()),
            "completed_confidence_score_count": int(normalized["confidence_score"].notna().sum()),
            "completed_spacing_measurable_count": int(normalized["spacing_measurable"].notna().sum()),
            "manual_sarcomere_length_completed_count": int(normalized["manual_sarcomere_length_um_optional"].notna().sum()),
            "invalid_values_by_field": normalization_audit["invalid_values_by_field"],
        },
        "visibility_vs_automated_oop": {
            "counts": grouped_counts(matched, "striations_visible", ["yes", "unclear", "no"]),
            "oop_medians": grouped_medians(matched, "striations_visible", "automated_patch_oop", ["yes", "unclear", "no"]),
        },
        "organisation_score_vs_automated_oop": {
            "counts": grouped_counts(matched, "organisation_score", [1, 2, 3, 4, 5]),
            "oop_medians": grouped_medians(matched, "organisation_score", "automated_patch_oop", [1, 2, 3, 4, 5]),
            "spearman": spearman_summary(matched, "organisation_score", "automated_patch_oop", min_n_correlation),
        },
        "oop_bin_vs_manual_organisation": {
            "counts": grouped_counts(matched, "oop_bin", ["low", "medium", "high"]),
            "median_organisation_score": grouped_medians(matched, "oop_bin", "organisation_score", ["low", "medium", "high"]),
            "mean_confidence_score": grouped_means(matched, "oop_bin", "confidence_score", ["low", "medium", "high"]),
            "median_confidence_score": grouped_medians(matched, "oop_bin", "confidence_score", ["low", "medium", "high"]),
        },
        "confidence_filtered": {
            "min_confidence": int(min_confidence),
            "row_count": int(len(high_conf)),
            "visibility_counts": grouped_counts(high_conf, "striations_visible", ["yes", "unclear", "no"]),
            "visibility_oop_medians": grouped_medians(high_conf, "striations_visible", "automated_patch_oop", ["yes", "unclear", "no"]),
            "organisation_oop_spearman": spearman_summary(high_conf, "organisation_score", "automated_patch_oop", min_n_correlation),
            "low_confidence_count": int((pd.to_numeric(matched["confidence_score"], errors="coerce") < int(min_confidence)).sum()),
        },
        "spacing": {
            "spacing_measurable_counts": grouped_counts(normalized, "spacing_measurable", ["yes", "unclear", "no"]),
            "manual_sarcomere_length_completed_count": int(normalized["manual_sarcomere_length_um_optional"].notna().sum()),
            "spacing_validation_status": "not_validated_from_this_file",
            "statement": "Manual sarcomere length was not completed sufficiently; spacing remains exploratory_low_yield.",
        },
        "orientation": {
            "dominant_orientation_used_as_primary": False,
            "expert_orientation_usable_primary_all_false": bool((normalized["expert_orientation_usable_primary"] == False).all()),
            "orientation_exploratory_requested": bool(allow_orientation_exploratory),
            "statement": "dominant_orientation_deg is not used for primary validation because the expert reported ambiguity in interpretation.",
        },
        "interpretation_flags": [
            "blinded_expert_annotation_validation_of_oop_organisation",
            "primary_manual_endpoints_are_striations_visible_organisation_score_confidence_score",
            "dominant_orientation_deg_not_used_as_primary_due_to_annotation_ambiguity",
            "manual_sarcomere_length_not_completed_spacing_not_validated",
            "no_clinical_or_disease_comparisons",
            "no_production_algorithms_changed",
        ],
    }
    return json_safe(summary)


def grouped_counts(table: pd.DataFrame, group_column: str, levels: list[Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    values = table[group_column] if group_column in table.columns else pd.Series(dtype=object)
    for level in levels:
        if isinstance(level, (int, float)):
            comparable = pd.to_numeric(values, errors="coerce")
            count = int((comparable == float(level)).sum())
        else:
            count = int((values.fillna("").astype(str).str.lower() == str(level).lower()).sum())
        result[str(level)] = count
    return result


def grouped_medians(table: pd.DataFrame, group_column: str, value_column: str, levels: list[Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for level in levels:
        subset = subset_for_level(table, group_column, level)
        values = pd.to_numeric(subset.get(value_column, pd.Series(dtype=float)), errors="coerce").dropna()
        result[str(level)] = None if values.empty else float(np.nanmedian(values))
    return result


def grouped_means(table: pd.DataFrame, group_column: str, value_column: str, levels: list[Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for level in levels:
        subset = subset_for_level(table, group_column, level)
        values = pd.to_numeric(subset.get(value_column, pd.Series(dtype=float)), errors="coerce").dropna()
        result[str(level)] = None if values.empty else float(np.nanmean(values))
    return result


def subset_for_level(table: pd.DataFrame, group_column: str, level: Any) -> pd.DataFrame:
    if group_column not in table.columns:
        return table.head(0)
    if isinstance(level, (int, float)):
        values = pd.to_numeric(table[group_column], errors="coerce")
        return table.loc[values == float(level)]
    return table.loc[table[group_column].fillna("").astype(str).str.lower() == str(level).lower()]


def spearman_summary(table: pd.DataFrame, x_column: str, y_column: str, min_n: int) -> dict[str, Any]:
    if x_column not in table.columns or y_column not in table.columns:
        return {"computed": False, "reason": "missing_columns", "n": 0, "rho": None, "p_value": None}
    subset = table[[x_column, y_column]].copy()
    subset[x_column] = pd.to_numeric(subset[x_column], errors="coerce")
    subset[y_column] = pd.to_numeric(subset[y_column], errors="coerce")
    subset = subset.dropna()
    n = int(len(subset))
    if n < int(min_n):
        return {"computed": False, "reason": "too_few_rows", "n": n, "rho": None, "p_value": None}
    if subset[x_column].nunique() < 2 or subset[y_column].nunique() < 2:
        return {"computed": False, "reason": "constant_input", "n": n, "rho": None, "p_value": None}
    rho, p_value = spearmanr(subset[x_column], subset[y_column])
    return {
        "computed": bool(np.isfinite(rho)),
        "reason": "computed" if np.isfinite(rho) else "not_finite",
        "n": n,
        "rho": float(rho) if np.isfinite(rho) else None,
        "p_value": float(p_value) if np.isfinite(p_value) else None,
        "caution": "Validation-supporting but still pilot/exploratory; no clinical inference.",
    }


def write_expert_annotation_validation_outputs(
    normalized: pd.DataFrame,
    matched: pd.DataFrame,
    summary: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    paths["normalized_csv"].parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(paths["normalized_csv"], index=False)
    matched.to_csv(paths["matched_csv"], index=False)
    paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    paths["summary_txt"].write_text(render_summary_text(summary), encoding="utf-8")


def render_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Expert annotation validation summary",
        f"total_rows: {summary['audit']['total_rows']}",
        f"matched_rows: {summary['audit']['annotations_matched_to_internal_key']}",
        f"completed_striations_visible_count: {summary['audit']['completed_striations_visible_count']}",
        f"completed_organisation_score_count: {summary['audit']['completed_organisation_score_count']}",
        f"completed_confidence_score_count: {summary['audit']['completed_confidence_score_count']}",
        f"visibility_oop_medians: {summary['visibility_vs_automated_oop']['oop_medians']}",
        f"organisation_oop_medians: {summary['organisation_score_vs_automated_oop']['oop_medians']}",
        f"organisation_oop_spearman: {summary['organisation_score_vs_automated_oop']['spearman']}",
        summary["orientation"]["statement"],
        summary["spacing"]["statement"],
    ]
    return "\n".join(lines) + "\n"
