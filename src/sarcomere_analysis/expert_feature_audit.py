from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

from .config import output_dir
from .zdisc_annotation import json_safe


MANUAL_COLUMNS = [
    "annotation_id",
    "patch_filename",
    "image_id",
    "donor_id",
    "patch_id",
    "striations_visible",
    "organisation_score",
    "confidence_score",
    "spacing_measurable",
    "validation_match_status",
]

IDENTIFIER_OR_LEAKAGE_TERMS = [
    "annotation_id",
    "patch_filename",
    "image_id",
    "donor_id",
    "patch_id",
    "health_status",
    "diagnosis",
    "disease",
    "label",
    "manual",
    "expert",
    "notes",
]

EXPLICIT_EXCLUDED_COLUMNS = {
    "striations_visible",
    "organisation_score",
    "confidence_score",
    "spacing_measurable",
    "validation_match_status",
    "oop_bin",
    "x0",
    "x1",
    "y0",
    "y1",
    "center_x",
    "center_y",
}

VISIBILITY_COLUMNS = [
    "feature",
    "n",
    "median_yes",
    "median_unclear",
    "median_no",
    "yes_minus_no",
    "abs_yes_minus_no",
    "p_value_or_blank",
    "missing_count",
]

ORGANISATION_COLUMNS = [
    "feature",
    "n",
    "spearman_rho",
    "spearman_p",
    "n_confidence_filtered",
    "spearman_rho_confidence_filtered",
    "spearman_p_confidence_filtered",
    "missing_count",
]

CONFIDENCE_COLUMNS = [
    "feature",
    "n",
    "spearman_rho",
    "spearman_p",
    "missing_count",
]


def default_expert_feature_audit_paths(cfg: dict[str, Any], output_directory: str | Path | None = None) -> dict[str, Path]:
    out_dir = Path(output_directory) if output_directory else output_dir(cfg) / "validation" / "expert_feature_audit"
    return {
        "feature_table": out_dir / "expert_feature_audit_feature_table.csv",
        "visibility_summary": out_dir / "expert_feature_audit_visibility_summary.csv",
        "organisation_summary": out_dir / "expert_feature_audit_organisation_summary.csv",
        "confidence_summary": out_dir / "expert_feature_audit_confidence_summary.csv",
        "summary_json": out_dir / "expert_feature_audit_summary.json",
        "summary_txt": out_dir / "expert_feature_audit_summary.txt",
    }


def audit_expert_feature_relationships(
    cfg: dict[str, Any],
    matched: str | Path | None = None,
    patch_features: str | Path | None = None,
    output_directory: str | Path | None = None,
    min_n: int = 10,
    min_confidence: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    root = output_dir(cfg)
    matched_path = Path(matched) if matched else root / "validation" / "expert_annotation_validation" / "expert_annotation_validation_matched.csv"
    patch_path = Path(patch_features) if patch_features else root / "tables" / "features_per_patch.csv"
    matched_df = pd.read_csv(matched_path, dtype={"annotation_id": str, "patch_filename": str, "image_id": str, "donor_id": str, "patch_id": str})
    patch_df = pd.read_csv(patch_path, dtype={"image_id": str, "donor_id": str, "patch_id": str})

    feature_table, feature_columns, excluded_columns = build_expert_feature_table(matched_df, patch_df)
    visibility = visibility_feature_summary(feature_table, feature_columns)
    organisation = organisation_feature_summary(feature_table, feature_columns, min_n=min_n, min_confidence=min_confidence)
    confidence = confidence_feature_summary(feature_table, feature_columns, min_n=min_n)
    summary = build_feature_audit_summary(
        feature_table,
        feature_columns,
        excluded_columns,
        visibility,
        organisation,
        confidence,
        min_n=min_n,
        min_confidence=min_confidence,
    )
    paths = default_expert_feature_audit_paths(cfg, output_directory)
    write_feature_audit_outputs(feature_table, visibility, organisation, confidence, summary, paths)
    return feature_table, visibility, organisation, confidence, summary, paths


def build_expert_feature_table(
    matched: pd.DataFrame,
    patch_features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    manual = matched.copy(deep=True)
    patches = patch_features.copy(deep=True)
    for frame in [manual, patches]:
        for column in ["image_id", "donor_id", "patch_id"]:
            if column in frame.columns:
                frame[column] = frame[column].fillna("").astype(str)
    joined = manual.merge(patches, on=["image_id", "donor_id", "patch_id"], how="left", suffixes=("", "_patch"))
    for column in MANUAL_COLUMNS:
        if column not in joined.columns:
            joined[column] = np.nan
    joined["organisation_score"] = pd.to_numeric(joined["organisation_score"], errors="coerce")
    joined["confidence_score"] = pd.to_numeric(joined["confidence_score"], errors="coerce")
    feature_columns, excluded_columns = identify_numeric_feature_columns(joined)
    output_columns = MANUAL_COLUMNS + feature_columns
    return joined[output_columns].copy(), feature_columns, excluded_columns


def identify_numeric_feature_columns(table: pd.DataFrame) -> tuple[list[str], list[str]]:
    feature_columns: list[str] = []
    excluded_columns: list[str] = []
    for column in table.columns:
        if is_excluded_column(column):
            excluded_columns.append(column)
            continue
        values = table[column]
        if pd.api.types.is_bool_dtype(values):
            table[column] = values.fillna(False).astype(int)
            feature_columns.append(column)
            continue
        if values.dtype == object:
            lowered = values.dropna().astype(str).str.lower()
            if len(lowered) and lowered.isin({"true", "false", "1", "0", "yes", "no"}).all():
                table[column] = values.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"}).astype(int)
                feature_columns.append(column)
                continue
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().any():
            table[column] = numeric
            feature_columns.append(column)
        else:
            excluded_columns.append(column)
    return feature_columns, sorted(set(excluded_columns))


def is_excluded_column(column: str) -> bool:
    name = str(column).lower()
    return name in EXPLICIT_EXCLUDED_COLUMNS or any(term in name for term in IDENTIFIER_OR_LEAKAGE_TERMS)


def visibility_feature_summary(feature_table: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for feature in feature_columns:
        values = pd.to_numeric(feature_table[feature], errors="coerce")
        medians = {
            level: median_for_group(feature_table, values, "striations_visible", level)
            for level in ["yes", "unclear", "no"]
        }
        yes_minus_no = np.nan
        if medians["yes"] is not None and medians["no"] is not None:
            yes_minus_no = float(medians["yes"] - medians["no"])
        rows.append(
            {
                "feature": feature,
                "n": int(values.notna().sum()),
                "median_yes": medians["yes"],
                "median_unclear": medians["unclear"],
                "median_no": medians["no"],
                "yes_minus_no": yes_minus_no,
                "abs_yes_minus_no": abs(yes_minus_no) if np.isfinite(yes_minus_no) else np.nan,
                "p_value_or_blank": kruskal_visibility_p_value(feature_table, feature),
                "missing_count": int(values.isna().sum()),
            }
        )
    result = pd.DataFrame(rows, columns=VISIBILITY_COLUMNS)
    if not result.empty:
        result = result.sort_values(["abs_yes_minus_no", "missing_count"], ascending=[False, True], na_position="last").reset_index(drop=True)
    return result


def median_for_group(table: pd.DataFrame, values: pd.Series, group_column: str, level: str) -> float | None:
    mask = table[group_column].fillna("").astype(str).str.lower() == str(level).lower()
    subset = values.loc[mask].dropna()
    return None if subset.empty else float(np.nanmedian(subset))


def kruskal_visibility_p_value(table: pd.DataFrame, feature: str) -> float | None:
    groups = []
    values = pd.to_numeric(table[feature], errors="coerce")
    for level in ["yes", "unclear", "no"]:
        mask = table["striations_visible"].fillna("").astype(str).str.lower() == level
        group = values.loc[mask].dropna()
        if len(group) >= 2:
            groups.append(group.to_numpy())
    if len(groups) < 2:
        return None
    combined = np.concatenate(groups)
    if np.unique(combined).size < 2:
        return None
    try:
        stat = kruskal(*groups)
    except ValueError:
        return None
    return float(stat.pvalue) if np.isfinite(stat.pvalue) else None


def organisation_feature_summary(
    feature_table: pd.DataFrame,
    feature_columns: list[str],
    min_n: int = 10,
    min_confidence: int = 3,
) -> pd.DataFrame:
    rows = []
    high_conf = feature_table.loc[pd.to_numeric(feature_table["confidence_score"], errors="coerce") >= int(min_confidence)].copy()
    for feature in feature_columns:
        full = spearman_pair(feature_table, "organisation_score", feature, min_n=min_n)
        filtered = spearman_pair(high_conf, "organisation_score", feature, min_n=min_n)
        values = pd.to_numeric(feature_table[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "n": full["n"],
                "spearman_rho": full["rho"],
                "spearman_p": full["p_value"],
                "n_confidence_filtered": filtered["n"],
                "spearman_rho_confidence_filtered": filtered["rho"],
                "spearman_p_confidence_filtered": filtered["p_value"],
                "missing_count": int(values.isna().sum()),
            }
        )
    result = pd.DataFrame(rows, columns=ORGANISATION_COLUMNS)
    if not result.empty:
        result["_rank"] = result["spearman_rho"].abs()
        result = result.sort_values(["_rank", "missing_count"], ascending=[False, True], na_position="last").drop(columns=["_rank"]).reset_index(drop=True)
    return result


def confidence_feature_summary(feature_table: pd.DataFrame, feature_columns: list[str], min_n: int = 10) -> pd.DataFrame:
    rows = []
    for feature in feature_columns:
        stats = spearman_pair(feature_table, "confidence_score", feature, min_n=min_n)
        values = pd.to_numeric(feature_table[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "n": stats["n"],
                "spearman_rho": stats["rho"],
                "spearman_p": stats["p_value"],
                "missing_count": int(values.isna().sum()),
            }
        )
    result = pd.DataFrame(rows, columns=CONFIDENCE_COLUMNS)
    if not result.empty:
        result["_rank"] = result["spearman_rho"].abs()
        result = result.sort_values(["_rank", "missing_count"], ascending=[False, True], na_position="last").drop(columns=["_rank"]).reset_index(drop=True)
    return result


def spearman_pair(table: pd.DataFrame, x_column: str, y_column: str, min_n: int = 10) -> dict[str, float | int | None | bool | str]:
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
    }


def build_feature_audit_summary(
    feature_table: pd.DataFrame,
    feature_columns: list[str],
    excluded_columns: list[str],
    visibility: pd.DataFrame,
    organisation: pd.DataFrame,
    confidence: pd.DataFrame,
    min_n: int,
    min_confidence: int,
) -> dict[str, Any]:
    oop_row = organisation.loc[organisation["feature"].isin(["patch_oop", "automated_patch_oop"])].head(1)
    oop_result = None
    if not oop_row.empty:
        row = oop_row.iloc[0]
        oop_result = {
            "feature": str(row["feature"]),
            "n": int(row["n"]),
            "spearman_rho": none_if_nan(row["spearman_rho"]),
            "spearman_p": none_if_nan(row["spearman_p"]),
        }
    summary = {
        "mode": "expert_annotation_feature_audit",
        "audit": {
            "rows_in_matched_expert_annotations": int(len(feature_table)),
            "numeric_automated_features_considered": int(len(feature_columns)),
            "features_considered": feature_columns,
            "features_excluded": excluded_columns,
            "missingness_per_feature": {
                feature: int(pd.to_numeric(feature_table[feature], errors="coerce").isna().sum())
                for feature in feature_columns
            },
            "completed_manual_endpoints": {
                "striations_visible": int(feature_table["striations_visible"].notna().sum()),
                "organisation_score": int(feature_table["organisation_score"].notna().sum()),
                "confidence_score": int(feature_table["confidence_score"].notna().sum()),
            },
        },
        "top_visibility_features_by_abs_yes_minus_no": top_records(visibility, "abs_yes_minus_no"),
        "top_organisation_features_by_abs_spearman": top_records_with_abs(organisation, "spearman_rho"),
        "top_confidence_filtered_organisation_features_by_abs_spearman": top_records_with_abs(organisation, "spearman_rho_confidence_filtered"),
        "top_confidence_features_by_abs_spearman": top_records_with_abs(confidence, "spearman_rho"),
        "oop_specific_statement": {
            "patch_oop_result": oop_result,
            "statement": "Patch OOP alone previously showed near-zero correlation with expert organisation score and is not validated as a standalone organisation endpoint.",
        },
        "parameters": {"min_n": int(min_n), "min_confidence": int(min_confidence)},
        "interpretation_flags": [
            "exploratory_feature_audit_only",
            "no_feature_selection_for_production",
            "no_threshold_changes",
            "no_clinical_claims",
            "small_n",
            "single_reviewer_expert_annotations",
            "dominant_orientation_not_used_as_primary",
            "spacing_not_validated",
        ],
    }
    return json_safe(summary)


def top_records(table: pd.DataFrame, sort_column: str, n: int = 10) -> list[dict[str, Any]]:
    if table.empty or sort_column not in table.columns:
        return []
    return json_safe(table.head(n).to_dict("records"))


def top_records_with_abs(table: pd.DataFrame, column: str, n: int = 10) -> list[dict[str, Any]]:
    if table.empty or column not in table.columns:
        return []
    ranked = table.copy()
    ranked["_abs"] = pd.to_numeric(ranked[column], errors="coerce").abs()
    ranked = ranked.sort_values("_abs", ascending=False, na_position="last").drop(columns=["_abs"])
    return json_safe(ranked.head(n).to_dict("records"))


def none_if_nan(value: Any) -> Any:
    if value is None:
        return None
    try:
        if not np.isfinite(float(value)):
            return None
    except (TypeError, ValueError):
        return value
    return float(value)


def write_feature_audit_outputs(
    feature_table: pd.DataFrame,
    visibility: pd.DataFrame,
    organisation: pd.DataFrame,
    confidence: pd.DataFrame,
    summary: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    paths["feature_table"].parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(paths["feature_table"], index=False)
    visibility.to_csv(paths["visibility_summary"], index=False)
    organisation.to_csv(paths["organisation_summary"], index=False)
    confidence.to_csv(paths["confidence_summary"], index=False)
    paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    paths["summary_txt"].write_text(render_feature_audit_text(summary), encoding="utf-8")


def render_feature_audit_text(summary: dict[str, Any]) -> str:
    lines = [
        "Expert annotation feature audit",
        f"rows: {summary['audit']['rows_in_matched_expert_annotations']}",
        f"numeric_features_considered: {summary['audit']['numeric_automated_features_considered']}",
        "",
        "Top visibility-associated features:",
    ]
    for row in summary["top_visibility_features_by_abs_yes_minus_no"][:10]:
        lines.append(f"- {row.get('feature')}: yes_minus_no={row.get('yes_minus_no')}, n={row.get('n')}")
    lines.append("")
    lines.append("Top organisation-associated features:")
    for row in summary["top_organisation_features_by_abs_spearman"][:10]:
        lines.append(f"- {row.get('feature')}: rho={row.get('spearman_rho')}, n={row.get('n')}")
    lines.append("")
    lines.append("Top confidence-filtered organisation-associated features:")
    for row in summary["top_confidence_filtered_organisation_features_by_abs_spearman"][:10]:
        lines.append(f"- {row.get('feature')}: rho={row.get('spearman_rho_confidence_filtered')}, n={row.get('n_confidence_filtered')}")
    lines.append("")
    lines.append(summary["oop_specific_statement"]["statement"])
    lines.append("Exploratory audit only; no production feature selection or threshold changes.")
    return "\n".join(lines) + "\n"
