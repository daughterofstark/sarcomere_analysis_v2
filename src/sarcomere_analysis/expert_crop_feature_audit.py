from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import kruskal, spearmanr

from .config import output_dir
from .orientation import (
    axial_order_parameter,
    orientation_params,
    orientation_weights,
    radians_to_degrees,
    structure_tensor_orientation,
)
from .zdisc_annotation import json_safe


CROP_FEATURE_COLUMNS = [
    "crop_oop",
    "crop_mean_orientation_deg",
    "crop_orientation_coherence_mean",
    "crop_orientation_coherence_median",
    "crop_orientation_weight_sum",
    "crop_orientation_valid_pixel_count",
    "crop_orientation_valid_pixel_fraction",
    "crop_gradient_energy",
    "crop_intensity_mean",
    "crop_intensity_std",
    "crop_contrast_p2_p98",
    "crop_entropy",
    "crop_laplacian_variance",
]

FEATURE_TABLE_COLUMNS = [
    "annotation_id",
    "patch_filename",
    "image_id",
    "donor_id",
    "patch_id",
    "oop_bin",
    "striations_visible",
    "organisation_score",
    "confidence_score",
    "spacing_measurable",
    "expert_orientation_usable_primary",
    "automated_patch_oop",
    "automated_patch_orientation_deg",
    "crop_path",
    "crop_found",
    "crop_shape",
    *CROP_FEATURE_COLUMNS,
]

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
    "median_organisation_low",
    "median_organisation_medium",
    "median_organisation_high",
    "missing_count",
]

CONFIDENCE_COLUMNS = [
    "feature",
    "n",
    "spearman_rho",
    "spearman_p",
    "missing_count",
]


def default_expert_crop_feature_audit_paths(
    cfg: dict[str, Any],
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    out_dir = Path(output_directory) if output_directory else output_dir(cfg) / "validation" / "expert_crop_feature_audit"
    return {
        "feature_table": out_dir / "expert_crop_feature_table.csv",
        "visibility_summary": out_dir / "expert_crop_visibility_summary.csv",
        "organisation_summary": out_dir / "expert_crop_organisation_summary.csv",
        "confidence_summary": out_dir / "expert_crop_confidence_summary.csv",
        "summary_json": out_dir / "expert_crop_feature_audit_summary.json",
        "summary_txt": out_dir / "expert_crop_feature_audit_summary.txt",
    }


def audit_expert_crop_features(
    cfg: dict[str, Any],
    crop_dir: str | Path | None = None,
    internal_key: str | Path | None = None,
    matched_annotations: str | Path | None = None,
    output_directory: str | Path | None = None,
    min_n: int = 10,
    min_confidence: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    root = output_dir(cfg)
    crop_directory = Path(crop_dir) if crop_dir else root / "expert_annotation_pack" / "patches"
    key_path = Path(internal_key) if internal_key else root / "expert_annotation_pack" / "internal_blinding_key.csv"
    matched_path = (
        Path(matched_annotations)
        if matched_annotations
        else root / "validation" / "expert_annotation_validation" / "expert_annotation_validation_matched.csv"
    )

    key = pd.read_csv(key_path, dtype={"annotation_id": str, "patch_filename": str, "image_id": str, "donor_id": str, "patch_id": str})
    matched = pd.read_csv(
        matched_path,
        dtype={"annotation_id": str, "patch_filename": str, "image_id": str, "donor_id": str, "patch_id": str},
    )

    feature_table = build_expert_crop_feature_table(matched, key, crop_directory, cfg)
    visibility = visibility_summary(feature_table, CROP_FEATURE_COLUMNS)
    organisation = organisation_summary(feature_table, CROP_FEATURE_COLUMNS, min_n=min_n, min_confidence=min_confidence)
    confidence = confidence_summary(feature_table, CROP_FEATURE_COLUMNS, min_n=min_n)
    summary = build_crop_feature_audit_summary(
        feature_table,
        visibility,
        organisation,
        confidence,
        min_n=min_n,
        min_confidence=min_confidence,
    )
    paths = default_expert_crop_feature_audit_paths(cfg, output_directory)
    write_crop_feature_audit_outputs(feature_table, visibility, organisation, confidence, summary, paths)
    return feature_table, visibility, organisation, confidence, summary, paths


def build_expert_crop_feature_table(
    matched: pd.DataFrame,
    internal_key: pd.DataFrame,
    crop_dir: str | Path,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    annotations = matched.copy(deep=True)
    key = internal_key.copy(deep=True)
    for frame in [annotations, key]:
        for column in ["annotation_id", "patch_filename", "image_id", "donor_id", "patch_id"]:
            if column in frame.columns:
                frame[column] = frame[column].fillna("").astype(str)

    key_columns = [column for column in ["annotation_id", "patch_filename", "image_id", "donor_id", "patch_id"] if column in key.columns]
    joined = annotations.merge(
        key[key_columns].drop_duplicates("annotation_id"),
        on="annotation_id",
        how="left",
        suffixes=("", "_key"),
    )
    for column in ["patch_filename", "image_id", "donor_id", "patch_id"]:
        key_column = f"{column}_key"
        if key_column in joined.columns:
            if column in joined.columns:
                joined[column] = joined[column].where(joined[column].notna() & (joined[column].astype(str) != ""), joined[key_column])
            else:
                joined[column] = joined[key_column]
            joined = joined.drop(columns=[key_column])

    joined["organisation_score"] = pd.to_numeric(joined.get("organisation_score"), errors="coerce")
    joined["confidence_score"] = pd.to_numeric(joined.get("confidence_score"), errors="coerce")
    crop_root = Path(crop_dir)
    feature_rows: list[dict[str, Any]] = []
    for _, row in joined.iterrows():
        filename = str(row.get("patch_filename") or f"{row.get('annotation_id')}.png")
        crop_path = crop_root / filename
        features = compute_features_for_crop_path(crop_path, cfg)
        base = {column: row.get(column, np.nan) for column in FEATURE_TABLE_COLUMNS if column not in {"crop_path", "crop_found", "crop_shape", *CROP_FEATURE_COLUMNS}}
        feature_rows.append(
            {
                **base,
                "crop_path": str(crop_path),
                **features,
            }
        )

    return pd.DataFrame(feature_rows, columns=FEATURE_TABLE_COLUMNS)


def compute_features_for_crop_path(path: str | Path, cfg: dict[str, Any]) -> dict[str, Any]:
    crop_path = Path(path)
    if not crop_path.exists():
        return missing_crop_features(False)
    image = load_png_grayscale_float(crop_path)
    features = compute_crop_features(image, cfg)
    features["crop_found"] = True
    features["crop_shape"] = f"{image.shape[0]}x{image.shape[1]}"
    return features


def missing_crop_features(found: bool = False) -> dict[str, Any]:
    return {
        "crop_found": bool(found),
        "crop_shape": "",
        **{column: np.nan for column in CROP_FEATURE_COLUMNS},
    }


def load_png_grayscale_float(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32)
    if array.size == 0:
        return array.astype(np.float32)
    values = array / 255.0
    return np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)


def compute_crop_features(image: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        return {**missing_crop_features(True), "crop_shape": ""} | {column: np.nan for column in CROP_FEATURE_COLUMNS}

    params = orientation_params(cfg)
    orientation_map, coherence_map, energy_map = structure_tensor_orientation(values, params)
    weights = orientation_weights(energy_map, coherence_map, str(params["weight_mode"]))
    valid = np.isfinite(orientation_map) & np.isfinite(weights) & (weights > 0)
    oop, mean_rad, weight_sum, valid_pixels = axial_order_parameter(
        orientation_map,
        weights,
        valid,
        float(params["min_orientation_weight_sum"]),
        int(params["min_orientation_valid_pixels"]),
    )
    finite = values[np.isfinite(values)]
    p2, p98 = (np.nan, np.nan) if finite.size == 0 else np.percentile(finite, [2, 98])
    gy, gx = np.gradient(values)
    gradient_energy = float(np.mean(gx * gx + gy * gy)) if values.size else np.nan
    laplacian_variance = float(np.var(ndi.laplace(values))) if values.size else np.nan
    entropy = image_entropy(values)
    valid_fraction = float(valid_pixels / values.size) if values.size else np.nan
    coherence_valid = coherence_map[np.isfinite(coherence_map)]
    return {
        "crop_found": True,
        "crop_shape": f"{values.shape[0]}x{values.shape[1]}",
        "crop_oop": oop,
        "crop_mean_orientation_deg": radians_to_degrees(mean_rad),
        "crop_orientation_coherence_mean": float(np.mean(coherence_valid)) if coherence_valid.size else np.nan,
        "crop_orientation_coherence_median": float(np.median(coherence_valid)) if coherence_valid.size else np.nan,
        "crop_orientation_weight_sum": weight_sum,
        "crop_orientation_valid_pixel_count": int(valid_pixels),
        "crop_orientation_valid_pixel_fraction": valid_fraction,
        "crop_gradient_energy": gradient_energy,
        "crop_intensity_mean": float(np.mean(finite)) if finite.size else np.nan,
        "crop_intensity_std": float(np.std(finite)) if finite.size else np.nan,
        "crop_contrast_p2_p98": float(p98 - p2) if np.isfinite(p2) and np.isfinite(p98) else np.nan,
        "crop_entropy": entropy,
        "crop_laplacian_variance": laplacian_variance,
    }


def image_entropy(values: np.ndarray, bins: int = 64) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    hist, _ = np.histogram(np.clip(finite, 0.0, 1.0), bins=bins, range=(0.0, 1.0), density=False)
    probs = hist.astype(np.float64)
    total = probs.sum()
    if total <= 0:
        return float("nan")
    probs = probs[probs > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


def visibility_summary(table: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for feature in feature_columns:
        values = pd.to_numeric(table[feature], errors="coerce")
        medians = {level: median_for_group(table, values, "striations_visible", level) for level in ["yes", "unclear", "no"]}
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
                "p_value_or_blank": kruskal_visibility_p_value(table, feature),
                "missing_count": int(values.isna().sum()),
            }
        )
    result = pd.DataFrame(rows, columns=VISIBILITY_COLUMNS)
    if not result.empty:
        result = result.sort_values(["abs_yes_minus_no", "missing_count"], ascending=[False, True], na_position="last").reset_index(drop=True)
    return result


def organisation_summary(
    table: pd.DataFrame,
    feature_columns: list[str],
    min_n: int = 10,
    min_confidence: int = 3,
) -> pd.DataFrame:
    rows = []
    high_conf = table.loc[pd.to_numeric(table["confidence_score"], errors="coerce") >= int(min_confidence)].copy()
    for feature in feature_columns:
        full = spearman_pair(table, "organisation_score", feature, min_n=min_n)
        filtered = spearman_pair(high_conf, "organisation_score", feature, min_n=min_n)
        values = pd.to_numeric(table[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "n": full["n"],
                "spearman_rho": full["rho"],
                "spearman_p": full["p_value"],
                "n_confidence_filtered": filtered["n"],
                "spearman_rho_confidence_filtered": filtered["rho"],
                "spearman_p_confidence_filtered": filtered["p_value"],
                "median_organisation_low": median_for_collapsed_organisation(table, values, "low"),
                "median_organisation_medium": median_for_collapsed_organisation(table, values, "medium"),
                "median_organisation_high": median_for_collapsed_organisation(table, values, "high"),
                "missing_count": int(values.isna().sum()),
            }
        )
    result = pd.DataFrame(rows, columns=ORGANISATION_COLUMNS)
    if not result.empty:
        result["_rank"] = pd.to_numeric(result["spearman_rho"], errors="coerce").abs()
        result = result.sort_values(["_rank", "missing_count"], ascending=[False, True], na_position="last").drop(columns=["_rank"]).reset_index(drop=True)
    return result


def confidence_summary(table: pd.DataFrame, feature_columns: list[str], min_n: int = 10) -> pd.DataFrame:
    rows = []
    for feature in feature_columns:
        stats = spearman_pair(table, "confidence_score", feature, min_n=min_n)
        values = pd.to_numeric(table[feature], errors="coerce")
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
        result["_rank"] = pd.to_numeric(result["spearman_rho"], errors="coerce").abs()
        result = result.sort_values(["_rank", "missing_count"], ascending=[False, True], na_position="last").drop(columns=["_rank"]).reset_index(drop=True)
    return result


def median_for_group(table: pd.DataFrame, values: pd.Series, group_column: str, level: str) -> float | None:
    mask = table[group_column].fillna("").astype(str).str.lower() == str(level).lower()
    subset = values.loc[mask].dropna()
    return None if subset.empty else float(np.nanmedian(subset))


def median_for_collapsed_organisation(table: pd.DataFrame, values: pd.Series, group: str) -> float | None:
    scores = pd.to_numeric(table["organisation_score"], errors="coerce")
    if group == "low":
        mask = scores.isin([1, 2])
    elif group == "medium":
        mask = scores == 3
    elif group == "high":
        mask = scores.isin([4, 5])
    else:
        raise ValueError(f"Unknown organisation group: {group}")
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


def build_crop_feature_audit_summary(
    feature_table: pd.DataFrame,
    visibility: pd.DataFrame,
    organisation: pd.DataFrame,
    confidence: pd.DataFrame,
    min_n: int,
    min_confidence: int,
) -> dict[str, Any]:
    previous_patch_oop = spearman_pair(feature_table, "organisation_score", "automated_patch_oop", min_n=min_n)
    crop_oop = spearman_pair(feature_table, "organisation_score", "crop_oop", min_n=min_n)
    high_conf = feature_table.loc[pd.to_numeric(feature_table["confidence_score"], errors="coerce") >= int(min_confidence)].copy()
    crop_oop_conf = spearman_pair(high_conf, "organisation_score", "crop_oop", min_n=min_n)
    summary = {
        "mode": "expert_visible_crop_feature_audit",
        "audit": {
            "rows": int(len(feature_table)),
            "crop_pngs_found": int(feature_table["crop_found"].fillna(False).astype(bool).sum()),
            "crop_pngs_missing": int((~feature_table["crop_found"].fillna(False).astype(bool)).sum()),
            "crop_features_computed": int(len(CROP_FEATURE_COLUMNS)),
            "completed_manual_endpoints": {
                "striations_visible": int(feature_table["striations_visible"].notna().sum()),
                "organisation_score": int(feature_table["organisation_score"].notna().sum()),
                "confidence_score": int(feature_table["confidence_score"].notna().sum()),
            },
        },
        "previous_production_patch_oop_vs_organisation": previous_patch_oop,
        "crop_oop_vs_organisation": crop_oop,
        "crop_oop_vs_organisation_confidence_filtered": crop_oop_conf,
        "region_mismatch_assessment": region_mismatch_assessment(previous_patch_oop, crop_oop),
        "top_visibility_features_by_abs_yes_minus_no": top_records(visibility, "abs_yes_minus_no"),
        "top_organisation_features_by_abs_spearman": top_records_with_abs(organisation, "spearman_rho"),
        "top_confidence_filtered_organisation_features_by_abs_spearman": top_records_with_abs(organisation, "spearman_rho_confidence_filtered"),
        "top_confidence_features_by_abs_spearman": top_records_with_abs(confidence, "spearman_rho"),
        "parameters": {"min_n": int(min_n), "min_confidence": int(min_confidence)},
        "interpretation_flags": [
            "expert_visible_crop_region_alignment_audit",
            "exploratory_only",
            "no_production_algorithm_changes",
            "no_threshold_changes",
            "no_clinical_claims",
            "dominant_orientation_not_primary_due_annotation_ambiguity",
            "spacing_not_validated",
        ],
    }
    return json_safe(summary)


def region_mismatch_assessment(previous: dict[str, Any], crop: dict[str, Any]) -> dict[str, Any]:
    prev_rho = previous.get("rho")
    crop_rho = crop.get("rho")
    improved = None
    if prev_rho is not None and crop_rho is not None:
        improved = abs(float(crop_rho)) > abs(float(prev_rho))
    return {
        "previous_patch_oop_rho": prev_rho,
        "crop_oop_rho": crop_rho,
        "absolute_crop_rho_exceeds_absolute_patch_rho": improved,
        "statement": "This tests whether human/computational region mismatch explains weak validation; it does not change production outputs.",
    }


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


def write_crop_feature_audit_outputs(
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
    paths["summary_txt"].write_text(render_crop_feature_audit_text(summary), encoding="utf-8")


def render_crop_feature_audit_text(summary: dict[str, Any]) -> str:
    lines = [
        "Expert-visible crop feature audit",
        f"rows: {summary['audit']['rows']}",
        f"crop_pngs_found: {summary['audit']['crop_pngs_found']}",
        f"crop_pngs_missing: {summary['audit']['crop_pngs_missing']}",
        "",
        f"previous_production_patch_oop_vs_organisation: {summary['previous_production_patch_oop_vs_organisation']}",
        f"crop_oop_vs_organisation: {summary['crop_oop_vs_organisation']}",
        f"crop_oop_vs_organisation_confidence_filtered: {summary['crop_oop_vs_organisation_confidence_filtered']}",
        f"region_mismatch_assessment: {summary['region_mismatch_assessment']}",
        "",
        "Top visibility-associated crop features:",
    ]
    for row in summary["top_visibility_features_by_abs_yes_minus_no"][:10]:
        lines.append(f"- {row.get('feature')}: yes_minus_no={row.get('yes_minus_no')}, n={row.get('n')}")
    lines.append("")
    lines.append("Top organisation-associated crop features:")
    for row in summary["top_organisation_features_by_abs_spearman"][:10]:
        lines.append(f"- {row.get('feature')}: rho={row.get('spearman_rho')}, n={row.get('n')}")
    lines.append("")
    lines.append("Top confidence-filtered organisation-associated crop features:")
    for row in summary["top_confidence_filtered_organisation_features_by_abs_spearman"][:10]:
        lines.append(f"- {row.get('feature')}: rho={row.get('spearman_rho_confidence_filtered')}, n={row.get('n_confidence_filtered')}")
    lines.append("")
    lines.append("Exploratory region-alignment audit only; no production algorithm, threshold, or output changes.")
    return "\n".join(lines) + "\n"
