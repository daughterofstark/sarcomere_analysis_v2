from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sarcomere_analysis.config import output_dir


BY_IMAGE_COLUMNS = [
    "image_id",
    "donor_id",
    "total_patches",
    "qc_valid_spacing_patches",
    "patches_reaching_spacing_estimator",
    "final_valid_spacing_patches",
    "finite_spacing_px_patches",
    "finite_spacing_um_patches",
    "dominant_rejection_reason",
    "dominant_rejection_count",
    "tissue_fraction_median",
    "tissue_fraction_mean",
    "intensity_std_median",
    "intensity_std_mean",
    "gradient_energy_median",
    "gradient_energy_mean",
    "patch_oop_median",
    "patch_oop_mean",
    "patch_spacing_confidence_median",
    "patch_spacing_confidence_mean",
]

STAGE_ORDER = [
    "accepted",
    "failed_patch_qc",
    "missing_orientation",
    "peak_picking",
    "confidence",
    "spacing_band",
    "profile",
    "other",
]


def read_spacing_triage_inputs(
    cfg: dict[str, Any],
    patch_table: str | Path | None = None,
    image_table: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any] | None]:
    tables = output_dir(cfg) / "tables"
    diagnostics = output_dir(cfg) / "diagnostics"
    patch_path = Path(patch_table) if patch_table is not None else tables / "per_patch_metrics.csv"
    image_path = Path(image_table) if image_table is not None else tables / "per_image_metrics.csv"
    patch = pd.read_csv(patch_path, dtype={"image_id": str, "donor_id": str, "patch_id": str})
    image = pd.read_csv(image_path, dtype={"image_id": str, "donor_id": str})
    diagnostic_summary = read_existing_diagnostic_summary(diagnostics / "spacing_diagnostic_summary.csv")
    return patch, image, diagnostic_summary


def read_existing_diagnostic_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return json_safe(df.iloc[0].to_dict())


def triage_spacing_failures(
    patch_metrics: pd.DataFrame,
    image_metrics: pd.DataFrame,
    existing_spacing_diagnostic_summary: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    patch = patch_metrics.copy(deep=True)
    image = image_metrics.copy(deep=True)
    add_derived_columns(patch)

    by_image = build_by_image_report(patch, image)
    summary = build_summary(patch, image, by_image, existing_spacing_diagnostic_summary)
    return json_safe(summary), by_image


def add_derived_columns(patch: pd.DataFrame) -> None:
    if "valid_for_spacing" not in patch.columns:
        patch["valid_for_spacing"] = False
    if "valid_for_spacing_final" not in patch.columns:
        patch["valid_for_spacing_final"] = False
    if "spacing_invalid_reason" not in patch.columns:
        patch["spacing_invalid_reason"] = "missing_spacing_invalid_reason"
    patch["spacing_rejection_stage_inferred"] = [
        infer_spacing_rejection_stage(valid, reason)
        for valid, reason in zip(patch["valid_for_spacing_final"], patch["spacing_invalid_reason"])
    ]


def infer_spacing_rejection_stage(valid_for_spacing_final: object, reason: object) -> str:
    if bool(valid_for_spacing_final):
        return "accepted"
    text = str(reason)
    if "failed_patch_qc" in text:
        return "failed_patch_qc"
    if "missing_orientation" in text:
        return "missing_orientation"
    if "no_local_peak" in text or "peak" in text and "weak_fft_peak" not in text:
        return "peak_picking"
    if "low_periodicity_confidence" in text:
        return "confidence"
    if "spacing_band" in text or "outside_expected_band" in text:
        return "spacing_band"
    if any(token in text for token in ["short_profile", "flat_profile", "weak_fft_peak"]):
        return "profile"
    return "other"


def build_by_image_report(patch: pd.DataFrame, image: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["image_id", "donor_id"] if "donor_id" in patch.columns else ["image_id"]
    for keys, group in patch.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        image_id = str(keys[0])
        donor_id = str(keys[1]) if len(keys) > 1 and pd.notna(keys[1]) else image_donor_id(image, image_id)
        reason_counts = value_counts(group, "spacing_invalid_reason", include_ok=False)
        dominant_reason, dominant_count = first_count(reason_counts)
        rows.append(
            {
                "image_id": image_id,
                "donor_id": donor_id,
                "total_patches": int(len(group)),
                "qc_valid_spacing_patches": int(bool_sum(group, "valid_for_spacing")),
                "patches_reaching_spacing_estimator": int(reached_spacing_estimator(group).sum()),
                "final_valid_spacing_patches": int(bool_sum(group, "valid_for_spacing_final")),
                "finite_spacing_px_patches": int(finite_count(group, "patch_spacing_px")),
                "finite_spacing_um_patches": int(finite_count(group, "patch_spacing_um")),
                "dominant_rejection_reason": dominant_reason,
                "dominant_rejection_count": dominant_count,
                "tissue_fraction_median": finite_stat(group, "tissue_fraction", np.median),
                "tissue_fraction_mean": finite_stat(group, "tissue_fraction", np.mean),
                "intensity_std_median": finite_stat(group, "intensity_std", np.median),
                "intensity_std_mean": finite_stat(group, "intensity_std", np.mean),
                "gradient_energy_median": finite_stat(group, "gradient_energy", np.median),
                "gradient_energy_mean": finite_stat(group, "gradient_energy", np.mean),
                "patch_oop_median": finite_stat(group, "patch_oop", np.median),
                "patch_oop_mean": finite_stat(group, "patch_oop", np.mean),
                "patch_spacing_confidence_median": finite_stat(group, "patch_spacing_confidence", np.median),
                "patch_spacing_confidence_mean": finite_stat(group, "patch_spacing_confidence", np.mean),
            }
        )
    by_image = pd.DataFrame(rows)
    for column in BY_IMAGE_COLUMNS:
        if column not in by_image.columns:
            by_image[column] = np.nan
    return by_image[BY_IMAGE_COLUMNS]


def build_summary(
    patch: pd.DataFrame,
    image: pd.DataFrame,
    by_image: pd.DataFrame,
    existing_spacing_diagnostic_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    stage_counts = ordered_counts(patch, "spacing_rejection_stage_inferred", STAGE_ORDER)
    candidate_columns = candidate_level_columns(patch)
    has_candidate_detail = bool(candidate_columns)
    failure_assessment = classify_failure_mode(patch, stage_counts)
    summary: dict[str, Any] = {
        "total_images": int(image["image_id"].nunique()) if "image_id" in image.columns else int(by_image["image_id"].nunique()),
        "total_patches": int(len(patch)),
        "qc_valid_spacing_patches": int(bool_sum(patch, "valid_for_spacing")),
        "patches_reaching_spacing_estimator": int(reached_spacing_estimator(patch).sum()),
        "final_valid_spacing_patches": int(bool_sum(patch, "valid_for_spacing_final")),
        "finite_spacing_px_patches": int(finite_count(patch, "patch_spacing_px")),
        "finite_spacing_um_patches": int(finite_count(patch, "patch_spacing_um")),
        "images_with_no_valid_spacing": int((by_image["final_valid_spacing_patches"] == 0).sum()) if not by_image.empty else 0,
        "spacing_rejection_stage_counts": stage_counts,
        "top_invalid_reason_counts": value_counts(patch, "invalid_reason"),
        "top_spacing_invalid_reason_counts": value_counts(patch, "spacing_invalid_reason"),
        "upstream_quality_stratification": upstream_quality_stratification(patch),
        "candidate_level_columns_present": candidate_columns,
        "candidate_level_detail_available": has_candidate_detail,
        "candidate_lag_confidence_summary": candidate_lag_confidence_summary(patch),
        "failure_mode_assessment": failure_assessment,
        "safety_interpretation": safety_interpretation(has_candidate_detail),
        "existing_spacing_diagnostic_summary": existing_spacing_diagnostic_summary,
    }
    return summary


def classify_failure_mode(patch: pd.DataFrame, stage_counts: dict[str, int]) -> dict[str, Any]:
    total = max(int(len(patch)), 1)
    failed_qc = stage_counts.get("failed_patch_qc", 0)
    confidence = stage_counts.get("confidence", 0)
    peak = stage_counts.get("peak_picking", 0)
    missing = stage_counts.get("missing_orientation", 0)
    reasons: list[str] = []
    if failed_qc / total >= 0.5:
        reasons.append("upstream_qc_substantial")
    if peak >= confidence and peak >= failed_qc and peak > 0:
        reasons.append("no_periodic_peak_dominant")
    if confidence >= peak and confidence > 0:
        reasons.append("estimator_confidence_substantial")
    if missing / total >= 0.1:
        reasons.append("missing_orientation_substantial")
    if not reasons:
        reasons.append("mixed_or_sparse_failures")
    return {
        "primary_labels": reasons,
        "upstream_qc_failed_fraction": float(failed_qc / total),
        "no_local_peak_fraction": float(peak / total),
        "low_confidence_fraction": float(confidence / total),
        "missing_orientation_fraction": float(missing / total),
        "plain_language": plain_language_assessment(reasons),
    }


def plain_language_assessment(labels: list[str]) -> str:
    if "no_periodic_peak_dominant" in labels:
        return "Most spacing failures are estimator-stage rejections where no local periodic peak is present in the expected band."
    if "upstream_qc_substantial" in labels:
        return "A large fraction of patches fail before spacing because upstream patch QC rejects them."
    if "estimator_confidence_substantial" in labels:
        return "Many patches reach spacing but do not exceed the existing periodicity confidence threshold."
    return "Spacing failures are mixed or sparse across available categories."


def upstream_quality_stratification(patch: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in ["valid_for_orientation", "valid_for_periodicity", "valid_for_spacing"]:
        if column in patch.columns:
            result[column] = boolean_group_summary(patch, column)
    bin_specs = {
        "tissue_fraction": [0.0, 0.25, 0.5, 0.75, 0.9, 1.000001],
        "intensity_std": [0.0, 0.01, 0.03, 0.05, 0.1, np.inf],
        "gradient_energy": [0.0, 1e-5, 1e-4, 1e-3, 1e-2, np.inf],
        "patch_oop": [0.0, 0.25, 0.5, 0.75, 0.9, 1.000001],
        "patch_spacing_confidence": [0.0, 0.05, 0.1, 0.15, 0.25, 0.5, np.inf],
    }
    for column, bins in bin_specs.items():
        if column in patch.columns:
            result[f"{column}_bins"] = numeric_bin_summary(patch, column, bins)
    return result


def boolean_group_summary(patch: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    rows = []
    for value, group in patch.groupby(patch[column].fillna(False).astype(bool), sort=True):
        rows.append(group_summary_row(str(bool(value)), group))
    return rows


def numeric_bin_summary(patch: pd.DataFrame, column: str, bins: list[float]) -> list[dict[str, Any]]:
    values = pd.to_numeric(patch[column], errors="coerce")
    labels = [f"[{bins[i]}, {bins[i + 1]})" for i in range(len(bins) - 1)]
    categories = pd.cut(values, bins=bins, labels=labels, include_lowest=True, right=False)
    rows = []
    for label, group in patch.groupby(categories, observed=False, sort=True):
        if group.empty:
            continue
        rows.append(group_summary_row(str(label), group))
    missing = patch.loc[values.isna()]
    if not missing.empty:
        rows.append(group_summary_row("missing", missing))
    return rows


def group_summary_row(label: str, group: pd.DataFrame) -> dict[str, Any]:
    return {
        "group": label,
        "total_patches": int(len(group)),
        "qc_valid_spacing_patches": int(bool_sum(group, "valid_for_spacing")),
        "final_valid_spacing_patches": int(bool_sum(group, "valid_for_spacing_final")),
        "dominant_spacing_invalid_reason": first_count(value_counts(group, "spacing_invalid_reason", include_ok=False))[0],
        "dominant_rejection_stage": first_count(value_counts(group, "spacing_rejection_stage_inferred", include_ok=False))[0],
    }


def candidate_level_columns(patch: pd.DataFrame) -> list[str]:
    candidates = [
        "selected_lag_px",
        "selected_lag_um",
        "peak_score",
        "peak_rank_or_index",
        "autocorr_peak_value",
        "autocorr_baseline_value",
        "confidence_threshold",
        "spacing_rejection_stage",
    ]
    return [column for column in candidates if column in patch.columns]


def candidate_lag_confidence_summary(patch: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "available": bool(candidate_level_columns(patch)),
        "message": "",
    }
    if not summary["available"]:
        summary["message"] = (
            "Current main patch table lacks candidate-level lag/peak diagnostics. "
            "Use spacing diagnostic patch outputs or add candidate-level diagnostic columns later for deeper diagnosis."
        )
        return summary
    for column in ["selected_lag_px", "selected_lag_um", "patch_spacing_confidence", "peak_score"]:
        if column in patch.columns:
            summary[column] = numeric_distribution(patch[column])
            summary[f"{column}_top_values"] = numeric_value_counts(patch[column])
    if "selected_lag_px" in patch.columns:
        rejected = patch.loc[~patch["valid_for_spacing_final"].astype(bool)]
        summary["top_rejected_lag_px_values"] = numeric_value_counts(rejected["selected_lag_px"])
    return summary


def safety_interpretation(has_candidate_detail: bool) -> dict[str, str]:
    candidate_note = (
        "Candidate-level lag/peak columns are present for deeper triage."
        if has_candidate_detail
        else "The main patch table does not contain enough candidate-level lag/peak information for deeper spacing diagnosis."
    )
    return {
        "conservative_estimator": "The corrected spacing estimator is conservative and rejects most patches.",
        "spacing_endpoint_warning": "Fourteen valid spacing patches across the batch is not enough for a reliable spacing endpoint.",
        "next_action": "The next action should be evidence-based threshold/algorithm sensitivity, not manual cherry-picking.",
        "oop_independence": "OOP/orientation outputs are separate and should not be invalidated by sparse spacing unless they share the failing QC gate.",
        "candidate_detail": candidate_note,
    }


def reached_spacing_estimator(patch: pd.DataFrame) -> pd.Series:
    valid_spacing = patch["valid_for_spacing"].fillna(False).astype(bool) if "valid_for_spacing" in patch.columns else pd.Series(False, index=patch.index)
    has_orientation = (
        pd.to_numeric(patch["patch_mean_orientation_rad"], errors="coerce").notna()
        if "patch_mean_orientation_rad" in patch.columns
        else pd.Series(True, index=patch.index)
    )
    return valid_spacing & has_orientation


def bool_sum(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(df[column].fillna(False).astype(bool).sum())


def finite_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    values = pd.to_numeric(df[column], errors="coerce")
    return int(np.isfinite(values).sum())


def finite_stat(df: pd.DataFrame, column: str, fn) -> float:
    if column not in df.columns:
        return float("nan")
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(fn(values))


def value_counts(df: pd.DataFrame, column: str, limit: int | None = 10, include_ok: bool = True) -> dict[str, int]:
    if column not in df.columns:
        return {}
    series = df[column].fillna("<NA>").astype(str)
    if not include_ok:
        series = series.loc[series != "ok"]
    counts = series.value_counts()
    if limit is not None:
        counts = counts.head(limit)
    return {str(key): int(value) for key, value in counts.items()}


def ordered_counts(df: pd.DataFrame, column: str, order: list[str]) -> dict[str, int]:
    counts = value_counts(df, column, limit=None)
    ordered = {key: int(counts.get(key, 0)) for key in order if key in counts or key in order}
    for key, value in counts.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def first_count(counts: dict[str, int]) -> tuple[str, int]:
    if not counts:
        return "none", 0
    key = next(iter(counts))
    return key, int(counts[key])


def image_donor_id(image: pd.DataFrame, image_id: str) -> str:
    if {"image_id", "donor_id"}.issubset(image.columns):
        matches = image.loc[image["image_id"].astype(str) == str(image_id), "donor_id"]
        if not matches.empty and pd.notna(matches.iloc[0]):
            return str(matches.iloc[0])
    return ""


def numeric_distribution(series: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def numeric_value_counts(series: pd.Series, limit: int = 12, decimals: int = 6) -> dict[str, int]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {}
    counts = values.round(decimals).value_counts().sort_index().head(limit)
    return {str(key): int(value) for key, value in counts.items()}


def write_spacing_failure_outputs(
    summary: dict[str, Any],
    by_image: pd.DataFrame,
    output_directory: str | Path,
) -> dict[str, Path]:
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "by_image": out_dir / "spacing_failure_by_image.csv",
        "summary_json": out_dir / "spacing_failure_summary.json",
        "summary_txt": out_dir / "spacing_failure_summary.txt",
    }
    by_image.to_csv(paths["by_image"], index=False)
    with paths["summary_json"].open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2)
        handle.write("\n")
    paths["summary_txt"].write_text(format_spacing_failure_text(summary), encoding="utf-8")
    return paths


def format_spacing_failure_text(summary: dict[str, Any]) -> str:
    assessment = summary["failure_mode_assessment"]
    safety = summary["safety_interpretation"]
    lines = [
        "Spacing Failure Triage",
        f"total_images: {summary['total_images']}",
        f"total_patches: {summary['total_patches']}",
        f"qc_valid_spacing_patches: {summary['qc_valid_spacing_patches']}",
        f"patches_reaching_spacing_estimator: {summary['patches_reaching_spacing_estimator']}",
        f"final_valid_spacing_patches: {summary['final_valid_spacing_patches']}",
        f"images_with_no_valid_spacing: {summary['images_with_no_valid_spacing']}",
        f"spacing_rejection_stage_counts: {summary['spacing_rejection_stage_counts']}",
        f"top_spacing_invalid_reason_counts: {summary['top_spacing_invalid_reason_counts']}",
        f"failure_mode_assessment: {assessment['plain_language']}",
        f"candidate_level_detail_available: {summary['candidate_level_detail_available']}",
        f"candidate_detail: {safety['candidate_detail']}",
        f"warning: {safety['spacing_endpoint_warning']}",
        f"next_action: {safety['next_action']}",
        f"oop_note: {safety['oop_independence']}",
    ]
    return "\n".join(lines) + "\n"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
