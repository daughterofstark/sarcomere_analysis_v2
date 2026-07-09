from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SENSITIVITY_COLUMNS = [
    "variant_id",
    "min_confidence",
    "band_min_px",
    "band_max_px",
    "peak_rule",
    "accepted_patch_count",
    "accepted_image_count",
    "images_with_zero_spacing",
    "median_valid_patches_per_image",
    "max_valid_patches_per_image",
    "accepted_lag_px_distribution",
    "accepted_spacing_um_distribution",
    "median_peak_confidence",
    "min_peak_confidence",
    "max_peak_confidence",
    "artefact_risk_flag",
    "interpretation_class",
]

DEFAULT_CONFIDENCE_GRID = [0.10, 0.12, 0.14, 0.15, 0.18, 0.20]
DEFAULT_BAND_PADDING_GRID = ["current", "min_minus_1", "max_plus_1", "both_plus_1"]
DEFAULT_PEAK_RULES = ["in_band_best_only", "current_selected_if_available", "global_best_allowed"]


def read_candidate_table(path: str | Path) -> pd.DataFrame:
    candidate_path = Path(path)
    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Candidate diagnostics table not found: {candidate_path}. "
            "Run: ../sarcgraph-env/bin/python scripts/diagnose_spacing_candidates.py --config configs/default.yaml --all --compare-main-table"
        )
    return pd.read_csv(candidate_path, dtype={"image_id": str, "donor_id": str, "patch_id": str})


def build_spacing_sensitivity_report(
    candidates: pd.DataFrame,
    confidence_grid: list[float] | None = None,
    band_padding_grid: list[str] | None = None,
    peak_rules: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    table = candidates.copy(deep=True)
    confidence_grid = confidence_grid or DEFAULT_CONFIDENCE_GRID
    band_padding_grid = band_padding_grid or DEFAULT_BAND_PADDING_GRID
    peak_rules = peak_rules or DEFAULT_PEAK_RULES
    rows = []
    variant_index = 0
    for padding in band_padding_grid:
        band_min, band_max = padded_band(table, padding)
        for rule in peak_rules:
            for confidence in confidence_grid:
                variant_index += 1
                rows.append(evaluate_variant(table, variant_index, float(confidence), band_min, band_max, rule))
    variants = pd.DataFrame(rows)
    variants = stabilize_sensitivity_columns(variants)
    summary = summarize_sensitivity_variants(variants, table)
    return variants, summary


def padded_band(candidates: pd.DataFrame, padding: str) -> tuple[float, float]:
    current_min = float(pd.to_numeric(candidates["expected_min_lag_px"], errors="coerce").median())
    current_max = float(pd.to_numeric(candidates["expected_max_lag_px"], errors="coerce").median())
    if padding == "current":
        return current_min, current_max
    if padding == "min_minus_1":
        return max(10.0, current_min - 1.0), current_max
    if padding == "max_plus_1":
        return current_min, min(20.0, current_max + 1.0)
    if padding == "both_plus_1":
        return max(10.0, current_min - 1.0), min(20.0, current_max + 1.0)
    raise ValueError(f"Unknown band padding variant: {padding}")


def evaluate_variant(
    candidates: pd.DataFrame,
    variant_index: int,
    min_confidence: float,
    band_min_px: float,
    band_max_px: float,
    peak_rule: str,
) -> dict[str, object]:
    accepted = accepted_candidates(candidates, min_confidence, band_min_px, band_max_px, peak_rule)
    accepted_count = int(accepted.sum())
    accepted_table = candidates.loc[accepted].copy()
    per_image = accepted_table.groupby("image_id").size() if accepted_count else pd.Series(dtype=int)
    total_images = int(candidates["image_id"].nunique()) if "image_id" in candidates.columns else 0
    accepted_image_count = int(per_image.size)
    lag_values = accepted_lag_values(accepted_table, peak_rule)
    spacing_values = accepted_spacing_values(accepted_table, peak_rule)
    confidence_values = accepted_confidence_values(accepted_table, peak_rule)
    risk_flag = artefact_risk_flag(candidates, accepted_table, lag_values, confidence_values, peak_rule, min_confidence, total_images)
    return {
        "variant_id": f"v{variant_index:03d}",
        "min_confidence": min_confidence,
        "band_min_px": float(band_min_px),
        "band_max_px": float(band_max_px),
        "peak_rule": peak_rule,
        "accepted_patch_count": accepted_count,
        "accepted_image_count": accepted_image_count,
        "images_with_zero_spacing": int(max(total_images - accepted_image_count, 0)),
        "median_valid_patches_per_image": float(per_image.median()) if not per_image.empty else 0.0,
        "max_valid_patches_per_image": int(per_image.max()) if not per_image.empty else 0,
        "accepted_lag_px_distribution": numeric_value_counts(lag_values),
        "accepted_spacing_um_distribution": numeric_value_counts(spacing_values),
        "median_peak_confidence": finite_stat(confidence_values, np.median),
        "min_peak_confidence": finite_stat(confidence_values, np.min),
        "max_peak_confidence": finite_stat(confidence_values, np.max),
        "artefact_risk_flag": risk_flag,
        "interpretation_class": interpretation_class(risk_flag, accepted_image_count, total_images, accepted_count),
    }


def accepted_candidates(
    candidates: pd.DataFrame,
    min_confidence: float,
    band_min_px: float,
    band_max_px: float,
    peak_rule: str,
) -> pd.Series:
    index = candidates.index
    if peak_rule == "in_band_best_only":
        lag = pd.to_numeric(candidates["best_in_band_lag_px"], errors="coerce")
        peak = pd.to_numeric(candidates["best_in_band_peak_value"], errors="coerce")
        confidence = pd.to_numeric(candidates["peak_confidence"], errors="coerce")
        return lag.notna() & peak.notna() & within_band(lag, band_min_px, band_max_px) & (confidence >= min_confidence)
    if peak_rule == "current_selected_if_available":
        lag = pd.to_numeric(candidates["selected_lag_px"], errors="coerce")
        peak = pd.to_numeric(candidates["selected_peak_value"], errors="coerce")
        confidence = pd.to_numeric(candidates["peak_confidence"], errors="coerce")
        return lag.notna() & peak.notna() & within_band(lag, band_min_px, band_max_px) & (confidence >= min_confidence)
    if peak_rule == "global_best_allowed":
        lag = pd.to_numeric(candidates["best_global_lag_px"], errors="coerce")
        peak = pd.to_numeric(candidates["best_global_peak_value"], errors="coerce")
        confidence = pd.to_numeric(candidates["peak_confidence"], errors="coerce").fillna(0.0)
        return lag.notna() & peak.notna() & (confidence >= min_confidence)
    raise ValueError(f"Unknown peak rule: {peak_rule}")


def within_band(values: pd.Series, band_min_px: float, band_max_px: float) -> pd.Series:
    return (values >= np.ceil(band_min_px)) & (values <= np.floor(band_max_px))


def accepted_lag_values(accepted_table: pd.DataFrame, peak_rule: str) -> pd.Series:
    if peak_rule == "global_best_allowed":
        return pd.to_numeric(accepted_table["best_global_lag_px"], errors="coerce")
    if peak_rule == "in_band_best_only":
        return pd.to_numeric(accepted_table["best_in_band_lag_px"], errors="coerce")
    return pd.to_numeric(accepted_table["selected_lag_px"], errors="coerce")


def accepted_spacing_values(accepted_table: pd.DataFrame, peak_rule: str) -> pd.Series:
    lag = accepted_lag_values(accepted_table, peak_rule)
    selected_lag = pd.to_numeric(accepted_table.get("selected_lag_px", pd.Series(index=accepted_table.index, dtype=float)), errors="coerce")
    selected_um = pd.to_numeric(accepted_table.get("selected_lag_um", pd.Series(index=accepted_table.index, dtype=float)), errors="coerce")
    pixel_size = (selected_um / selected_lag).replace([np.inf, -np.inf], np.nan).dropna()
    if pixel_size.empty:
        return pd.Series(dtype=float)
    return lag * float(pixel_size.median())


def accepted_confidence_values(accepted_table: pd.DataFrame, peak_rule: str) -> pd.Series:
    _ = peak_rule
    return pd.to_numeric(accepted_table.get("peak_confidence", pd.Series(index=accepted_table.index, dtype=float)), errors="coerce")


def artefact_risk_flag(
    candidates: pd.DataFrame,
    accepted_table: pd.DataFrame,
    lag_values: pd.Series,
    confidence_values: pd.Series,
    peak_rule: str,
    min_confidence: float,
    total_images: int,
) -> str:
    accepted_count = int(len(accepted_table))
    if accepted_count == 0:
        return "uninformative_low_yield"
    if peak_rule == "global_best_allowed":
        outside = best_global_outside_current_band(accepted_table)
        if float(outside.mean()) >= 0.25:
            return "high_artefact_risk"
    if repeated_lag_fraction(lag_values) >= 0.75 and accepted_count >= 10:
        return "high_artefact_risk"
    if min_confidence < 0.15 and low_confidence_fraction(confidence_values, 0.15) >= 0.5:
        return "high_artefact_risk"
    accepted_images = int(accepted_table["image_id"].nunique()) if "image_id" in accepted_table.columns else 0
    if total_images > 0 and accepted_images / total_images < 0.1:
        return "low_coverage"
    return "low_risk_for_review"


def interpretation_class(risk_flag: str, accepted_image_count: int, total_images: int, accepted_patch_count: int) -> str:
    if risk_flag == "high_artefact_risk":
        return "high_artefact_risk"
    if accepted_patch_count == 0 or accepted_image_count == 0:
        return "uninformative_low_yield"
    if total_images > 0 and accepted_image_count / total_images < 0.1:
        return "conservative_low_yield"
    return "plausible_for_review"


def best_global_outside_current_band(table: pd.DataFrame) -> pd.Series:
    lag = pd.to_numeric(table["best_global_lag_px"], errors="coerce")
    min_lag = pd.to_numeric(table["expected_min_lag_px"], errors="coerce")
    max_lag = pd.to_numeric(table["expected_max_lag_px"], errors="coerce")
    return lag.notna() & ((lag < np.ceil(min_lag)) | (lag > np.floor(max_lag)))


def repeated_lag_fraction(lag_values: pd.Series) -> float:
    values = pd.to_numeric(lag_values, errors="coerce").dropna()
    if values.empty:
        return 0.0
    counts = values.round(6).value_counts()
    return float(counts.iloc[0] / len(values))


def low_confidence_fraction(confidence_values: pd.Series, threshold: float) -> float:
    values = pd.to_numeric(confidence_values, errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float((values < threshold).mean())


def summarize_sensitivity_variants(variants: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, Any]:
    low_risk = variants.loc[variants["interpretation_class"].isin(["plausible_for_review", "conservative_low_yield"])].copy()
    low_risk = low_risk.sort_values(["accepted_image_count", "accepted_patch_count"], ascending=False)
    summary = {
        "candidate_rows": int(len(candidates)),
        "variant_count": int(len(variants)),
        "interpretation_class_counts": value_counts(variants["interpretation_class"]),
        "artefact_risk_flag_counts": value_counts(variants["artefact_risk_flag"]),
        "best_low_risk_variants_by_accepted_image_count": low_risk.head(10).to_dict(orient="records"),
        "max_accepted_image_count": int(variants["accepted_image_count"].max()) if not variants.empty else 0,
        "max_accepted_patch_count": int(variants["accepted_patch_count"].max()) if not variants.empty else 0,
        "caution": "This report does not recommend a final threshold. It is evidence for future algorithm/threshold review only.",
    }
    return json_safe(summary)


def write_sensitivity_outputs(variants: pd.DataFrame, summary: dict[str, Any], output_directory: str | Path) -> dict[str, Path]:
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "variants": out_dir / "spacing_sensitivity_variants.csv",
        "summary_json": out_dir / "spacing_sensitivity_summary.json",
        "summary_txt": out_dir / "spacing_sensitivity_summary.txt",
    }
    stabilize_sensitivity_columns(variants).to_csv(paths["variants"], index=False)
    with paths["summary_json"].open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2)
        handle.write("\n")
    paths["summary_txt"].write_text(format_sensitivity_summary(summary), encoding="utf-8")
    return paths


def stabilize_sensitivity_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in SENSITIVITY_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
    return result[SENSITIVITY_COLUMNS]


def format_sensitivity_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Spacing Sensitivity Report",
        f"candidate_rows: {summary['candidate_rows']}",
        f"variant_count: {summary['variant_count']}",
        f"interpretation_class_counts: {summary['interpretation_class_counts']}",
        f"artefact_risk_flag_counts: {summary['artefact_risk_flag_counts']}",
        f"max_accepted_image_count: {summary['max_accepted_image_count']}",
        f"max_accepted_patch_count: {summary['max_accepted_patch_count']}",
        f"caution: {summary['caution']}",
    ]
    return "\n".join(lines) + "\n"


def finite_stat(values: pd.Series, fn) -> float:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(fn(array))


def numeric_value_counts(values: pd.Series, limit: int = 20, decimals: int = 6) -> str:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return ""
    counts = series.round(decimals).value_counts().sort_index().head(limit)
    return "; ".join(f"{index:g}:{int(value)}" for index, value in counts.items())


def value_counts(series: pd.Series) -> dict[str, int]:
    counts = series.fillna("<NA>").astype(str).value_counts()
    return {str(key): int(value) for key, value in counts.items()}


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
