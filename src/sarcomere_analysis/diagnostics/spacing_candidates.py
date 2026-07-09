from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sarcomere_analysis.config import manifest_csv_path, output_dir
from sarcomere_analysis.io import build_manifest, load_tiff
from sarcomere_analysis.masking import compute_tissue_mask
from sarcomere_analysis.orientation import compute_orientation_analysis
from sarcomere_analysis.preprocessing import preprocess_image
from sarcomere_analysis.qc import compute_patch_qc
from sarcomere_analysis.spacing.autocorrelation import (
    local_maxima_lags,
    normalized_autocorrelation,
    prepare_autocorrelation_profile,
    select_autocorrelation_peak,
)
from sarcomere_analysis.spacing.base import estimate_patch_spacing, spacing_band_px, spacing_params, px_to_um


SPACING_CANDIDATE_COLUMNS = [
    "image_id",
    "donor_id",
    "patch_id",
    "y0",
    "x0",
    "y1",
    "x1",
    "valid_for_spacing_qc",
    "final_valid_for_spacing",
    "final_invalid_reason",
    "expected_min_lag_px",
    "expected_max_lag_px",
    "selected_lag_px",
    "selected_lag_um",
    "selected_peak_value",
    "baseline_value",
    "peak_prominence",
    "peak_confidence",
    "n_local_peaks_total",
    "n_local_peaks_in_band",
    "best_in_band_lag_px",
    "best_in_band_peak_value",
    "best_global_lag_px",
    "best_global_peak_value",
    "rejected_reason_diagnostic",
]


def load_manifest_for_candidates(cfg: dict[str, Any]) -> pd.DataFrame:
    path = manifest_csv_path(cfg)
    if path.exists():
        return pd.read_csv(path, dtype={"image_id": str, "donor_id": str, "region_id": str})
    return build_manifest(cfg)


def select_candidate_manifest_rows(
    manifest: pd.DataFrame,
    image_ids: set[str] | None,
    run_all: bool,
    max_images: int | None,
) -> pd.DataFrame:
    if not run_all and not image_ids:
        raise ValueError("Pass --image-id or --all for candidate diagnostics.")
    selected = manifest.copy()
    if image_ids:
        selected = selected.loc[selected["image_id"].astype(str).isin(image_ids)].copy()
    if max_images is not None:
        selected = selected.head(max_images).copy()
    return selected.reset_index(drop=True)


def diagnose_spacing_candidates_for_manifest(
    cfg: dict[str, Any],
    selected_manifest: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for _, row in selected_manifest.iterrows():
        frames.append(diagnose_spacing_candidates_for_image(row, cfg))
    if not frames:
        return pd.DataFrame(columns=SPACING_CANDIDATE_COLUMNS)
    result = pd.concat(frames, ignore_index=True)
    return stabilize_candidate_columns(result)


def diagnose_spacing_candidates_for_image(row: pd.Series, cfg: dict[str, Any]) -> pd.DataFrame:
    image_id = str(row["image_id"])
    donor_id = str(row["donor_id"]) if "donor_id" in row and pd.notna(row["donor_id"]) else ""
    raw = load_tiff(Path(str(row["image_path"])))
    preprocessing = preprocess_image(raw, cfg)
    mask = compute_tissue_mask(preprocessing.image, cfg)
    patch_qc = compute_patch_qc(preprocessing.image, mask.mask, image_id, cfg)
    orientation = compute_orientation_analysis(preprocessing.image, mask.mask, patch_qc, cfg)
    patch_metrics = orientation.patch_metrics.copy()
    if "donor_id" not in patch_metrics.columns:
        patch_metrics.insert(1, "donor_id", donor_id)
    rows = [
        diagnose_patch_candidates(preprocessing.image, patch_row, cfg)
        for _, patch_row in patch_metrics.iterrows()
        if reaches_spacing_candidate_stage(patch_row)
    ]
    return stabilize_candidate_columns(pd.DataFrame(rows))


def reaches_spacing_candidate_stage(patch_row: pd.Series) -> bool:
    if not bool(patch_row.get("valid_for_spacing", False)):
        return False
    theta = patch_row.get("patch_mean_orientation_rad", np.nan)
    return bool(np.isfinite(float(theta)))


def diagnose_patch_candidates(image: np.ndarray, patch_row: pd.Series, cfg: dict[str, Any]) -> dict[str, object]:
    params = spacing_params(cfg)
    method = str(params["method"])
    if method != "autocorrelation":
        raise NotImplementedError("Candidate diagnostics currently support spacing.method=autocorrelation.")

    min_px, max_px = spacing_band_px(cfg)
    y0, y1 = int(patch_row["y0"]), int(patch_row["y1"])
    x0, x1 = int(patch_row["x0"]), int(patch_row["x1"])
    patch = image[y0:y1, x0:x1]
    theta = float(patch_row["patch_mean_orientation_rad"])
    final_result = estimate_patch_spacing(image, patch_row, params, cfg)
    base = {
        "image_id": str(patch_row.get("image_id", "")),
        "donor_id": str(patch_row.get("donor_id", "")) if pd.notna(patch_row.get("donor_id", "")) else "",
        "patch_id": str(patch_row.get("patch_id", "")),
        "y0": y0,
        "x0": x0,
        "y1": y1,
        "x1": x1,
        "valid_for_spacing_qc": bool(patch_row.get("valid_for_spacing", False)),
        "final_valid_for_spacing": bool(final_result.valid_for_spacing_final),
        "final_invalid_reason": str(final_result.spacing_invalid_reason),
        "expected_min_lag_px": float(min_px),
        "expected_max_lag_px": float(max_px),
        "selected_lag_px": float("nan"),
        "selected_lag_um": float("nan"),
        "selected_peak_value": float("nan"),
        "baseline_value": float("nan"),
        "peak_prominence": float("nan"),
        "peak_confidence": float(final_result.patch_spacing_confidence),
        "n_local_peaks_total": 0,
        "n_local_peaks_in_band": 0,
        "best_in_band_lag_px": float("nan"),
        "best_in_band_peak_value": float("nan"),
        "best_global_lag_px": float("nan"),
        "best_global_peak_value": float("nan"),
        "rejected_reason_diagnostic": str(final_result.spacing_invalid_reason),
    }

    ac_params = cfg.get("spacing", {}).get("autocorrelation", {})
    profile = prepare_autocorrelation_profile(
        patch,
        theta,
        bin_px=float(ac_params.get("profile_bin_px", 1.0)),
        min_length=int(ac_params.get("min_profile_length_px", 32)),
    )
    autocorr = normalized_autocorrelation(profile)
    if autocorr.size == 0:
        base["rejected_reason_diagnostic"] = "flat_or_short_profile"
        return base

    total_peak_lags = local_maxima_lags(autocorr, 1, autocorr.size - 1)
    min_lag = int(np.ceil(min_px))
    max_lag = min(int(np.floor(max_px)), autocorr.size - 1)
    in_band_lags = local_maxima_lags(autocorr, min_lag, max_lag) if min_lag <= max_lag else []
    base["n_local_peaks_total"] = int(len(total_peak_lags))
    base["n_local_peaks_in_band"] = int(len(in_band_lags))
    best_global = best_peak(autocorr, total_peak_lags)
    best_band = best_peak(autocorr, in_band_lags)
    if best_global is not None:
        base["best_global_lag_px"] = float(best_global[0])
        base["best_global_peak_value"] = float(best_global[1])
    if best_band is not None:
        base["best_in_band_lag_px"] = float(best_band[0])
        base["best_in_band_peak_value"] = float(best_band[1])

    selection = select_autocorrelation_peak(autocorr, min_px, max_px, cfg)
    if selection.get("reason") == "ok":
        lag = float(selection["lag"])
        peak = float(selection["peak"])
        baseline = float(selection["baseline"])
        confidence = float(selection["confidence"])
        base.update(
            {
                "selected_lag_px": lag,
                "selected_lag_um": px_to_um(lag, cfg),
                "selected_peak_value": peak,
                "baseline_value": baseline,
                "peak_prominence": confidence,
                "peak_confidence": confidence,
            }
        )
    else:
        base["rejected_reason_diagnostic"] = str(selection.get("reason", final_result.spacing_invalid_reason))

    if best_global is not None and best_band is None:
        base["rejected_reason_diagnostic"] = "best_global_peak_outside_expected_band"
    return base


def best_peak(autocorr: np.ndarray, lags: list[int]) -> tuple[int, float] | None:
    if not lags:
        return None
    values = np.asarray([autocorr[lag] for lag in lags], dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return None
    finite_lags = np.asarray(lags, dtype=int)[finite]
    finite_values = values[finite]
    index = int(np.argmax(finite_values))
    return int(finite_lags[index]), float(finite_values[index])


def summarize_spacing_candidates(
    candidates: pd.DataFrame,
    main_patch_table: pd.DataFrame | None = None,
) -> dict[str, Any]:
    table = stabilize_candidate_columns(candidates)
    summary: dict[str, Any] = {
        "total_analyzed_patches": int(len(table)),
        "qc_valid_spacing_patches": int(table["valid_for_spacing_qc"].fillna(False).astype(bool).sum()) if not table.empty else 0,
        "patches_with_any_local_peak": int((pd.to_numeric(table["n_local_peaks_total"], errors="coerce") > 0).sum()) if not table.empty else 0,
        "patches_with_local_peak_inside_expected_band": int((pd.to_numeric(table["n_local_peaks_in_band"], errors="coerce") > 0).sum()) if not table.empty else 0,
        "patches_where_best_global_peak_outside_expected_band": int(best_global_outside_band(table).sum()) if not table.empty else 0,
        "best_in_band_lag_px_distribution": numeric_value_counts(table["best_in_band_lag_px"]) if not table.empty else {},
        "best_global_lag_px_distribution": numeric_value_counts(table["best_global_lag_px"]) if not table.empty else {},
        "confidence_quantiles": numeric_quantiles(table["peak_confidence"]) if not table.empty else empty_quantiles(),
        "prominence_quantiles": numeric_quantiles(table["peak_prominence"]) if not table.empty else empty_quantiles(),
        "final_accepted_patch_count": int(table["final_valid_for_spacing"].fillna(False).astype(bool).sum()) if not table.empty else 0,
        "rejected_reason_diagnostic_counts": value_counts(table, "rejected_reason_diagnostic"),
        "main_table_comparison": compare_to_main_table(table, main_patch_table) if main_patch_table is not None else None,
        "interpretation": candidate_interpretation(),
    }
    return json_safe(summary)


def best_global_outside_band(table: pd.DataFrame) -> pd.Series:
    best = pd.to_numeric(table["best_global_lag_px"], errors="coerce")
    min_lag = pd.to_numeric(table["expected_min_lag_px"], errors="coerce")
    max_lag = pd.to_numeric(table["expected_max_lag_px"], errors="coerce")
    return best.notna() & ((best < np.ceil(min_lag)) | (best > np.floor(max_lag)))


def compare_to_main_table(candidates: pd.DataFrame, main_patch_table: pd.DataFrame) -> dict[str, Any]:
    main = main_patch_table.copy()
    if not {"image_id", "patch_id", "valid_for_spacing_final"}.issubset(main.columns):
        return {"available": False, "mismatch_count": None, "message": "Main table lacks comparison columns."}
    merged = candidates[["image_id", "patch_id", "final_valid_for_spacing"]].merge(
        main[["image_id", "patch_id", "valid_for_spacing_final"]],
        on=["image_id", "patch_id"],
        how="left",
    )
    merged["main_valid_for_spacing_final"] = merged["valid_for_spacing_final"].fillna(False).astype(bool)
    merged["diagnostic_final_valid"] = merged["final_valid_for_spacing"].fillna(False).astype(bool)
    mismatches = merged.loc[merged["main_valid_for_spacing_final"] != merged["diagnostic_final_valid"]]
    return {
        "available": True,
        "compared_patches": int(len(merged)),
        "mismatch_count": int(len(mismatches)),
        "mismatch_examples": mismatches[["image_id", "patch_id", "diagnostic_final_valid", "main_valid_for_spacing_final"]]
        .head(20)
        .to_dict(orient="records"),
    }


def write_spacing_candidate_outputs(
    candidates: pd.DataFrame,
    summary: dict[str, Any],
    output_directory: str | Path,
) -> dict[str, Path]:
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidates": out_dir / "spacing_candidates.csv",
        "summary_json": out_dir / "spacing_candidates_summary.json",
        "summary_txt": out_dir / "spacing_candidates_summary.txt",
    }
    stabilize_candidate_columns(candidates).to_csv(paths["candidates"], index=False)
    with paths["summary_json"].open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2)
        handle.write("\n")
    paths["summary_txt"].write_text(format_spacing_candidate_summary(summary), encoding="utf-8")
    return paths


def stabilize_candidate_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in SPACING_CANDIDATE_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
    return result[SPACING_CANDIDATE_COLUMNS]


def value_counts(df: pd.DataFrame, column: str, limit: int = 20) -> dict[str, int]:
    if column not in df.columns:
        return {}
    counts = df[column].fillna("<NA>").astype(str).value_counts().head(limit)
    return {str(key): int(value) for key, value in counts.items()}


def numeric_value_counts(series: pd.Series, limit: int = 20, decimals: int = 6) -> dict[str, int]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {}
    counts = values.round(decimals).value_counts().sort_index().head(limit)
    return {str(key): int(value) for key, value in counts.items()}


def numeric_quantiles(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return empty_quantiles()
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def empty_quantiles() -> dict[str, None | int]:
    return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}


def candidate_interpretation() -> dict[str, str]:
    return {
        "diagnostic_only": "Candidate-level diagnostics inspect spacing peaks without changing endpoint metrics.",
        "no_threshold_tuning": "These outputs are intended to guide evidence-based sensitivity analysis, not manual threshold tuning.",
        "spacing_warning": "Sparse accepted spacing remains preliminary and should not be treated as a validated biological endpoint.",
    }


def format_spacing_candidate_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Spacing Candidate Diagnostics",
        f"total_analyzed_patches: {summary['total_analyzed_patches']}",
        f"qc_valid_spacing_patches: {summary['qc_valid_spacing_patches']}",
        f"patches_with_any_local_peak: {summary['patches_with_any_local_peak']}",
        f"patches_with_local_peak_inside_expected_band: {summary['patches_with_local_peak_inside_expected_band']}",
        f"patches_where_best_global_peak_outside_expected_band: {summary['patches_where_best_global_peak_outside_expected_band']}",
        f"final_accepted_patch_count: {summary['final_accepted_patch_count']}",
        f"best_in_band_lag_px_distribution: {summary['best_in_band_lag_px_distribution']}",
        f"best_global_lag_px_distribution: {summary['best_global_lag_px_distribution']}",
        f"confidence_quantiles: {summary['confidence_quantiles']}",
        f"prominence_quantiles: {summary['prominence_quantiles']}",
        f"rejected_reason_diagnostic_counts: {summary['rejected_reason_diagnostic_counts']}",
        f"main_table_comparison: {summary['main_table_comparison']}",
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
