from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sarcomere_analysis.config import output_dir
from sarcomere_analysis.outputs import write_heatmap

from .autocorrelation import (
    directional_profile,
    normalized_autocorrelation,
    prepare_autocorrelation_profile,
    select_autocorrelation_peak,
)
from .base import spacing_band_px, spacing_params, px_to_um


SPACING_DIAGNOSTIC_COLUMNS = [
    "image_id",
    "donor_id",
    "patch_id",
    "y0",
    "x0",
    "y1",
    "x1",
    "valid_for_spacing",
    "valid_for_spacing_final",
    "patch_spacing_um",
    "patch_spacing_px",
    "patch_spacing_confidence",
    "expected_spacing_min_px",
    "expected_spacing_max_px",
    "selected_lag_px",
    "selected_lag_um",
    "peak_score",
    "peak_rank_or_index",
    "autocorr_peak_value",
    "autocorr_baseline_value",
    "confidence_threshold",
    "spacing_rejection_stage",
    "spacing_invalid_reason",
]

SPACING_BY_IMAGE_COLUMNS = [
    "image_id",
    "donor_id",
    "total_patches",
    "valid_for_spacing_qc_count",
    "missing_orientation_count",
    "failed_patch_qc_count",
    "low_periodicity_confidence_count",
    "accepted_spacing_count",
    "accepted_spacing_fraction",
    "selected_lag_px_min",
    "selected_lag_px_median",
    "selected_lag_px_max",
    "selected_lag_px_counts",
    "selected_lag_um_min",
    "selected_lag_um_median",
    "selected_lag_um_max",
    "selected_lag_um_counts",
    "accepted_near_lower_bound_count",
    "accepted_near_lower_bound_fraction",
    "accepted_near_upper_bound_count",
    "accepted_near_upper_bound_fraction",
    "accepted_confidence_median",
    "rejected_confidence_median",
    "top_invalid_reasons",
    "top_rejection_stages",
]

SPACING_SUMMARY_COLUMNS = [
    "total_images",
    "total_patches",
    "valid_for_spacing_qc_count",
    "missing_orientation_count",
    "failed_patch_qc_count",
    "low_periodicity_confidence_count",
    "accepted_spacing_count",
    "accepted_spacing_fraction",
    "accepted_selected_lag_px_min",
    "accepted_selected_lag_px_median",
    "accepted_selected_lag_px_max",
    "accepted_selected_lag_px_counts",
    "accepted_selected_lag_um_min",
    "accepted_selected_lag_um_median",
    "accepted_selected_lag_um_max",
    "accepted_selected_lag_um_counts",
    "accepted_near_lower_bound_count",
    "accepted_near_lower_bound_fraction",
    "accepted_near_upper_bound_count",
    "accepted_near_upper_bound_fraction",
    "accepted_confidence_median",
    "rejected_confidence_median",
    "top_invalid_reasons",
    "top_rejection_stages",
]


def diagnose_spacing_analysis(
    image: np.ndarray,
    patch_metrics: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows = [diagnose_patch_spacing(image, row, config) for _, row in patch_metrics.iterrows()]
    result = pd.DataFrame(rows)
    for column in SPACING_DIAGNOSTIC_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
    return result[SPACING_DIAGNOSTIC_COLUMNS]


def diagnose_patch_spacing(image: np.ndarray, patch_row: pd.Series, config: dict[str, Any]) -> dict[str, object]:
    min_px, max_px = spacing_band_px(config)
    threshold = float(config.get("spacing", {}).get("min_periodicity_confidence", 0.15))
    method = str(spacing_params(config)["method"])
    row = base_diagnostic_row(patch_row, min_px, max_px, threshold)

    earlier_reason = str(patch_row.get("invalid_reason", "invalid"))
    if not bool(patch_row["valid_for_spacing"]):
        return finish_invalid(row, "failed_patch_qc", _append_reason(earlier_reason, "failed_patch_qc"))

    theta = float(patch_row["patch_mean_orientation_rad"])
    if not np.isfinite(theta):
        return finish_invalid(row, "missing_orientation", _append_reason(earlier_reason, "missing_orientation"))

    patch = image[
        int(patch_row["y0"]) : int(patch_row["y1"]),
        int(patch_row["x0"]) : int(patch_row["x1"]),
    ]
    if method == "autocorrelation":
        return diagnose_autocorrelation_patch(patch, theta, config, row)
    if method == "fft":
        return diagnose_fft_patch(patch, theta, config, row)
    raise ValueError("spacing.method must be 'autocorrelation' or 'fft'")


def diagnose_autocorrelation_patch(
    patch: np.ndarray,
    orientation_rad: float,
    config: dict[str, Any],
    row: dict[str, object],
) -> dict[str, object]:
    params = config.get("spacing", {}).get("autocorrelation", {})
    profile = prepare_autocorrelation_profile(
        patch,
        orientation_rad,
        bin_px=float(params.get("profile_bin_px", 1.0)),
        min_length=int(params.get("min_profile_length_px", 32)),
    )
    if profile.size == 0:
        return finish_invalid(row, "profile", "short_profile")
    if not np.isfinite(profile).all() or float(np.std(profile)) <= 0:
        return finish_invalid(row, "profile", "flat_profile")

    autocorr = normalized_autocorrelation(profile)
    if autocorr.size == 0:
        return finish_invalid(row, "profile", "flat_profile")

    min_px = float(row["expected_spacing_min_px"])
    max_px = float(row["expected_spacing_max_px"])
    selection = select_autocorrelation_peak(autocorr, min_px, max_px, config)
    if selection["reason"] != "ok":
        return finish_invalid(row, rejection_stage_for_reason(str(selection["reason"])), str(selection["reason"]))

    lag = float(selection["lag"])
    peak = float(selection["peak"])
    baseline = float(selection["baseline"])
    confidence = float(selection["confidence"])

    row.update(
        {
            "selected_lag_px": lag,
            "selected_lag_um": px_to_um(lag, config),
            "peak_score": peak,
            "peak_rank_or_index": int(selection["peak_index"]),
            "autocorr_peak_value": peak,
            "autocorr_baseline_value": baseline,
            "patch_spacing_confidence": confidence,
        }
    )
    threshold = float(row["confidence_threshold"])
    if not np.isfinite(confidence) or confidence < threshold:
        return finish_invalid(row, "confidence", "low_periodicity_confidence")
    return finish_valid(row, lag, px_to_um(lag, config), peak, confidence)


def rejection_stage_for_reason(reason: str) -> str:
    if reason == "spacing_band_out_of_range":
        return "spacing_band"
    if reason == "no_local_peak":
        return "peak_picking"
    return "profile"


def diagnose_fft_patch(
    patch: np.ndarray,
    orientation_rad: float,
    config: dict[str, Any],
    row: dict[str, object],
) -> dict[str, object]:
    params = config.get("spacing", {}).get("fft", {})
    profile = directional_profile(patch, orientation_rad, bin_px=1.0)
    min_length = int(params.get("min_profile_length_px", 32))
    if profile.size < min_length:
        return finish_invalid(row, "profile", "short_profile")

    profile = profile.astype(np.float64, copy=False)
    profile = profile - float(np.mean(profile))
    if not np.isfinite(profile).all() or float(np.std(profile)) <= 0:
        return finish_invalid(row, "profile", "flat_profile")

    windowed = profile * np.hanning(profile.size)
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(profile.size, d=1.0)
    power = np.abs(spectrum) ** 2
    if power.size <= 1:
        return finish_invalid(row, "profile", "weak_fft_peak")

    min_px = float(row["expected_spacing_min_px"])
    max_px = float(row["expected_spacing_max_px"])
    min_freq = 1.0 / max_px
    max_freq = 1.0 / min_px
    band_mask = (freqs >= min_freq) & (freqs <= max_freq)
    band_mask[0] = False
    if not np.any(band_mask):
        return finish_invalid(row, "spacing_band", "spacing_band_out_of_range")

    band_power = power[band_mask]
    band_freqs = freqs[band_mask]
    peak_index = int(np.argmax(band_power))
    peak_power = float(band_power[peak_index])
    baseline = float(np.median(band_power))
    if baseline <= 0 or not np.isfinite(baseline):
        return finish_invalid(row, "profile", "weak_fft_peak")

    ratio = peak_power / baseline
    required_ratio = float(params.get("peak_prominence_ratio", 4.0))
    confidence = max(0.0, (ratio - 1.0) / max(required_ratio - 1.0, 1e-12))
    spacing_px = float(1.0 / band_freqs[peak_index])
    row.update(
        {
            "selected_lag_px": spacing_px,
            "selected_lag_um": px_to_um(spacing_px, config),
            "peak_score": ratio,
            "peak_rank_or_index": peak_index,
            "autocorr_peak_value": float("nan"),
            "autocorr_baseline_value": baseline,
            "patch_spacing_confidence": confidence,
        }
    )
    threshold = float(row["confidence_threshold"])
    if ratio < required_ratio or confidence < threshold:
        return finish_invalid(row, "confidence", "low_periodicity_confidence")
    return finish_valid(row, spacing_px, px_to_um(spacing_px, config), ratio, confidence)


def base_diagnostic_row(patch_row: pd.Series, min_px: float, max_px: float, threshold: float) -> dict[str, object]:
    return {
        "image_id": str(patch_row.get("image_id", "")),
        "donor_id": str(patch_row.get("donor_id", "")) if pd.notna(patch_row.get("donor_id", "")) else "",
        "patch_id": str(patch_row.get("patch_id", "")),
        "y0": int(patch_row["y0"]),
        "x0": int(patch_row["x0"]),
        "y1": int(patch_row["y1"]),
        "x1": int(patch_row["x1"]),
        "valid_for_spacing": bool(patch_row["valid_for_spacing"]),
        "valid_for_spacing_final": False,
        "patch_spacing_um": float("nan"),
        "patch_spacing_px": float("nan"),
        "patch_spacing_confidence": 0.0,
        "expected_spacing_min_px": float(min_px),
        "expected_spacing_max_px": float(max_px),
        "selected_lag_px": float("nan"),
        "selected_lag_um": float("nan"),
        "peak_score": float("nan"),
        "peak_rank_or_index": float("nan"),
        "autocorr_peak_value": float("nan"),
        "autocorr_baseline_value": float("nan"),
        "confidence_threshold": float(threshold),
        "spacing_rejection_stage": "uncomputed",
        "spacing_invalid_reason": "uncomputed",
    }


def finish_invalid(row: dict[str, object], stage: str, reason: str) -> dict[str, object]:
    row["valid_for_spacing_final"] = False
    row["spacing_rejection_stage"] = stage
    row["spacing_invalid_reason"] = reason
    return row


def finish_valid(
    row: dict[str, object],
    spacing_px: float,
    spacing_um: float,
    score: float,
    confidence: float,
) -> dict[str, object]:
    row.update(
        {
            "valid_for_spacing_final": True,
            "patch_spacing_px": float(spacing_px),
            "patch_spacing_um": float(spacing_um),
            "patch_spacing_confidence": float(confidence),
            "peak_score": float(score),
            "spacing_rejection_stage": "accepted",
            "spacing_invalid_reason": "ok",
        }
    )
    return row


def summarize_spacing_diagnostics(diagnostics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_image_rows = []
    for (image_id, donor_id), group in diagnostics.groupby(["image_id", "donor_id"], dropna=False):
        by_image_rows.append(summarize_group(group, image_id=str(image_id), donor_id=str(donor_id)))
    by_image = pd.DataFrame(by_image_rows)
    for column in SPACING_BY_IMAGE_COLUMNS:
        if column not in by_image.columns:
            by_image[column] = np.nan
    by_image = by_image[SPACING_BY_IMAGE_COLUMNS]

    summary = pd.DataFrame([summarize_group(diagnostics, image_id=None, donor_id=None)])
    summary = summary.rename(
        columns={
            "selected_lag_px_min": "accepted_selected_lag_px_min",
            "selected_lag_px_median": "accepted_selected_lag_px_median",
            "selected_lag_px_max": "accepted_selected_lag_px_max",
            "selected_lag_px_counts": "accepted_selected_lag_px_counts",
            "selected_lag_um_min": "accepted_selected_lag_um_min",
            "selected_lag_um_median": "accepted_selected_lag_um_median",
            "selected_lag_um_max": "accepted_selected_lag_um_max",
            "selected_lag_um_counts": "accepted_selected_lag_um_counts",
        }
    )
    summary["total_images"] = int(diagnostics["image_id"].nunique()) if not diagnostics.empty else 0
    for column in SPACING_SUMMARY_COLUMNS:
        if column not in summary.columns:
            summary[column] = np.nan
    return summary[SPACING_SUMMARY_COLUMNS], by_image


def summarize_group(group: pd.DataFrame, image_id: str | None, donor_id: str | None) -> dict[str, object]:
    total = int(len(group))
    accepted = group.loc[group["valid_for_spacing_final"].astype(bool)].copy() if total else pd.DataFrame()
    rejected = group.loc[~group["valid_for_spacing_final"].astype(bool)].copy() if total else pd.DataFrame()
    accepted_count = int(len(accepted))
    min_px = float(group["expected_spacing_min_px"].iloc[0]) if total else float("nan")
    max_px = float(group["expected_spacing_max_px"].iloc[0]) if total else float("nan")
    near_lower = accepted["selected_lag_px"] <= np.ceil(min_px)
    near_upper = accepted["selected_lag_px"] >= np.floor(max_px)

    row: dict[str, object] = {
        "total_patches": total,
        "valid_for_spacing_qc_count": int(group["valid_for_spacing"].sum()) if total else 0,
        "missing_orientation_count": count_stage(group, "missing_orientation"),
        "failed_patch_qc_count": count_stage(group, "failed_patch_qc"),
        "low_periodicity_confidence_count": int((group["spacing_invalid_reason"] == "low_periodicity_confidence").sum()) if total else 0,
        "accepted_spacing_count": accepted_count,
        "accepted_spacing_fraction": float(accepted_count / total) if total else float("nan"),
        "selected_lag_px_min": finite_stat(accepted["selected_lag_px"], np.min),
        "selected_lag_px_median": finite_stat(accepted["selected_lag_px"], np.median),
        "selected_lag_px_max": finite_stat(accepted["selected_lag_px"], np.max),
        "selected_lag_px_counts": numeric_top_counts(accepted["selected_lag_px"]),
        "selected_lag_um_min": finite_stat(accepted["selected_lag_um"], np.min),
        "selected_lag_um_median": finite_stat(accepted["selected_lag_um"], np.median),
        "selected_lag_um_max": finite_stat(accepted["selected_lag_um"], np.max),
        "selected_lag_um_counts": numeric_top_counts(accepted["selected_lag_um"]),
        "accepted_near_lower_bound_count": int(near_lower.sum()) if accepted_count else 0,
        "accepted_near_lower_bound_fraction": float(near_lower.mean()) if accepted_count else float("nan"),
        "accepted_near_upper_bound_count": int(near_upper.sum()) if accepted_count else 0,
        "accepted_near_upper_bound_fraction": float(near_upper.mean()) if accepted_count else float("nan"),
        "accepted_confidence_median": finite_stat(accepted["patch_spacing_confidence"], np.median),
        "rejected_confidence_median": finite_stat(rejected["patch_spacing_confidence"], np.median),
        "top_invalid_reasons": top_counts(group["spacing_invalid_reason"]),
        "top_rejection_stages": top_counts(group["spacing_rejection_stage"]),
    }
    if image_id is not None:
        row["image_id"] = image_id
    if donor_id is not None:
        row["donor_id"] = donor_id
    return row


def write_diagnostic_tables(
    diagnostics: pd.DataFrame,
    output_directory: str | Path,
    write_patch_diagnostics: bool,
) -> dict[str, Path]:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    summary, by_image = summarize_spacing_diagnostics(diagnostics)
    paths = {
        "summary": output_path / "spacing_diagnostic_summary.csv",
        "by_image": output_path / "spacing_diagnostic_by_image.csv",
    }
    summary.to_csv(paths["summary"], index=False)
    by_image.to_csv(paths["by_image"], index=False)
    if write_patch_diagnostics:
        for image_id, group in diagnostics.groupby("image_id", sort=False):
            path = output_path / f"{image_id}_spacing_patch_diagnostics.csv"
            group.to_csv(path, index=False)
        paths["patch_diagnostics_dir"] = output_path
    return paths


def write_spacing_confidence_heatmap(
    diagnostics: pd.DataFrame,
    image_shape: tuple[int, int],
    image_id: str,
    config: dict[str, Any],
    output_directory: str | Path | None = None,
) -> Path:
    out_dir = Path(output_directory) if output_directory is not None else output_dir(config) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{image_id}_spacing_confidence_heatmap.png"
    return write_heatmap("patch_spacing_confidence", diagnostics, image_shape, path, config)


def write_autocorrelation_debug_plot(
    patch: np.ndarray,
    orientation_rad: float,
    config: dict[str, Any],
    image_id: str,
    patch_id: str,
    output_directory: str | Path,
) -> Path:
    import os

    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(out_dir / ".cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = config.get("spacing", {}).get("autocorrelation", {})
    raw_profile = directional_profile(patch, orientation_rad, bin_px=float(params.get("profile_bin_px", 1.0)))
    profile = prepare_autocorrelation_profile(
        patch,
        orientation_rad,
        bin_px=float(params.get("profile_bin_px", 1.0)),
        min_length=int(params.get("min_profile_length_px", 32)),
    )
    autocorr = normalized_autocorrelation(profile)
    min_px, max_px = spacing_band_px(config)
    selection = select_autocorrelation_peak(autocorr, min_px, max_px, config) if autocorr.size else {"reason": "flat_profile"}

    path = out_dir / f"{image_id}_patch_{patch_id}_autocorr_debug.png"

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    axes[0].imshow(patch, cmap="gray")
    axes[0].set_title("Patch")
    axes[0].axis("off")

    axes[1].plot(raw_profile, color="black", linewidth=1.2)
    axes[1].set_title("Directional profile")
    axes[1].set_xlabel("Profile bin")
    axes[1].set_ylabel("Intensity")

    if autocorr.size:
        lags = np.arange(autocorr.size)
        axes[2].plot(lags, autocorr, color="black", linewidth=1.2)
        axes[2].axvspan(float(np.ceil(min_px)), float(np.floor(max_px)), color="#8fb7ff", alpha=0.25, label="Expected band")
        if selection.get("reason") == "ok":
            lag = float(selection["lag"])
            axes[2].axvline(lag, color="#d33", linestyle="--", linewidth=1.2, label=f"Selected {lag:g}px")
        axes[2].legend(loc="best", fontsize=8)
    axes[2].set_title(f"Autocorrelation: {selection.get('reason')}")
    axes[2].set_xlabel("Lag (px)")
    axes[2].set_ylabel("Normalized autocorr")

    fig.suptitle(f"{image_id} / {patch_id}")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def finite_stat(values: pd.Series, fn) -> float:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(fn(array))


def count_stage(group: pd.DataFrame, stage: str) -> int:
    if group.empty:
        return 0
    return int((group["spacing_rejection_stage"] == stage).sum())


def top_counts(series: pd.Series, n: int = 8) -> str:
    if series.empty:
        return ""
    counts = series.fillna("NA").astype(str).value_counts().head(n)
    return "; ".join(f"{index}:{int(value)}" for index, value in counts.items())


def numeric_top_counts(series: pd.Series, n: int = 12, decimals: int = 6) -> str:
    if series.empty:
        return ""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return ""
    counts = values.round(decimals).value_counts().sort_index().head(n)
    return "; ".join(f"{index:g}:{int(value)}" for index, value in counts.items())


def _append_reason(existing: str, reason: str) -> str:
    if existing in {"", "ok", "nan"}:
        return reason
    if reason in existing.split(";"):
        return existing
    return f"{existing};{reason}"
