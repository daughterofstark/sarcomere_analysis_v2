from __future__ import annotations

from typing import Any

import numpy as np

from .base import (
    PatchSpacingResult,
    invalid_spacing_result,
    spacing_band_px,
    valid_spacing_result,
)


def estimate_spacing_autocorrelation(
    patch: np.ndarray,
    orientation_rad: float,
    config: dict[str, Any],
) -> PatchSpacingResult:
    method = "autocorrelation"
    params = config.get("spacing", {}).get("autocorrelation", {})
    profile = prepare_autocorrelation_profile(
        patch,
        orientation_rad,
        bin_px=float(params.get("profile_bin_px", 1.0)),
        min_length=int(params.get("min_profile_length_px", 32)),
    )
    if profile.size == 0:
        return invalid_spacing_result(method, "short_profile")
    if not np.isfinite(profile).all() or float(np.std(profile)) <= 0:
        return invalid_spacing_result(method, "flat_profile")

    autocorr = normalized_autocorrelation(profile)
    if autocorr.size == 0:
        return invalid_spacing_result(method, "flat_profile")

    min_px, max_px = spacing_band_px(config)
    selection = select_autocorrelation_peak(autocorr, min_px, max_px, config)
    if selection["reason"] != "ok":
        return invalid_spacing_result(method, str(selection["reason"]))

    confidence = float(selection["confidence"])
    threshold = float(config.get("spacing", {}).get("min_periodicity_confidence", 0.15))
    if not np.isfinite(confidence) or confidence < threshold:
        return invalid_spacing_result(method, "low_periodicity_confidence")

    return valid_spacing_result(
        float(selection["lag"]),
        score=float(selection["peak"]),
        confidence=confidence,
        method=method,
        config=config,
    )


def prepare_autocorrelation_profile(
    patch: np.ndarray,
    orientation_rad: float,
    bin_px: float,
    min_length: int,
) -> np.ndarray:
    profile = directional_profile(patch, orientation_rad, bin_px=bin_px)
    if profile.size < min_length:
        return np.array([], dtype=np.float64)
    profile = profile.astype(np.float64, copy=False)
    return profile - float(np.mean(profile))


def normalized_autocorrelation(profile: np.ndarray) -> np.ndarray:
    if profile.size == 0 or not np.isfinite(profile).all() or float(np.std(profile)) <= 0:
        return np.array([], dtype=np.float64)
    autocorr = np.correlate(profile, profile, mode="full")[profile.size - 1 :]
    if autocorr.size == 0 or autocorr[0] <= 0:
        return np.array([], dtype=np.float64)
    return autocorr / autocorr[0]


def select_autocorrelation_peak(
    autocorr: np.ndarray,
    min_px: float,
    max_px: float,
    config: dict[str, Any],
) -> dict[str, float | int | str]:
    min_lag = int(np.ceil(min_px))
    max_lag = int(np.floor(max_px))
    if max_lag >= autocorr.size:
        max_lag = autocorr.size - 1
    if min_lag > max_lag:
        return {"reason": "spacing_band_out_of_range"}

    band = autocorr[min_lag : max_lag + 1]
    if band.size == 0 or not np.isfinite(band).any():
        return {"reason": "spacing_band_out_of_range"}

    local_peak_lags = local_maxima_lags(autocorr, min_lag, max_lag)
    if not local_peak_lags:
        return {"reason": "no_local_peak"}

    values = np.asarray([autocorr[lag] for lag in local_peak_lags], dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return {"reason": "no_local_peak"}
    finite_lags = np.asarray(local_peak_lags, dtype=int)[finite]
    finite_values = values[finite]
    peak_order = np.argsort(finite_values)[::-1]
    lag = int(finite_lags[int(peak_order[0])])
    peak = float(autocorr[lag])
    params = config.get("spacing", {}).get("autocorrelation", {})
    baseline_percentile = float(params.get("peak_baseline_percentile", 50.0))
    baseline = float(np.nanpercentile(band, baseline_percentile))
    confidence = max(0.0, peak - baseline)
    return {
        "reason": "ok",
        "lag": float(lag),
        "peak": peak,
        "baseline": baseline,
        "confidence": confidence,
        "peak_index": int(lag - min_lag),
    }


def local_maxima_lags(autocorr: np.ndarray, min_lag: int, max_lag: int) -> list[int]:
    peaks: list[int] = []
    for lag in range(min_lag, max_lag + 1):
        center = float(autocorr[lag])
        if not np.isfinite(center):
            continue
        left = float(autocorr[lag - 1]) if lag - 1 >= 0 else float("-inf")
        right = float(autocorr[lag + 1]) if lag + 1 < autocorr.size else float("-inf")
        if center > left and center >= right:
            peaks.append(lag)
    return peaks


def directional_profile(patch: np.ndarray, orientation_rad: float, bin_px: float = 1.0) -> np.ndarray:
    values = np.asarray(patch, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        return np.array([], dtype=np.float32)
    if not np.isfinite(orientation_rad):
        return np.array([], dtype=np.float32)

    height, width = values.shape
    yy, xx = np.indices(values.shape, dtype=np.float32)
    yy = yy - (height - 1) / 2.0
    xx = xx - (width - 1) / 2.0
    projection = xx * np.cos(orientation_rad) + yy * np.sin(orientation_rad)
    bins = np.floor((projection - float(np.min(projection))) / max(bin_px, 1e-6)).astype(int)
    counts = np.bincount(bins.ravel(), minlength=int(bins.max()) + 1)
    sums = np.bincount(bins.ravel(), weights=values.ravel(), minlength=int(bins.max()) + 1)
    valid = counts > 0
    if not np.any(valid):
        return np.array([], dtype=np.float32)
    profile = sums[valid] / counts[valid]
    return profile.astype(np.float32, copy=False)
