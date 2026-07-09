from __future__ import annotations

from typing import Any

import numpy as np

from .autocorrelation import directional_profile
from .base import (
    PatchSpacingResult,
    invalid_spacing_result,
    spacing_band_px,
    valid_spacing_result,
)


def estimate_spacing_fft(
    patch: np.ndarray,
    orientation_rad: float,
    config: dict[str, Any],
) -> PatchSpacingResult:
    method = "fft"
    params = config.get("spacing", {}).get("fft", {})
    profile = directional_profile(patch, orientation_rad, bin_px=1.0)
    min_length = int(params.get("min_profile_length_px", 32))
    if profile.size < min_length:
        return invalid_spacing_result(method, "short_profile")

    profile = profile.astype(np.float64, copy=False)
    profile = profile - float(np.mean(profile))
    if not np.isfinite(profile).all() or float(np.std(profile)) <= 0:
        return invalid_spacing_result(method, "flat_profile")

    windowed = profile * np.hanning(profile.size)
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(profile.size, d=1.0)
    power = np.abs(spectrum) ** 2
    if power.size <= 1:
        return invalid_spacing_result(method, "weak_fft_peak")

    min_px, max_px = spacing_band_px(config)
    min_freq = 1.0 / max_px
    max_freq = 1.0 / min_px
    band_mask = (freqs >= min_freq) & (freqs <= max_freq)
    band_mask[0] = False
    if not np.any(band_mask):
        return invalid_spacing_result(method, "spacing_band_out_of_range")

    band_power = power[band_mask]
    band_freqs = freqs[band_mask]
    peak_index = int(np.argmax(band_power))
    peak_power = float(band_power[peak_index])
    baseline = float(np.median(band_power))
    if baseline <= 0 or not np.isfinite(baseline):
        return invalid_spacing_result(method, "weak_fft_peak")

    ratio = peak_power / baseline
    required_ratio = float(params.get("peak_prominence_ratio", 4.0))
    confidence = max(0.0, (ratio - 1.0) / max(required_ratio - 1.0, 1e-12))
    threshold = float(config.get("spacing", {}).get("min_periodicity_confidence", 0.15))
    if ratio < required_ratio or confidence < threshold:
        return invalid_spacing_result(method, "low_periodicity_confidence")

    spacing_px = float(1.0 / band_freqs[peak_index])
    return valid_spacing_result(
        spacing_px,
        score=ratio,
        confidence=confidence,
        method=method,
        config=config,
    )
