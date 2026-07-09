from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sarcomere_analysis.config import get_calibration


PATCH_SPACING_COLUMNS = [
    "patch_spacing_um",
    "patch_spacing_px",
    "patch_periodicity_score",
    "patch_spacing_confidence",
    "patch_spacing_method",
    "valid_for_spacing_final",
    "spacing_invalid_reason",
]

IMAGE_SPACING_COLUMNS = [
    "image_spacing_mean_um",
    "image_spacing_median_um",
    "image_spacing_std_um",
    "image_spacing_cv",
    "n_spacing_valid_patches",
    "spacing_valid_fraction",
]


@dataclass(frozen=True)
class PatchSpacingResult:
    patch_spacing_um: float
    patch_spacing_px: float
    patch_periodicity_score: float
    patch_spacing_confidence: float
    patch_spacing_method: str
    valid_for_spacing_final: bool
    spacing_invalid_reason: str


@dataclass(frozen=True)
class ImageSpacingResult:
    patch_metrics: pd.DataFrame
    image_metrics: dict[str, float | int]


def compute_spacing_analysis(
    image: np.ndarray,
    patch_metrics: pd.DataFrame,
    config: dict[str, Any],
) -> ImageSpacingResult:
    params = spacing_params(config)
    rows: list[dict[str, object]] = []
    for _, row in patch_metrics.iterrows():
        result = estimate_patch_spacing(image, row, params, config)
        rows.append(
            {
                "patch_spacing_um": result.patch_spacing_um,
                "patch_spacing_px": result.patch_spacing_px,
                "patch_periodicity_score": result.patch_periodicity_score,
                "patch_spacing_confidence": result.patch_spacing_confidence,
                "patch_spacing_method": result.patch_spacing_method,
                "valid_for_spacing_final": result.valid_for_spacing_final,
                "spacing_invalid_reason": result.spacing_invalid_reason,
            }
        )

    spacing_df = pd.DataFrame(rows, columns=PATCH_SPACING_COLUMNS)
    combined = pd.concat([patch_metrics.reset_index(drop=True), spacing_df], axis=1)
    return ImageSpacingResult(
        patch_metrics=combined,
        image_metrics=summarize_image_spacing(combined),
    )


def estimate_patch_spacing(
    image: np.ndarray,
    patch_row: pd.Series,
    params: dict[str, Any],
    config: dict[str, Any],
) -> PatchSpacingResult:
    method = str(params["method"])
    earlier_reason = str(patch_row.get("invalid_reason", "invalid"))
    if not bool(patch_row["valid_for_spacing"]):
        return invalid_spacing_result(method, _append_reason(earlier_reason, "failed_patch_qc"))

    theta = float(patch_row["patch_mean_orientation_rad"])
    if not np.isfinite(theta):
        return invalid_spacing_result(method, _append_reason(earlier_reason, "missing_orientation"))

    patch = image[
        int(patch_row["y0"]) : int(patch_row["y1"]),
        int(patch_row["x0"]) : int(patch_row["x1"]),
    ]
    if method == "autocorrelation":
        from .autocorrelation import estimate_spacing_autocorrelation

        result = estimate_spacing_autocorrelation(patch, theta, config)
    elif method == "fft":
        from .fft import estimate_spacing_fft

        result = estimate_spacing_fft(patch, theta, config)
    else:
        raise ValueError("spacing.method must be 'autocorrelation' or 'fft'")

    if (not result.valid_for_spacing_final) and bool(params["fallback_to_fft"]) and method != "fft":
        from .fft import estimate_spacing_fft

        fallback = estimate_spacing_fft(patch, theta, config)
        if fallback.valid_for_spacing_final:
            return fallback

    return result


def summarize_image_spacing(patch_metrics: pd.DataFrame) -> dict[str, float | int]:
    valid = patch_metrics.loc[patch_metrics["valid_for_spacing_final"], "patch_spacing_um"].to_numpy(dtype=float)
    valid = valid[np.isfinite(valid)]
    n_valid = int(valid.size)
    total = int(len(patch_metrics))
    if n_valid == 0:
        return {
            "image_spacing_mean_um": float("nan"),
            "image_spacing_median_um": float("nan"),
            "image_spacing_std_um": float("nan"),
            "image_spacing_cv": float("nan"),
            "n_spacing_valid_patches": 0,
            "spacing_valid_fraction": 0.0 if total > 0 else float("nan"),
        }

    mean = float(np.mean(valid))
    std = float(np.std(valid))
    return {
        "image_spacing_mean_um": mean,
        "image_spacing_median_um": float(np.median(valid)),
        "image_spacing_std_um": std,
        "image_spacing_cv": float(std / mean) if mean > 0 else float("nan"),
        "n_spacing_valid_patches": n_valid,
        "spacing_valid_fraction": float(n_valid / total) if total > 0 else float("nan"),
    }


def spacing_params(config: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "method": "autocorrelation",
        "fallback_to_fft": False,
        "min_spacing_um": None,
        "max_spacing_um": None,
        "min_periodicity_confidence": 0.15,
        "autocorrelation": {
            "profile_bin_px": 1.0,
            "min_profile_length_px": 32,
            "peak_baseline_percentile": 50.0,
        },
        "fft": {
            "min_profile_length_px": 32,
            "peak_prominence_ratio": 4.0,
        },
    }
    params = dict(defaults)
    params.update(config.get("spacing", {}))
    params["method"] = str(params["method"]).lower()
    if params["method"] not in {"autocorrelation", "fft"}:
        raise ValueError("spacing.method must be 'autocorrelation' or 'fft'")
    if float(params["min_periodicity_confidence"]) < 0:
        raise ValueError("spacing.min_periodicity_confidence must be non-negative")
    min_um, max_um = spacing_band_um(config)
    if min_um <= 0 or max_um <= 0 or min_um >= max_um:
        raise ValueError("spacing min/max band must be positive and min < max")
    return params


def spacing_band_um(config: dict[str, Any]) -> tuple[float, float]:
    calibration = get_calibration(config)
    spacing_config = config.get("spacing", {})
    min_um = spacing_config.get("min_spacing_um", calibration.expected_spacing_um_min)
    max_um = spacing_config.get("max_spacing_um", calibration.expected_spacing_um_max)
    if min_um is None:
        min_um = calibration.expected_spacing_um_min
    if max_um is None:
        max_um = calibration.expected_spacing_um_max
    return float(min_um), float(max_um)


def spacing_band_px(config: dict[str, Any]) -> tuple[float, float]:
    calibration = get_calibration(config)
    min_um, max_um = spacing_band_um(config)
    return min_um / calibration.pixel_size_um, max_um / calibration.pixel_size_um


def px_to_um(value_px: float, config: dict[str, Any]) -> float:
    return float(value_px * get_calibration(config).pixel_size_um)


def invalid_spacing_result(method: str, reason: str) -> PatchSpacingResult:
    return PatchSpacingResult(
        patch_spacing_um=float("nan"),
        patch_spacing_px=float("nan"),
        patch_periodicity_score=float("nan"),
        patch_spacing_confidence=0.0,
        patch_spacing_method=method,
        valid_for_spacing_final=False,
        spacing_invalid_reason=reason,
    )


def valid_spacing_result(
    spacing_px: float,
    score: float,
    confidence: float,
    method: str,
    config: dict[str, Any],
) -> PatchSpacingResult:
    return PatchSpacingResult(
        patch_spacing_um=px_to_um(spacing_px, config),
        patch_spacing_px=float(spacing_px),
        patch_periodicity_score=float(score),
        patch_spacing_confidence=float(confidence),
        patch_spacing_method=method,
        valid_for_spacing_final=True,
        spacing_invalid_reason="ok",
    )


def _append_reason(existing: str, reason: str) -> str:
    if existing in {"", "ok", "nan"}:
        return reason
    if reason in existing.split(";"):
        return existing
    return f"{existing};{reason}"
