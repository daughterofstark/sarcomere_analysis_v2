from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .patches import generate_patch_grid


PATCH_QC_COLUMNS = [
    "image_id",
    "patch_id",
    "y0",
    "x0",
    "y1",
    "x1",
    "center_y",
    "center_x",
    "tissue_fraction",
    "intensity_mean",
    "intensity_std",
    "rms_contrast",
    "gradient_energy",
    "near_zero_fraction",
    "saturated_fraction",
    "valid_for_orientation",
    "valid_for_periodicity",
    "valid_for_spacing",
    "invalid_reason",
]


def compute_patch_qc(
    image: np.ndarray,
    tissue_mask: np.ndarray,
    image_id: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    if image.shape != tissue_mask.shape:
        raise ValueError(f"Image shape {image.shape} does not match tissue mask shape {tissue_mask.shape}")

    params = qc_params(config)
    rows: list[dict[str, object]] = []
    for patch in generate_patch_grid(image.shape, image_id, config):
        image_patch = image[patch.y0 : patch.y1, patch.x0 : patch.x1]
        mask_patch = tissue_mask[patch.y0 : patch.y1, patch.x0 : patch.x1]
        metrics = patch_quality_metrics(image_patch, mask_patch, params)
        flags = patch_validity(metrics, params)
        rows.append(
            {
                "image_id": patch.image_id,
                "patch_id": patch.patch_id,
                "y0": patch.y0,
                "x0": patch.x0,
                "y1": patch.y1,
                "x1": patch.x1,
                "center_y": patch.center_y,
                "center_x": patch.center_x,
                **metrics,
                **flags,
            }
        )

    return pd.DataFrame(rows, columns=PATCH_QC_COLUMNS)


def patch_quality_metrics(
    image_patch: np.ndarray,
    mask_patch: np.ndarray,
    params: dict[str, float],
) -> dict[str, float]:
    if image_patch.size == 0:
        return {
            "tissue_fraction": 0.0,
            "intensity_mean": float("nan"),
            "intensity_std": float("nan"),
            "rms_contrast": float("nan"),
            "gradient_energy": 0.0,
            "near_zero_fraction": 1.0,
            "saturated_fraction": 0.0,
        }

    values = np.asarray(image_patch, dtype=np.float32)
    tissue_fraction = float(np.mean(mask_patch))
    if np.any(mask_patch):
        measured = values[mask_patch]
    else:
        measured = values.ravel()

    intensity_mean = float(np.mean(measured))
    intensity_std = float(np.std(measured))
    rms_contrast = intensity_std
    gradient_energy = _gradient_energy(values, mask_patch)
    near_zero_fraction = float(np.mean(values <= params["near_zero_threshold"]))
    saturated_fraction = float(np.mean(values >= params["saturation_threshold"]))
    return {
        "tissue_fraction": tissue_fraction,
        "intensity_mean": intensity_mean,
        "intensity_std": intensity_std,
        "rms_contrast": rms_contrast,
        "gradient_energy": gradient_energy,
        "near_zero_fraction": near_zero_fraction,
        "saturated_fraction": saturated_fraction,
    }


def patch_validity(metrics: dict[str, float], params: dict[str, float]) -> dict[str, object]:
    reasons: list[str] = []
    if not np.isfinite(metrics["intensity_mean"]):
        reasons.append("empty_patch")
    if metrics["tissue_fraction"] < params["min_tissue_fraction"]:
        reasons.append("low_tissue_fraction")
    if not np.isfinite(metrics["rms_contrast"]) or metrics["rms_contrast"] < params["min_contrast"]:
        reasons.append("low_contrast")
    if metrics["gradient_energy"] < params["min_gradient_energy"]:
        reasons.append("low_gradient_energy")

    enough_tissue = "low_tissue_fraction" not in reasons and "empty_patch" not in reasons
    enough_contrast = "low_contrast" not in reasons and "empty_patch" not in reasons
    enough_gradient = "low_gradient_energy" not in reasons and "empty_patch" not in reasons

    valid_for_orientation = enough_tissue and enough_contrast and enough_gradient
    valid_for_periodicity = enough_tissue and enough_contrast
    valid_for_spacing = valid_for_periodicity
    return {
        "valid_for_orientation": bool(valid_for_orientation),
        "valid_for_periodicity": bool(valid_for_periodicity),
        "valid_for_spacing": bool(valid_for_spacing),
        "invalid_reason": "ok" if not reasons else ";".join(reasons),
    }


def qc_params(config: dict[str, Any]) -> dict[str, float]:
    defaults = {
        "min_tissue_fraction": 0.50,
        "min_contrast": 0.03,
        "min_gradient_energy": 0.0001,
        "near_zero_threshold": 0.02,
        "saturation_threshold": 0.98,
    }
    params = dict(defaults)
    params.update(config.get("qc", {}))
    parsed = {key: float(value) for key, value in params.items()}
    if not 0.0 <= parsed["min_tissue_fraction"] <= 1.0:
        raise ValueError("qc.min_tissue_fraction must be between 0 and 1")
    if parsed["min_contrast"] < 0:
        raise ValueError("qc.min_contrast must be non-negative")
    if parsed["min_gradient_energy"] < 0:
        raise ValueError("qc.min_gradient_energy must be non-negative")
    return parsed


def _gradient_energy(values: np.ndarray, mask: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    gy, gx = np.gradient(values.astype(np.float32, copy=False))
    energy = gx * gx + gy * gy
    if np.any(mask):
        return float(np.mean(energy[mask]))
    return float(np.mean(energy))
