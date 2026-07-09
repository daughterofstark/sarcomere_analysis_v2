from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi


ORIENTATION_COLUMNS = [
    "patch_oop",
    "patch_mean_orientation_rad",
    "patch_mean_orientation_deg",
    "patch_orientation_weight_sum",
    "patch_orientation_valid_pixels",
]


@dataclass(frozen=True)
class OrientationResult:
    orientation_map: np.ndarray
    coherence_map: np.ndarray
    energy_map: np.ndarray
    image_metrics: dict[str, float | int]
    patch_metrics: pd.DataFrame


def compute_orientation_analysis(
    image: np.ndarray,
    tissue_mask: np.ndarray,
    patch_qc: pd.DataFrame,
    config: dict[str, Any],
) -> OrientationResult:
    params = orientation_params(config)
    orientation_map, coherence_map, energy_map = structure_tensor_orientation(image, params)
    weights = orientation_weights(energy_map, coherence_map, params["weight_mode"])

    image_valid = tissue_mask & np.isfinite(orientation_map) & np.isfinite(weights) & (weights > 0)
    image_oop, image_mean_rad, image_weight_sum, image_valid_pixels = axial_order_parameter(
        orientation_map,
        weights,
        image_valid,
        params["min_orientation_weight_sum"],
        int(params["min_orientation_valid_pixels"]),
    )

    patch_metrics = compute_patch_orientation_metrics(
        patch_qc,
        orientation_map,
        weights,
        tissue_mask,
        params,
    )
    heterogeneity = oop_heterogeneity(patch_metrics["patch_oop"].to_numpy(), str(params["heterogeneity_method"]))
    n_valid_patches = int(patch_metrics["patch_oop"].notna().sum())

    image_metrics: dict[str, float | int] = {
        "image_oop": image_oop,
        "image_mean_orientation_rad": image_mean_rad,
        "image_mean_orientation_deg": radians_to_degrees(image_mean_rad),
        "image_orientation_weight_sum": image_weight_sum,
        "image_orientation_valid_pixels": image_valid_pixels,
        "image_oop_heterogeneity": heterogeneity,
        "n_orientation_valid_patches": n_valid_patches,
    }
    return OrientationResult(
        orientation_map=orientation_map,
        coherence_map=coherence_map,
        energy_map=energy_map,
        image_metrics=image_metrics,
        patch_metrics=patch_metrics,
    )


def structure_tensor_orientation(
    image: np.ndarray,
    params: dict[str, float | str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D image for orientation analysis, got shape {values.shape}")

    iy, ix = np.gradient(values)
    sigma = float(params["tensor_sigma_px"])
    jxx = ndi.gaussian_filter(ix * ix, sigma=sigma)
    jxy = ndi.gaussian_filter(ix * iy, sigma=sigma)
    jyy = ndi.gaussian_filter(iy * iy, sigma=sigma)

    orientation = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    trace = jxx + jyy
    diff = jxx - jyy
    root = np.sqrt(np.maximum(diff * diff + 4.0 * jxy * jxy, 0.0))
    lambda1 = 0.5 * (trace + root)
    lambda2 = 0.5 * (trace - root)

    eps = float(params["eps"])
    coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + eps)
    coherence = np.clip(coherence, 0.0, 1.0)
    energy = np.maximum(lambda1 + lambda2, 0.0)

    return (
        orientation.astype(np.float32, copy=False),
        coherence.astype(np.float32, copy=False),
        energy.astype(np.float32, copy=False),
    )


def compute_patch_orientation_metrics(
    patch_qc: pd.DataFrame,
    orientation_map: np.ndarray,
    weights: np.ndarray,
    tissue_mask: np.ndarray,
    params: dict[str, float | str],
) -> pd.DataFrame:
    rows = []
    for _, row in patch_qc.iterrows():
        y0, y1 = int(row["y0"]), int(row["y1"])
        x0, x1 = int(row["x0"]), int(row["x1"])
        metrics = {
            "patch_oop": float("nan"),
            "patch_mean_orientation_rad": float("nan"),
            "patch_mean_orientation_deg": float("nan"),
            "patch_orientation_weight_sum": 0.0,
            "patch_orientation_valid_pixels": 0,
        }
        if bool(row["valid_for_orientation"]):
            patch_valid = (
                tissue_mask[y0:y1, x0:x1]
                & np.isfinite(orientation_map[y0:y1, x0:x1])
                & np.isfinite(weights[y0:y1, x0:x1])
                & (weights[y0:y1, x0:x1] > 0)
            )
            oop, mean_rad, weight_sum, valid_pixels = axial_order_parameter(
                orientation_map[y0:y1, x0:x1],
                weights[y0:y1, x0:x1],
                patch_valid,
                float(params["min_orientation_weight_sum"]),
                int(params["min_orientation_valid_pixels"]),
            )
            metrics = {
                "patch_oop": oop,
                "patch_mean_orientation_rad": mean_rad,
                "patch_mean_orientation_deg": radians_to_degrees(mean_rad),
                "patch_orientation_weight_sum": weight_sum,
                "patch_orientation_valid_pixels": valid_pixels,
            }
        rows.append(metrics)

    orientation_df = pd.DataFrame(rows, columns=ORIENTATION_COLUMNS)
    return pd.concat([patch_qc.reset_index(drop=True), orientation_df], axis=1)


def axial_order_parameter(
    theta: np.ndarray,
    weights: np.ndarray,
    valid_mask: np.ndarray,
    min_weight_sum: float,
    min_valid_pixels: int,
) -> tuple[float, float, float, int]:
    valid = valid_mask & np.isfinite(theta) & np.isfinite(weights) & (weights > 0)
    valid_pixels = int(np.count_nonzero(valid))
    if valid_pixels < min_valid_pixels:
        return float("nan"), float("nan"), 0.0, valid_pixels

    selected_weights = weights[valid].astype(np.float64, copy=False)
    weight_sum = float(np.sum(selected_weights))
    if not np.isfinite(weight_sum) or weight_sum < min_weight_sum:
        return float("nan"), float("nan"), weight_sum, valid_pixels

    selected_theta = theta[valid].astype(np.float64, copy=False)
    axial_vectors = selected_weights * np.exp(2j * selected_theta)
    mean_vector = np.sum(axial_vectors) / weight_sum
    oop = float(np.abs(mean_vector))
    mean_rad = normalize_axial_angle(0.5 * float(np.angle(mean_vector)))
    return float(np.clip(oop, 0.0, 1.0)), mean_rad, weight_sum, valid_pixels


def orientation_weights(energy: np.ndarray, coherence: np.ndarray, weight_mode: str) -> np.ndarray:
    mode = str(weight_mode).lower()
    if mode == "energy":
        weights = energy
    elif mode == "coherence":
        weights = coherence
    elif mode == "energy_x_coherence":
        weights = energy * coherence
    else:
        raise ValueError("orientation.weight_mode must be energy, coherence, or energy_x_coherence")
    weights = np.where(np.isfinite(weights), weights, 0.0)
    return np.maximum(weights, 0.0).astype(np.float32, copy=False)


def oop_heterogeneity(values: np.ndarray, method: str) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    if method == "std":
        return float(np.std(finite))
    if method == "iqr":
        q75, q25 = np.percentile(finite, [75, 25])
        return float(q75 - q25)
    raise ValueError("orientation.heterogeneity_method must be 'std' or 'iqr'")


def orientation_params(config: dict[str, Any]) -> dict[str, float | str]:
    defaults: dict[str, float | str] = {
        "tensor_sigma_px": 2.0,
        "weight_mode": "energy_x_coherence",
        "min_orientation_weight_sum": 0.001,
        "min_orientation_valid_pixels": 64,
        "heterogeneity_method": "std",
        "eps": 1.0e-12,
    }
    params = dict(defaults)
    params.update(config.get("orientation", {}))
    params["weight_mode"] = str(params["weight_mode"]).lower()
    params["heterogeneity_method"] = str(params["heterogeneity_method"]).lower()

    if float(params["tensor_sigma_px"]) < 0:
        raise ValueError("orientation.tensor_sigma_px must be non-negative")
    if params["weight_mode"] not in {"energy", "coherence", "energy_x_coherence"}:
        raise ValueError("orientation.weight_mode must be energy, coherence, or energy_x_coherence")
    if float(params["min_orientation_weight_sum"]) < 0:
        raise ValueError("orientation.min_orientation_weight_sum must be non-negative")
    if int(params["min_orientation_valid_pixels"]) < 1:
        raise ValueError("orientation.min_orientation_valid_pixels must be positive")
    if params["heterogeneity_method"] not in {"std", "iqr"}:
        raise ValueError("orientation.heterogeneity_method must be 'std' or 'iqr'")
    if float(params["eps"]) <= 0:
        raise ValueError("orientation.eps must be positive")
    return params


def normalize_axial_angle(angle: float) -> float:
    return float(((angle + np.pi / 2.0) % np.pi) - np.pi / 2.0)


def radians_to_degrees(angle: float) -> float:
    if not np.isfinite(angle):
        return float("nan")
    return float(np.degrees(angle))
