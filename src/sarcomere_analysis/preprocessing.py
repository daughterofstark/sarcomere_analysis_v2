from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .config import output_dir


@dataclass(frozen=True)
class PreprocessingResult:
    image: np.ndarray
    metadata: dict[str, object]


def preprocess_image(raw: np.ndarray, config: dict[str, Any]) -> PreprocessingResult:
    params = preprocessing_params(config)
    image = np.asarray(raw).astype(np.float32, copy=False)

    lower_percentile = float(params["lower_percentile"])
    upper_percentile = float(params["upper_percentile"])
    lower_value, upper_value = np.percentile(image, [lower_percentile, upper_percentile])

    clipped = np.clip(image, lower_value, upper_value)
    scaled = _rescale_unit(clipped, lower_value, upper_value)

    corrected, background_metadata = subtract_background(scaled, params)

    if bool(params["enable_denoise"]):
        denoise_sigma = float(params["denoise_sigma_px"])
        if denoise_sigma > 0:
            corrected = ndi.gaussian_filter(corrected, sigma=denoise_sigma)

    corrected = np.clip(corrected, 0.0, 1.0)
    output_dtype = str(params["output_dtype"])
    if output_dtype != "float32":
        raise ValueError(f"Unsupported preprocessing.output_dtype: {output_dtype}")
    output = corrected.astype(np.float32, copy=False)

    metadata: dict[str, object] = {
        "input_dtype": str(raw.dtype),
        "output_dtype": str(output.dtype),
        "lower_percentile": lower_percentile,
        "upper_percentile": upper_percentile,
        "lower_clip_value": float(lower_value),
        "upper_clip_value": float(upper_value),
        "enable_denoise": bool(params["enable_denoise"]),
        "denoise_sigma_px": float(params["denoise_sigma_px"]),
    }
    metadata.update(background_metadata)
    return PreprocessingResult(image=output, metadata=metadata)


def preprocessing_params(config: dict[str, Any]) -> dict[str, object]:
    defaults: dict[str, object] = {
        "lower_percentile": 1.0,
        "upper_percentile": 99.8,
        "background_method": "gaussian",
        "background_sigma_px": 50.0,
        "enable_denoise": False,
        "denoise_sigma_px": 0.75,
        "output_dtype": "float32",
        "cache_preprocessed": False,
    }
    params = dict(defaults)
    params.update(config.get("preprocessing", {}))

    lower = float(params["lower_percentile"])
    upper = float(params["upper_percentile"])
    if not (0.0 <= lower < upper <= 100.0):
        raise ValueError("preprocessing percentiles must satisfy 0 <= lower < upper <= 100")

    method = str(params["background_method"]).lower()
    if method not in {"gaussian", "none"}:
        raise ValueError("preprocessing.background_method must be 'gaussian' or 'none'")
    params["background_method"] = method

    if float(params["background_sigma_px"]) < 0:
        raise ValueError("preprocessing.background_sigma_px must be non-negative")
    if float(params["denoise_sigma_px"]) < 0:
        raise ValueError("preprocessing.denoise_sigma_px must be non-negative")

    return params


def subtract_background(image: np.ndarray, params: dict[str, object]) -> tuple[np.ndarray, dict[str, object]]:
    method = str(params["background_method"]).lower()
    if method == "none":
        return image, {"background_method": "none", "background_sigma_px": 0.0}

    sigma = float(params["background_sigma_px"])
    if sigma <= 0:
        return image, {"background_method": "gaussian", "background_sigma_px": sigma}

    background = ndi.gaussian_filter(image, sigma=sigma)
    corrected = image - background
    corrected = _rescale_unit(corrected, float(np.min(corrected)), float(np.max(corrected)))
    return corrected, {"background_method": "gaussian", "background_sigma_px": sigma}


def save_preprocessed_npz(
    image_id: str,
    result: PreprocessingResult,
    config: dict[str, Any],
    subdir: str = "previews",
) -> Path:
    preview_dir = output_dir(config) / subdir
    preview_dir.mkdir(parents=True, exist_ok=True)
    path = preview_dir / f"{image_id}_preprocessed.npz"
    np.savez_compressed(path, image=result.image, metadata=result.metadata)
    return path


def _rescale_unit(image: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    span = max_value - min_value
    if not np.isfinite(span) or span <= 0:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - min_value) / span).astype(np.float32, copy=False)
